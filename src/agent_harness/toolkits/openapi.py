"""An OpenAPI document, as tools: one for every operation the API offers.

    from agent_harness import Agent, openapi_tools

    api = openapi_tools("openapi.json", token=os.environ["SHOP_TOKEN"])
    agent = Agent("support", "Help with orders.", tools=api)

    api.names                                    # ['getOrder', 'listOrders', 'refundOrder']
    await api.call("getOrder", orderId="4182")   # call one yourself, to see it work

The document can be a file (JSON or YAML), a URL, the text itself, or a mapping
already loaded. OpenAPI 3.0 and 3.1 are read as they are; Swagger 2.0 is read
too.

Each operation becomes a tool whose arguments are the operation's own: its path,
query and header parameters, and the fields of its request body. The model sees
names, types and descriptions taken from the document; what it never sees is the
credential — that is added to the request here.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import re
from collections.abc import Callable, Iterable, Iterator
from fnmatch import fnmatch
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urljoin

from ..errors import ConfigurationError, ToolError
from ..tools import Permission, Tool

__all__ = ["OpenAPIToolkit", "openapi_tools", "toolkits_from_config"]

METHODS = ("get", "put", "post", "delete", "patch", "head", "options")
_READS = {"get", "head", "options"}
#: Headers the document may list as parameters but which are not the model's to set.
_OWN_HEADERS = {"accept", "content-type", "authorization"}
#: What is kept of a schema. Everything else is noise to a model, or something
#: one provider or another refuses.
_KEEP = ("type", "description", "enum", "const", "format", "default", "minimum",
         "maximum", "minLength", "maxLength", "pattern", "minItems", "maxItems")
_NAME = re.compile(r"[^A-Za-z0-9_-]+")
_RETRY_STATUS = {429, 502, 503, 504}
_MAX_DEPTH = 8


# ----------------------------------------------------------------------
# reading the document
# ----------------------------------------------------------------------
def _parse(text: str, where: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except ValueError:
        import yaml

        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigurationError(
                f"{where} is neither JSON nor YAML: {exc}") from None
    if not isinstance(data, dict):
        raise ConfigurationError(f"{where} is not an OpenAPI document")
    return data


def _load(source: Any, transport: Any = None) -> tuple[dict[str, Any], str]:
    """The document, and the URL it came from if it came from one."""
    if isinstance(source, dict):
        return source, ""
    if isinstance(source, Path):
        source = str(source)
    if not isinstance(source, str) or not source.strip():
        raise ConfigurationError(
            "an OpenAPI document is a file, a URL, its text, or a mapping — got "
            f"{type(source).__name__}")
    text = source.strip()
    if text.startswith(("http://", "https://")):
        import httpx

        try:
            with httpx.Client(timeout=30.0, follow_redirects=True,
                              transport=transport) as client:
                response = client.get(text, headers={"accept": "application/json, */*"})
        except httpx.HTTPError as exc:
            raise ConfigurationError(f"{text} could not be fetched: {exc}") from None
        if response.status_code >= 400:
            raise ConfigurationError(f"{text} returned HTTP {response.status_code}")
        return _parse(response.text, text), text
    if text.startswith("{") or "\n" in text:
        return _parse(text, "the document"), ""
    file = Path(text).expanduser()
    if not file.is_file():
        raise ConfigurationError(f"no OpenAPI document at {file}")
    return _parse(file.read_text(encoding="utf-8"), str(file)), ""


class _Document:
    """An OpenAPI document, with its references followed."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        if not isinstance(data.get("paths"), dict) or not (
                data.get("openapi") or data.get("swagger")):
            raise ConfigurationError(
                "this is not an OpenAPI document: it needs `openapi` (or "
                "`swagger`) and `paths`")
        self.swagger = bool(data.get("swagger")) and not data.get("openapi")

    def resolve(self, node: Any, seen: tuple[str, ...] = ()) -> tuple[Any, tuple[str, ...]]:
        """Follow `$ref` until there is something to read. A reference that
        comes back round to itself ends as an empty object."""
        while isinstance(node, dict) and "$ref" in node:
            ref = str(node["$ref"])
            if ref in seen or len(seen) > 40:
                return {"type": "object"}, seen
            if not ref.startswith("#/"):
                return {"description": f"(defined elsewhere: {ref})"}, seen
            target: Any = self.data
            for part in ref[2:].split("/"):
                key = part.replace("~1", "/").replace("~0", "~")
                if isinstance(target, dict) and key in target:
                    target = target[key]
                else:
                    raise ConfigurationError(
                        f"the document refers to {ref}, which is not in it")
            node, seen = target, (*seen, ref)
        return node, seen

    def schema(self, node: Any, seen: tuple[str, ...] = (), depth: int = 0, *,
               request: bool = True) -> dict[str, Any]:
        """A schema as a model should be shown it: references followed, `allOf`
        merged, and only the keywords that describe a value kept."""
        node, seen = self.resolve(node, seen)
        if not isinstance(node, dict) or depth > _MAX_DEPTH:
            return {}
        if "allOf" in node:
            merged = self._merged(node, seen)
            node = {**{k: v for k, v in node.items() if k != "allOf"},
                    **{k: v for k, v in merged.items() if v}}

        out: dict[str, Any] = {k: node[k] for k in _KEEP if k in node}
        if isinstance(out.get("description"), str):
            out["description"] = " ".join(out["description"].split())[:500]
        for key in ("anyOf", "oneOf"):
            if isinstance(node.get(key), list):
                options = [self.schema(o, seen, depth + 1, request=request)
                           for o in node[key]]
                out["anyOf"] = [o for o in options if o] or [{}]
        properties = node.get("properties")
        if isinstance(properties, dict):
            out.setdefault("type", "object")
            kept = {}
            for name, prop in properties.items():
                resolved, _ = self.resolve(prop, seen)
                if request and isinstance(resolved, dict) and resolved.get("readOnly"):
                    continue          # the server sets it; the caller cannot
                kept[name] = self.schema(prop, seen, depth + 1, request=request)
            out["properties"] = kept
            required = [r for r in node.get("required") or [] if r in kept]
            if required:
                out["required"] = list(dict.fromkeys(required))
        if "items" in node:
            out.setdefault("type", "array")
            out["items"] = self.schema(node["items"], seen, depth + 1, request=request)
        extra = node.get("additionalProperties")
        if isinstance(extra, dict):
            out.setdefault("type", "object")
            out["additionalProperties"] = self.schema(extra, seen, depth + 1,
                                                      request=request)
        return out

    def _merged(self, node: dict[str, Any], seen: tuple[str, ...]) -> dict[str, Any]:
        """`allOf` inside `allOf`: flattened one level at a time."""
        out: dict[str, Any] = {"properties": {}, "required": []}
        for part in node.get("allOf") or []:
            piece, _ = self.resolve(part, seen)
            if isinstance(piece, dict):
                if "allOf" in piece:
                    piece = self._merged(piece, seen)
                out["properties"].update(piece.get("properties") or {})
                out["required"] += piece.get("required") or []
                for key in ("type", "description", "enum", "format", "items"):
                    if key in piece:
                        out.setdefault(key, piece[key])
        out["properties"].update(node.get("properties") or {})
        out["required"] += node.get("required") or []
        return out

    # ---- where the API is ---------------------------------------------------
    def server(self, origin: str) -> str:
        data = self.data
        if self.swagger:
            host = data.get("host")
            if not host:
                return origin.rsplit("/", 1)[0] if origin else ""
            scheme = (data.get("schemes") or ["https"])[0]
            return f"{scheme}://{host}{data.get('basePath') or ''}"
        servers = data.get("servers") or []
        if not servers or not isinstance(servers[0], dict):
            return origin.rsplit("/", 1)[0] if origin else ""
        url = str(servers[0].get("url") or "")
        for name, variable in (servers[0].get("variables") or {}).items():
            url = url.replace("{" + name + "}", str((variable or {}).get("default", "")))
        # "/v1" means "on whatever host the document came from".
        return urljoin(origin, url) if origin and not url.startswith("http") else url

    def security_schemes(self) -> dict[str, dict[str, Any]]:
        if self.swagger:
            found = self.data.get("securityDefinitions") or {}
            out = {}
            for name, scheme in found.items():
                scheme = dict(scheme or {})
                if scheme.get("type") == "basic":
                    scheme = {"type": "http", "scheme": "basic"}
                out[name] = scheme
            return out
        found = (self.data.get("components") or {}).get("securitySchemes") or {}
        return {name: self.resolve(scheme)[0] for name, scheme in found.items()}


# ----------------------------------------------------------------------
# one operation
# ----------------------------------------------------------------------
class _Field:
    """One argument of a tool, and where in the request it goes."""

    __slots__ = ("name", "wire", "where", "required", "explode", "style", "encoded")

    def __init__(self, name: str, wire: str, where: str, *, required: bool = False,
                 explode: bool = True, style: str = "", encoded: bool = False) -> None:
        self.name, self.wire, self.where = name, wire, where
        self.required, self.explode, self.style = required, explode, style
        self.encoded = encoded          # sent as JSON text, as `content` asks


class _Operation:
    """An operation of the API, read once, called many times."""

    def __init__(self, *, method: str, path: str, operation_id: str, summary: str,
                 tags: list[str]) -> None:
        self.method, self.path = method, path
        self.operation_id, self.summary, self.tags = operation_id, summary, tags
        self.fields: list[_Field] = []
        self.properties: dict[str, Any] = {}
        self.body_kind = ""             # json · form · multipart · text
        self.media_type = ""
        self.whole_body = False         # the body is one argument, not its fields
        self.deep: set[str] = set()     # form fields the document wants as a[b]=c

    def add(self, field: _Field, schema: dict[str, Any]) -> None:
        self.fields.append(field)
        self.properties[field.name] = schema

    @property
    def parameters(self) -> dict[str, Any]:
        required = [f.name for f in self.fields if f.required]
        return {"type": "object", "properties": self.properties,
                **({"required": required} if required else {})}


def _argument(name: str) -> str:
    """A parameter's name as an argument a model can name: `filter[status]` and
    `X-Request-Id` are both fine on the wire and both awkward in a tool call."""
    cleaned = _NAME.sub("_", name).strip("_")
    return cleaned or "value"


class OpenAPIToolkit:
    """The operations of an API, as tools.

    Args:
        spec: the OpenAPI document — a file, a URL, its text, or a mapping.
        base_url: where the API is. Left out, the document's own `servers`.
        token: a bearer token, sent as `Authorization: Bearer …`. A function is
            called for each request, so a token that expires can be renewed.
        api_key: the key for the document's `apiKey` security scheme, sent in
            the header, query or cookie that scheme names.
        basic: a `(user, password)` pair.
        credentials: scheme name → secret, for a document with several schemes.
        headers, params: sent with every request. A parameter fixed here is not
            offered to the model.
        include, exclude: which operations become tools — globs matched against
            the tool's name, its `operationId`, or `"GET /orders/{id}"`.
        tags, methods: only operations with one of these tags, or these methods.
        deprecated: include operations the document marks deprecated.
        writes: the permission of a tool that changes something (anything but
            GET, HEAD, OPTIONS): "allow", "ask" (needs an approver), or "deny".
        prefix: put in front of every tool name — to keep two APIs apart.
        name: what to call this API; the tag its tools carry. Left out, the
            document's title.
        timeout, retries: per request. Only reads are retried.
        max_response_chars: how much of a response the model is shown.
        cache_reads: let the harness cache what GETs return.
        transport, http: an `httpx` transport or client of your own.
    """

    def __init__(self, spec: Any, *, base_url: str | None = None,
                 token: str | Callable[[], Any] | None = None,
                 api_key: str | None = None, basic: tuple[str, str] | None = None,
                 credentials: dict[str, str] | None = None,
                 headers: dict[str, Any] | None = None,
                 params: dict[str, Any] | None = None,
                 include: Iterable[str] | None = None,
                 exclude: Iterable[str] | None = None,
                 tags: Iterable[str] | None = None,
                 methods: Iterable[str] | None = None, deprecated: bool = False,
                 writes: Permission = "allow", prefix: str = "", name: str | None = None,
                 timeout: float = 30.0, retries: int = 2,
                 max_response_chars: int = 40_000, cache_reads: bool = False,
                 transport: Any = None, http: Any = None) -> None:
        data, origin = _load(spec, transport)
        self.document = _Document(data)
        info = data.get("info") or {}
        self.title = str(info.get("title") or "API")
        self.name = _argument(name or self.title).lower()
        self.base_url = (base_url or self.document.server(origin)).rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise ConfigurationError(
                f"{self.title}: the document does not say where the API is "
                f"(servers: {self.base_url or 'none'}) — pass base_url=\"https://…\"")
        if writes not in ("allow", "ask", "deny"):
            raise ConfigurationError(
                f"writes must be allow, ask or deny — got {writes!r}")
        self.timeout, self.retries = timeout, max(0, retries)
        self.max_response_chars = max_response_chars
        self._transport, self._http, self._owns_http = transport, http, http is None

        self._headers: dict[str, Any] = dict(headers or {})
        self._params: dict[str, Any] = dict(params or {})
        self._cookies: dict[str, Any] = {}
        self._authorise(token, api_key, basic, credentials or {})

        #: Operations that could not be made into tools, and why.
        self.skipped: list[tuple[str, str]] = []
        self.operations: dict[str, _Operation] = {}
        self.tools: list[Tool] = []
        wanted_tags = set(tags) if tags is not None else None
        wanted_methods = ({m.lower() for m in methods} if methods is not None
                          else set(METHODS))
        include = list(include) if include is not None else None
        exclude = list(exclude or [])

        for path, item in data["paths"].items():
            item, _ = self.document.resolve(item)
            if not isinstance(item, dict):
                continue
            for method in METHODS:
                raw = item.get(method)
                if not isinstance(raw, dict) or method not in wanted_methods:
                    continue
                label = f"{method.upper()} {path}"
                if raw.get("deprecated") and not deprecated:
                    self.skipped.append((label, "deprecated"))
                    continue
                if wanted_tags is not None and not wanted_tags & set(raw.get("tags") or []):
                    continue
                tool_name = self._tool_name(prefix, raw.get("operationId"), method, path)
                keys = (tool_name, str(raw.get("operationId") or ""), label)
                if include is not None and not any(
                        fnmatch(k, p) for k in keys for p in include):
                    continue
                if any(fnmatch(k, p) for k in keys for p in exclude):
                    continue
                try:
                    operation = self._read(method, path, item, raw)
                except _Unsupported as exc:
                    self.skipped.append((label, str(exc)))
                    continue
                self.operations[tool_name] = operation
                self.tools.append(self._tool(tool_name, operation, raw, writes,
                                             cache_reads))

    # ---- credentials ---------------------------------------------------------
    def _authorise(self, token: Any, api_key: str | None,
                   basic: tuple[str, str] | None, credentials: dict[str, str]) -> None:
        schemes = self.document.security_schemes()
        if token is not None:
            self._headers["Authorization"] = (
                (lambda: _bearer(token)) if callable(token) else f"Bearer {token}")
        if basic is not None:
            self._headers["Authorization"] = "Basic " + base64.b64encode(
                f"{basic[0]}:{basic[1]}".encode()).decode()
        if api_key is not None:
            names = [n for n, s in schemes.items() if s.get("type") == "apiKey"]
            if not names:
                raise ConfigurationError(
                    f"{self.title}: api_key= was given, but the document declares "
                    "no apiKey security scheme to say where it goes — pass it as "
                    "headers={\"X-API-Key\": …} or params={\"api_key\": …} instead")
            credentials = {names[0]: api_key, **credentials}
        for scheme_name, secret in credentials.items():
            scheme = schemes.get(scheme_name)
            if scheme is None:
                raise ConfigurationError(
                    f"{self.title}: no security scheme called {scheme_name!r}; the "
                    f"document declares: {', '.join(sorted(schemes)) or 'none'}")
            kind = scheme.get("type")
            if kind == "apiKey":
                target = {"header": self._headers, "query": self._params,
                          "cookie": self._cookies}.get(scheme.get("in", "header"))
                target[scheme.get("name") or scheme_name] = secret
            elif kind == "http" and str(scheme.get("scheme", "")).lower() == "basic":
                self._headers["Authorization"] = "Basic " + base64.b64encode(
                    secret.encode()).decode()
            else:
                # http bearer, oauth2, openIdConnect: a token, sent the same way.
                self._headers["Authorization"] = f"Bearer {secret}"

    # ---- reading an operation ----------------------------------------------
    def _tool_name(self, prefix: str, operation_id: Any, method: str, path: str) -> str:
        base = _NAME.sub("_", str(operation_id)).strip("_") if operation_id else ""
        if not base:
            words = [w for w in re.split(r"[^A-Za-z0-9]+", path) if w]
            base = "_".join([method, *words]) or method
        name = f"{prefix}{base}"[:64].rstrip("_")
        candidate, n = name, 2
        while candidate in self.operations:
            suffix = f"_{n}"
            candidate, n = f"{name[:64 - len(suffix)]}{suffix}", n + 1
        return candidate

    def _read(self, method: str, path: str, item: dict[str, Any],
              raw: dict[str, Any]) -> _Operation:
        doc = self.document
        operation = _Operation(
            method=method, path=path, operation_id=str(raw.get("operationId") or ""),
            summary=str(raw.get("summary") or ""), tags=list(raw.get("tags") or []))
        fixed_headers = {h.lower() for h in self._headers}
        taken: set[str] = set()

        def claim(wanted: str, where: str) -> str:
            name = wanted if wanted not in taken else f"{wanted}_{where}"
            n = 2
            while name in taken:
                name, n = f"{wanted}_{where}_{n}", n + 1
            taken.add(name)
            return name

        # The operation's own parameters replace the path's of the same name.
        merged: dict[tuple[str, str], dict[str, Any]] = {}
        for entry in [*(item.get("parameters") or []), *(raw.get("parameters") or [])]:
            param, _ = doc.resolve(entry)
            if isinstance(param, dict) and param.get("name") and param.get("in"):
                merged[(str(param["name"]), str(param["in"]))] = param

        body_schema: Any = None
        body_required = False
        form_fields: list[dict[str, Any]] = []
        for (wire, where), param in merged.items():
            if where == "body":                       # Swagger 2
                body_schema = param.get("schema") or {}
                operation.body_kind, operation.media_type = "json", "application/json"
                body_required = bool(param.get("required"))
                continue
            if where == "formData":                   # Swagger 2
                form_fields.append(param)
                continue
            if where not in ("path", "query", "header", "cookie"):
                continue
            if where == "header" and (wire.lower() in _OWN_HEADERS
                                      or wire.lower() in fixed_headers):
                continue
            if where == "query" and wire in self._params:
                continue
            if where == "cookie" and wire in self._cookies:
                continue
            encoded = False
            if isinstance(param.get("content"), dict) and param["content"]:
                media = next(iter(param["content"].values())) or {}
                schema = doc.schema(media.get("schema"))
                encoded = True
            elif "schema" in param:
                schema = doc.schema(param["schema"])
            else:                                     # Swagger 2: the type is inline
                schema = doc.schema({k: v for k, v in param.items()
                                     if k in (*_KEEP, "items")})
            if param.get("description") and "description" not in schema:
                schema["description"] = " ".join(str(param["description"]).split())[:500]
            style = str(param.get("style") or ("form" if where in ("query", "cookie")
                                               else "simple"))
            explode = param.get("explode", style == "form")
            if param.get("collectionFormat") == "multi":
                explode = True
            elif "collectionFormat" in param:
                explode = False
            operation.add(_Field(claim(_argument(wire), where), wire, where,
                                 required=bool(param.get("required")) or where == "path",
                                 explode=bool(explode), style=style, encoded=encoded),
                          schema or {"type": "string"})

        for name in re.findall(r"\{([^}]+)\}", path):
            if not any(f.where == "path" and f.wire == name for f in operation.fields):
                # In the path and not declared: still has to be filled in.
                operation.add(_Field(claim(_argument(name), "path"), name, "path",
                                     required=True), {"type": "string"})

        if form_fields:
            consumes = raw.get("consumes") or self.document.data.get("consumes") or []
            operation.body_kind = "multipart" if "multipart/form-data" in consumes \
                else "form"
            body_schema = {"type": "object", "properties": {
                f["name"]: {k: v for k, v in f.items() if k in (*_KEEP, "items")}
                for f in form_fields},
                "required": [f["name"] for f in form_fields if f.get("required")]}
        request_body, _ = doc.resolve(raw.get("requestBody"))
        if isinstance(request_body, dict) and isinstance(request_body.get("content"), dict):
            content = request_body["content"]
            body_required = bool(request_body.get("required"))
            chosen = next((m for m in content if "json" in m.lower()), None)
            kind = "json"
            if chosen is None:
                for wanted, label in (("application/x-www-form-urlencoded", "form"),
                                      ("multipart/form-data", "multipart"),
                                      ("text/", "text")):
                    chosen = next((m for m in content if m.lower().startswith(wanted)),
                                  None)
                    if chosen is not None:
                        kind = label
                        break
            if chosen is None:
                raise _Unsupported(
                    f"its request body is {', '.join(content) or 'undeclared'}, "
                    "which cannot be sent from a tool call")
            operation.body_kind, operation.media_type = kind, chosen
            body_schema = (content[chosen] or {}).get("schema") or {}
            operation.deep = {
                name for name, how in ((content[chosen] or {}).get("encoding") or {}).items()
                if isinstance(how, dict) and how.get("style") == "deepObject"}

        if body_schema is not None:
            self._body(operation, body_schema, body_required, claim)
        return operation

    def _body(self, operation: _Operation, raw_schema: Any, required: bool,
              claim: Callable[[str, str], str]) -> None:
        """The request body, as arguments. An object's fields become arguments of
        their own, which is what a model fills in best; anything else is one
        argument called `body`."""
        doc = self.document
        resolved, seen = doc.resolve(raw_schema)
        schema = doc.schema(raw_schema)
        properties = schema.get("properties") or {}
        binary = [name for name, prop in (
            (resolved.get("properties") or {}) if isinstance(resolved, dict) else {}
        ).items() if _is_file(doc.resolve(prop, seen)[0])]
        for name in binary:
            if name in (schema.get("required") or []):
                raise _Unsupported(f"it needs a file upload ({name})")
            properties.pop(name, None)

        flat = (schema.get("type") == "object" and properties
                and "additionalProperties" not in schema and "anyOf" not in schema)
        if operation.body_kind == "text" or not flat:
            if operation.body_kind in ("form", "multipart"):
                # A form with no fields declared: there is nothing to fill in,
                # and the operation is called without a body.
                operation.body_kind = ""
                return
            operation.whole_body = True
            if operation.body_kind == "text":
                schema = {"type": "string", **{k: v for k, v in schema.items()
                                               if k == "description"}}
            schema.setdefault("description", "The request body.")
            operation.add(_Field(claim("body", "body"), "body", "body",
                                 required=required), schema)
            return
        needed = set(schema.get("required") or [])
        for name, prop in properties.items():
            operation.add(_Field(claim(_argument(name), "body"), name, "body",
                                 required=name in needed), prop or {})

    def _tool(self, name: str, operation: _Operation, raw: dict[str, Any],
              writes: Permission, cache_reads: bool) -> Tool:
        toolkit = self

        async def call_api(**arguments: Any) -> Any:
            return await toolkit._call(name, operation, arguments)

        call_api.__name__ = name
        summary = operation.summary.strip()
        detail = " ".join(str(raw.get("description") or "").split())
        said = summary if not detail or detail == summary else (
            f"{summary}. {detail}" if summary else detail)
        description = (f"{said[:900].rstrip('.')}. " if said else "") + (
            f"({operation.method.upper()} {operation.path})")
        reads = operation.method in _READS
        tool = Tool(
            call_api, name=name, description=description,
            parameters=operation.parameters,
            permission="allow" if reads else writes,
            cacheable=cache_reads and operation.method == "get",
            max_output_chars=self.max_response_chars,
            tags=["openapi", self.name, "read" if reads else "write",
                  *(_argument(t).lower() for t in operation.tags)],
        )
        tool.operation = {"method": operation.method.upper(), "path": operation.path,  # type: ignore[attr-defined]
                          "operation_id": operation.operation_id, "api": self.title}
        return tool

    # ---- calling -------------------------------------------------------------
    @property
    def http(self) -> Any:
        if self._http is None:
            import httpx

            # Redirects are not followed: a credential must not go to wherever
            # a response points.
            self._http = httpx.AsyncClient(timeout=self.timeout,
                                           transport=self._transport,
                                           follow_redirects=False)
        return self._http

    async def _call(self, name: str, operation: _Operation,
                    arguments: dict[str, Any]) -> Any:
        """Build the request the document describes, send it, read the answer."""
        known = {f.name: f for f in operation.fields}
        unknown = sorted(set(arguments) - set(known))
        if unknown:
            raise ToolError(
                f"{name} does not take {', '.join(unknown)}; it takes "
                f"{', '.join(known) or 'no arguments'}", tool=name)
        given = {k: v for k, v in arguments.items() if v is not None}
        missing = [f.name for f in operation.fields if f.required and f.name not in given]
        if missing:
            raise ToolError(f"{name} needs {', '.join(missing)}", tool=name)

        path = operation.path
        query: list[tuple[str, str]] = []
        headers: dict[str, str] = {"accept": "application/json, text/plain;q=0.9, */*;q=0.5"}
        cookies: dict[str, str] = {}
        body: dict[str, Any] = {}
        whole: Any = None
        for key, value in given.items():
            field = known[key]
            if field.where == "path":
                path = path.replace("{" + field.wire + "}",
                                    quote(_plain(value, ","), safe=""))
            elif field.where == "query":
                query.extend(_query(field, value))
            elif field.where == "header":
                headers[field.wire] = _plain(value, ",")
            elif field.where == "cookie":
                cookies[field.wire] = _plain(value, ",")
            elif operation.whole_body:
                whole = value
            else:
                body[field.wire] = value
        for key, value in self._params.items():
            query.append((key, str(await _secret(value))))
        for key, value in self._headers.items():
            headers[key] = str(await _secret(value))
        for key, value in self._cookies.items():
            cookies[key] = str(await _secret(value))

        request: dict[str, Any] = {"params": query, "headers": headers}
        if cookies:
            request["cookies"] = cookies
        payload = whole if operation.whole_body else (body or None)
        if operation.body_kind and payload is not None:
            if operation.body_kind == "json":
                request["json"] = payload
                if operation.media_type and operation.media_type != "application/json":
                    request["content"] = json.dumps(payload).encode()
                    request.pop("json")
                    headers["content-type"] = operation.media_type
            elif operation.body_kind == "form":
                request["content"] = urlencode(_form(payload or {}, operation.deep)).encode()
                headers["content-type"] = "application/x-www-form-urlencoded"
            elif operation.body_kind == "multipart":
                request["files"] = {k: (None, _plain(v, ","))
                                    for k, v in (payload or {}).items()}
            else:
                request["content"] = str(payload).encode()
                headers["content-type"] = operation.media_type or "text/plain"

        import httpx

        url = f"{self.base_url}{path}"
        method = operation.method.upper()
        attempts = self.retries + 1 if operation.method in _READS else 1
        for attempt in range(attempts):
            last = attempt == attempts - 1
            try:
                response = await self.http.request(method, url, **request)
            except httpx.HTTPError as exc:
                if last:
                    raise ToolError(
                        f"{method} {operation.path} could not be reached: "
                        f"{type(exc).__name__}: {exc}", tool=name) from None
            else:
                if response.status_code not in _RETRY_STATUS or last:
                    return self._answer(name, operation, response)
            await asyncio.sleep(min(0.3 * 2 ** attempt, 5.0))
        raise ToolError(f"{method} {operation.path} did not answer", tool=name)

    def _answer(self, name: str, operation: _Operation, response: Any) -> Any:
        status = response.status_code
        kind = response.headers.get("content-type", "").lower()
        label = f"{operation.method.upper()} {operation.path}"
        if 300 <= status < 400:
            raise ToolError(
                f"{label} answered {status}, a redirect to "
                f"{response.headers.get('location', 'elsewhere')}, which is not "
                "followed", tool=name)
        if status >= 400:
            detail = " ".join(response.text.split())[:600]
            raise ToolError(f"{label} returned {status}"
                            + (f": {detail}" if detail else ""), tool=name)
        if status == 204 or not response.content:
            return f"{label} succeeded ({status}, no content)."
        if "json" in kind:
            try:
                return response.json()
            except ValueError:
                return response.text
        if kind.startswith("text/") or any(w in kind for w in ("xml", "yaml", "csv",
                                                               "javascript")) or not kind:
            return response.text
        return (f"{label} returned {len(response.content)} bytes of "
                f"{kind.split(';')[0]}, which is not text.")

    async def call(self, name: str, /, **arguments: Any) -> Any:
        """Call one operation yourself — the way to check an API is wired up
        before an agent is trusted with it. Raises `ToolError` on a failure."""
        if name not in self.operations:
            raise ToolError(f"no operation {name!r}; there are: "
                            f"{', '.join(self.names) or 'none'}", tool=name)
        return await self._call(name, self.operations[name], arguments)

    # ---- the toolkit as a collection ------------------------------------------
    @property
    def names(self) -> list[str]:
        return [tool.name for tool in self.tools]

    def get(self, name: str) -> Tool:
        for tool in self.tools:
            if tool.name == name:
                return tool
        raise ToolError(f"no operation {name!r}; there are: "
                        f"{', '.join(self.names) or 'none'}", tool=name)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self.tools)

    def __len__(self) -> int:
        return len(self.tools)

    def __contains__(self, name: object) -> bool:
        return name in self.operations

    def describe(self) -> str:
        """Every tool, the request behind it, and what it takes."""
        lines = [f"{self.title} — {self.base_url} — {len(self.tools)} tools"]
        for tool in self.tools:
            operation = self.operations[tool.name]
            arguments = ", ".join(
                f.name + ("" if f.required else "?") for f in operation.fields)
            mark = "" if tool.permission == "allow" else f" [{tool.permission}]"
            lines.append(f"  {tool.name}({arguments}){mark}")
            lines.append(f"      {operation.method.upper()} {operation.path}"
                         + (f" — {operation.summary}" if operation.summary else ""))
        for label, reason in self.skipped:
            lines.append(f"  (skipped {label}: {reason})")
        return "\n".join(lines)

    async def aclose(self) -> None:
        if self._http is not None and self._owns_http:
            await self._http.aclose()
        self._http = None

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        return f"<OpenAPIToolkit {self.title} tools={len(self.tools)}>"


class _Unsupported(Exception):
    """An operation that cannot be a tool. It is skipped, and the reason kept."""


def _is_file(schema: Any) -> bool:
    return isinstance(schema, dict) and (
        schema.get("format") in ("binary", "base64") or schema.get("type") == "file")


def _plain(value: Any, separator: str) -> str:
    """A value as it is written in a URL or a header."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return separator.join(_plain(v, separator) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _query(field: _Field, value: Any) -> list[tuple[str, str]]:
    """One query parameter, serialised the way the document says."""
    if field.encoded:
        return [(field.wire, json.dumps(value, separators=(",", ":")))]
    if isinstance(value, (list, tuple)):
        if field.explode:
            return [(field.wire, _plain(v, ",")) for v in value]
        separator = {"spaceDelimited": " ", "pipeDelimited": "|"}.get(field.style, ",")
        return [(field.wire, separator.join(_plain(v, ",") for v in value))]
    if isinstance(value, dict):
        if field.style == "deepObject":
            return [(f"{field.wire}[{k}]", _plain(v, ",")) for k, v in value.items()]
        if field.explode:
            return [(str(k), _plain(v, ",")) for k, v in value.items()]
        return [(field.wire, ",".join(f"{k},{_plain(v, ',')}" for k, v in value.items()))]
    return [(field.wire, _plain(value, ","))]


def _form(fields: dict[str, Any], deep: set[str]) -> list[tuple[str, str]]:
    """A form body. A list is the field repeated; an object — or anything the
    document marks `deepObject` — is written `a[b]=c`, which is what APIs that
    take nested forms expect."""
    pairs: list[tuple[str, str]] = []

    def nested(key: str, value: Any) -> None:
        if isinstance(value, dict):
            for inner, item in value.items():
                nested(f"{key}[{inner}]", item)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                nested(f"{key}[{index}]", item)
        elif value is not None:
            pairs.append((key, _plain(value, ",")))

    for key, value in fields.items():
        if key in deep or isinstance(value, dict):
            nested(key, value)
        elif isinstance(value, (list, tuple)):
            pairs.extend((key, _plain(item, ",")) for item in value)
        else:
            pairs.append((key, _plain(value, ",")))
    return pairs


async def _secret(value: Any) -> Any:
    """A credential, or the function that fetches a fresh one."""
    if callable(value):
        value = value()
        if inspect.isawaitable(value):
            value = await value
    return value


async def _bearer(token: Callable[[], Any]) -> str:
    return f"Bearer {await _secret(token)}"


def openapi_tools(spec: Any, **options: Any) -> OpenAPIToolkit:
    """An OpenAPI document as tools. See `OpenAPIToolkit` for the options.

        Agent("support", tools=openapi_tools("openapi.json", token=TOKEN))
    """
    return OpenAPIToolkit(spec, **options)


def toolkits_from_config(config: dict[str, Any] | None,
                         base_dir: str | Path | None = None) -> list[OpenAPIToolkit]:
    """The toolkits a file declares under `openapi:`.

        openapi:
          shop:
            spec: ./shop.openapi.json
            token_env: SHOP_TOKEN          # the secret is named, never written
            include: [getOrder, listOrders]
            writes: ask

    A relative `spec` is looked for next to the file that declares it.
    """
    import os

    out: list[OpenAPIToolkit] = []
    for name, entry in (config or {}).items():
        options = {"spec": entry} if isinstance(entry, str) else dict(entry or {})
        spec = options.pop("spec", None) or options.pop("url", None)
        if not spec:
            raise ConfigurationError(f"openapi.{name} needs a `spec`: a file or a URL")
        if (isinstance(spec, str) and base_dir and "\n" not in spec
                and not spec.startswith(("http://", "https://", "{", "/", "~"))):
            spec = str(Path(base_dir) / spec)
        for key in ("token", "api_key"):
            variable = options.pop(f"{key}_env", None)
            if variable:
                options[key] = os.environ.get(variable)
                if not options[key]:
                    raise ConfigurationError(
                        f"openapi.{name}: the environment variable {variable} is "
                        "not set")
        if isinstance(options.get("basic"), list):
            options["basic"] = tuple(options["basic"])
        options.setdefault("name", name)
        try:
            out.append(OpenAPIToolkit(spec, **options))
        except TypeError as exc:
            raise ConfigurationError(f"openapi.{name}: {exc}") from None
    return out
