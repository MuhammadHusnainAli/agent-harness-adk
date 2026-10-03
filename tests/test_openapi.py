"""An OpenAPI document, as tools."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest

from agent_harness import (
    Agent,
    Blueprint,
    ConfigurationError,
    FakeProvider,
    Harness,
    OpenAPIToolkit,
    PolicyGate,
    ToolError,
    Workflow,
    openapi_tools,
    tool_call,
)
from agent_harness.cli import main

SHOP: dict[str, Any] = {
    "openapi": "3.0.3",
    "info": {"title": "Shop API", "version": "1.2.0"},
    "servers": [{"url": "https://{region}.shop.test/v1",
                 "variables": {"region": {"default": "eu"}}}],
    "paths": {
        "/orders": {
            "get": {
                "operationId": "listOrders", "summary": "List orders", "tags": ["orders"],
                "parameters": [
                    {"name": "status", "in": "query",
                     "schema": {"type": "array", "items": {"$ref": "#/components/schemas/Status"}}},
                    {"name": "customer.id", "in": "query", "schema": {"type": "string"}},
                    {"name": "fields", "in": "query", "explode": False,
                     "schema": {"type": "array", "items": {"type": "string"}}},
                    {"name": "filter", "in": "query", "style": "deepObject",
                     "schema": {"type": "object"}},
                    {"$ref": "#/components/parameters/Limit"},
                    {"name": "X-Request-Id", "in": "header", "schema": {"type": "string"}},
                    {"name": "Authorization", "in": "header", "schema": {"type": "string"}},
                ],
                "responses": {"200": {"description": "ok"}},
            },
            "post": {
                "operationId": "createOrder", "summary": "Create an order",
                "description": "Creates an order for a customer.", "tags": ["orders"],
                "requestBody": {"required": True, "content": {"application/json": {
                    "schema": {"$ref": "#/components/schemas/NewOrder"}}}},
                "responses": {"201": {"description": "created"}},
            },
        },
        "/orders/{orderId}": {
            "parameters": [{"name": "orderId", "in": "path", "required": True,
                            "description": "The order number.",
                            "schema": {"type": "string"}}],
            "get": {"operationId": "getOrder", "summary": "Get one order",
                    "tags": ["orders"], "responses": {"200": {"description": "ok"}}},
            "delete": {"operationId": "cancel order!", "tags": ["orders"],
                       "responses": {"204": {"description": "gone"}}},
            "patch": {
                "summary": "Change an order",
                "requestBody": {"content": {"application/merge-patch+json": {
                    "schema": {"type": "object", "properties": {
                        "orderId": {"type": "string"}, "note": {"type": "string"}}}}}},
                "responses": {"200": {"description": "ok"}},
            },
        },
        "/orders/{orderId}/tags": {
            "put": {"operationId": "setTags",
                    "requestBody": {"required": True, "content": {"application/json": {
                        "schema": {"type": "array", "items": {"type": "string"}}}}},
                    "responses": {"200": {"description": "ok"}}},
        },
        "/login": {
            "post": {"operationId": "login", "tags": ["auth"],
                     "requestBody": {"content": {"application/x-www-form-urlencoded": {
                         "encoding": {"scopes": {"style": "deepObject"}},
                         "schema": {"type": "object", "required": ["user"],
                                    "properties": {"user": {"type": "string"},
                                                   "remember": {"type": "boolean"},
                                                   "device": {"type": "object"},
                                                   "roles": {"type": "array", "items": {"type": "string"}},
                                                   "scopes": {"type": "array", "items": {"type": "string"}}}}}}},
                     "responses": {"200": {"description": "ok"}}},
        },
        "/notes": {
            "post": {"operationId": "addNote",
                     "requestBody": {"required": True, "content": {"text/plain": {
                         "schema": {"type": "string"}}}},
                     "responses": {"200": {"description": "ok"}}},
        },
        "/upload": {
            "post": {"operationId": "upload",
                     "requestBody": {"content": {"multipart/form-data": {
                         "schema": {"type": "object", "required": ["file"],
                                    "properties": {"file": {"type": "string",
                                                            "format": "binary"}}}}}},
                     "responses": {"200": {"description": "ok"}}},
        },
        "/legacy": {"get": {"operationId": "oldThing", "deprecated": True,
                            "responses": {"200": {"description": "ok"}}}},
        "/tree": {"get": {"operationId": "getTree", "parameters": [
            {"name": "node", "in": "query", "content": {"application/json": {
                "schema": {"$ref": "#/components/schemas/Node"}}}}],
            "responses": {"200": {"description": "ok"}}}},
    },
    "components": {
        "parameters": {"Limit": {"name": "limit", "in": "query",
                                 "description": "How many to return.",
                                 "schema": {"type": "integer", "default": 20,
                                            "maximum": 100}}},
        "schemas": {
            "Status": {"type": "string", "enum": ["open", "shipped", "refunded"]},
            "Base": {"type": "object", "required": ["customer"], "properties": {
                "customer": {"type": "string", "description": "Customer id.",
                             "example": "c_1", "x-internal": True}}},
            "NewOrder": {"allOf": [
                {"$ref": "#/components/schemas/Base"},
                {"type": "object", "required": ["items"], "properties": {
                    "id": {"type": "string", "readOnly": True},
                    "items": {"type": "array", "items": {
                        "type": "object", "properties": {
                            "sku": {"type": "string"},
                            "quantity": {"type": "integer", "minimum": 1}}}},
                    "status": {"$ref": "#/components/schemas/Status"},
                    "gift": {"type": "boolean", "nullable": True}}}]},
            "Node": {"type": "object", "properties": {
                "name": {"type": "string"},
                "children": {"type": "array",
                             "items": {"$ref": "#/components/schemas/Node"}}}},
        },
        "securitySchemes": {
            "key": {"type": "apiKey", "in": "header", "name": "X-API-Key"},
            "session": {"type": "apiKey", "in": "query", "name": "session"},
            "oauth": {"type": "oauth2", "flows": {}},
            "login": {"type": "http", "scheme": "basic"},
        },
    },
}


class Server:
    """What the API would answer, and a record of what it was asked."""

    def __init__(self, answers: dict[str, Any] | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.answers = answers or {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.answers.get(f"{request.method} {request.url.path}")
        if callable(answer):
            answer = answer(request)
        if isinstance(answer, httpx.Response):
            return answer
        return httpx.Response(200, json=answer if answer is not None else {"ok": True})

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]


def shop(server: Server | None = None, **kw: Any) -> tuple[OpenAPIToolkit, Server]:
    server = server or Server()
    return openapi_tools(SHOP, transport=httpx.MockTransport(server), **kw), server


# ----------------------------------------------------------------------
# the tools a document becomes
# ----------------------------------------------------------------------
def test_every_operation_is_a_tool_with_the_operations_own_arguments():
    api, _ = shop()
    assert api.names == ["listOrders", "createOrder", "getOrder", "cancel_order",
                         "patch_orders_orderId", "setTags", "login", "addNote",
                         "getTree"]
    assert api.base_url == "https://eu.shop.test/v1" and api.title == "Shop API"
    assert set(api.skipped) == {("POST /upload", "it needs a file upload (file)"),
                                ("GET /legacy", "deprecated")}
    assert len(api) == 9 and "getOrder" in api

    listing = api.get("listOrders")
    assert listing.description == "List orders. (GET /orders)"
    properties = listing.parameters["properties"]
    # Named so a model can say them; the credential's header is not offered.
    assert list(properties) == ["status", "customer_id", "fields", "filter", "limit",
                                "X-Request-Id"]
    assert properties["status"] == {"type": "array", "items": {
        "type": "string", "enum": ["open", "shipped", "refunded"]}}
    assert properties["limit"] == {"type": "integer", "default": 20, "maximum": 100,
                                   "description": "How many to return."}
    assert "required" not in listing.parameters
    assert listing.tags == {"openapi", "shop_api", "read", "orders"}
    assert listing.operation == {"method": "GET", "path": "/orders",
                                 "operation_id": "listOrders", "api": "Shop API"}

    one = api.get("getOrder").parameters
    assert one == {"type": "object", "required": ["orderId"], "properties": {
        "orderId": {"type": "string", "description": "The order number."}}}


def test_a_request_body_becomes_arguments():
    api, _ = shop()
    create = api.get("createOrder")
    assert create.description == (
        "Create an order. Creates an order for a customer. (POST /orders)")
    properties = create.parameters["properties"]
    # `allOf` merged, the reference followed, and what the server sets left out.
    assert list(properties) == ["customer", "items", "status", "gift"]
    assert properties["customer"] == {"type": "string", "description": "Customer id."}
    assert properties["items"]["items"]["properties"]["quantity"] == {
        "type": "integer", "minimum": 1}
    assert create.parameters["required"] == ["customer", "items"]
    assert "write" in create.tags

    # A body that is not an object is one argument.
    assert api.get("setTags").parameters == {
        "type": "object", "required": ["orderId", "body"], "properties": {
            "orderId": {"type": "string"},
            "body": {"type": "array", "items": {"type": "string"},
                     "description": "The request body."}}}
    # A body field with a parameter's name is told apart.
    assert list(api.get("patch_orders_orderId").parameters["properties"]) == [
        "orderId", "orderId_body", "note"]
    # A schema that refers to itself ends, rather than going round for ever.
    node = api.get("getTree").parameters["properties"]["node"]
    assert node["properties"]["children"]["items"] == {"type": "object"}


# ----------------------------------------------------------------------
# the requests they make
# ----------------------------------------------------------------------
async def test_arguments_go_where_the_document_says():
    api, server = shop()
    await api.call("listOrders", status=["open", "shipped"], customer_id="c 1",
                   fields=["id", "total"], filter={"min": 10, "paid": True}, limit=5,
                   **{"X-Request-Id": "r-1"})
    sent = server.last
    assert sent.method == "GET" and sent.url.path == "/v1/orders"
    assert sent.url.params.multi_items() == [
        ("status", "open"), ("status", "shipped"), ("customer.id", "c 1"),
        ("fields", "id,total"), ("filter[min]", "10"), ("filter[paid]", "true"),
        ("limit", "5")]
    assert sent.headers["x-request-id"] == "r-1"

    await api.call("getOrder", orderId="41/82 ?")
    assert server.last.url.raw_path == b"/v1/orders/41%2F82%20%3F"   # stays one segment

    await api.call("getTree", node={"name": "root", "children": []})
    assert json.loads(server.last.url.params["node"]) == {"name": "root", "children": []}


async def test_bodies_are_sent_as_the_document_says():
    api, server = shop()
    await api.call("createOrder", customer="c_1", gift=None,
                   items=[{"sku": "tea", "quantity": 2}])
    sent = server.last
    assert sent.method == "POST" and sent.headers["content-type"] == "application/json"
    assert json.loads(sent.content) == {"customer": "c_1",
                                        "items": [{"sku": "tea", "quantity": 2}]}

    await api.call("setTags", orderId="7", body=["gift", "fragile"])
    assert server.last.method == "PUT" and json.loads(server.last.content) == [
        "gift", "fragile"]

    await api.call("patch_orders_orderId", orderId="7", orderId_body="8", note="hi")
    assert server.last.headers["content-type"] == "application/merge-patch+json"
    assert json.loads(server.last.content) == {"orderId": "8", "note": "hi"}
    assert server.last.url.path == "/v1/orders/7"

    await api.call("login", user="ada", remember=True)
    assert server.last.headers["content-type"] == "application/x-www-form-urlencoded"
    assert server.last.content == b"user=ada&remember=true"
    await api.call("login", user="ada", device={"os": "linux", "tags": ["a"]},
                   roles=["admin", "dev"], scopes=["read"])
    assert server.last.content.decode() == (
        "user=ada&device%5Bos%5D=linux&device%5Btags%5D%5B0%5D=a&roles=admin"
        "&roles=dev&scopes%5B0%5D=read")

    await api.call("addNote", body="call back tomorrow")
    assert server.last.content == b"call back tomorrow"
    assert server.last.headers["content-type"] == "text/plain"


async def test_what_comes_back_is_what_the_model_reads():
    server = Server({
        "GET /v1/orders/1": {"id": "1", "status": "shipped"},
        "GET /v1/orders/2": httpx.Response(200, text="plain words",
                                           headers={"content-type": "text/plain"}),
        "GET /v1/orders/3": httpx.Response(200, content=b"\x89PNG",
                                           headers={"content-type": "image/png"}),
        "DELETE /v1/orders/1": httpx.Response(204),
        "GET /v1/orders/404": httpx.Response(404, json={"error": "no such order"}),
        "GET /v1/orders/302": httpx.Response(302, headers={"location": "https://evil.test"}),
    })
    api, _ = shop(server)
    assert await api.call("getOrder", orderId="1") == {"id": "1", "status": "shipped"}
    assert await api.call("getOrder", orderId="2") == "plain words"
    assert await api.call("getOrder", orderId="3") == (
        "GET /orders/{orderId} returned 4 bytes of image/png, which is not text.")
    assert await api.call("cancel_order", orderId="1") == (
        "DELETE /orders/{orderId} succeeded (204, no content).")
    with pytest.raises(ToolError, match=r'returned 404: \{"error":"no such order"\}'):
        await api.call("getOrder", orderId="404")
    with pytest.raises(ToolError, match="a redirect to https://evil.test, which is not"):
        await api.call("getOrder", orderId="302")

    with pytest.raises(ToolError, match="getOrder needs orderId"):
        await api.call("getOrder")
    with pytest.raises(ToolError, match="does not take colour; it takes orderId"):
        await api.call("getOrder", orderId="1", colour="red")
    with pytest.raises(ToolError, match="no operation 'nope'"):
        await api.call("nope")
    await api.aclose()


async def test_a_read_is_tried_again_and_a_write_is_not():
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, json={"ok": calls["n"]})

    api, server = shop(Server({"GET /v1/orders/1": flaky, "POST /v1/orders": flaky}))
    assert await api.call("getOrder", orderId="1") == {"ok": 3}
    calls["n"] = 0
    with pytest.raises(ToolError, match="returned 503"):
        await api.call("createOrder", customer="c", items=[])
    assert calls["n"] == 1

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    dead, _ = shop(Server({"GET /v1/orders/1": down}), retries=0)
    with pytest.raises(ToolError, match="could not be reached: ConnectError"):
        await dead.call("getOrder", orderId="1")


# ----------------------------------------------------------------------
# credentials
# ----------------------------------------------------------------------
async def test_credentials_are_sent_and_never_offered_to_the_model():
    api, server = shop(token="tok-1")
    await api.call("getOrder", orderId="1")
    assert server.last.headers["authorization"] == "Bearer tok-1"

    api, server = shop(api_key="k-9")
    await api.call("getOrder", orderId="1")
    assert server.last.headers["x-api-key"] == "k-9"

    api, server = shop(credentials={"session": "s-1", "login": "ada:pw"})
    await api.call("listOrders")
    assert server.last.url.params["session"] == "s-1"
    assert server.last.headers["authorization"] == "Basic YWRhOnB3"

    api, server = shop(basic=("ada", "pw"), headers={"X-Request-Id": "fixed"},
                       params={"limit": 3})
    await api.call("listOrders")
    assert server.last.headers["authorization"] == "Basic YWRhOnB3"
    assert server.last.headers["x-request-id"] == "fixed"
    assert server.last.url.params["limit"] == "3"
    # What is fixed here is not the model's to set.
    assert "limit" not in api.get("listOrders").parameters["properties"]
    assert "X-Request-Id" not in api.get("listOrders").parameters["properties"]
    for tool in api:
        assert "tok" not in json.dumps(tool.parameters) + tool.description

    issued = iter(["t1", "t2"])

    async def fresh() -> str:
        return next(issued)

    api, server = shop(token=fresh)
    await api.call("getOrder", orderId="1")
    await api.call("getOrder", orderId="1")
    assert [r.headers["authorization"] for r in server.requests] == [
        "Bearer t1", "Bearer t2"]

    with pytest.raises(ConfigurationError, match="no security scheme called 'nope'"):
        shop(credentials={"nope": "x"})
    with pytest.raises(ConfigurationError, match="declares no apiKey security scheme"):
        openapi_tools({"openapi": "3.1.0", "paths": {}, "servers": [{"url": "https://a.test"}]},
                      api_key="k")


# ----------------------------------------------------------------------
# choosing which operations
# ----------------------------------------------------------------------
def test_operations_are_chosen_by_name_tag_and_method():
    assert shop(include=["get*", "POST /login"])[0].names == [
        "getOrder", "login", "getTree"]
    assert shop(exclude=["*Order*", "cancel*", "patch_*"])[0].names == [
        "setTags", "login", "addNote", "getTree"]
    assert shop(tags=["auth"])[0].names == ["login"]
    assert shop(methods=["GET"])[0].names == ["listOrders", "getOrder", "getTree"]
    assert "oldThing" in shop(deprecated=True)[0].names
    assert shop(prefix="shop_", methods=["get"])[0].names == [
        "shop_listOrders", "shop_getOrder", "shop_getTree"]

    careful, _ = shop(writes="ask")
    assert careful.get("createOrder").permission == "ask"
    assert careful.get("getOrder").permission == "allow"
    with pytest.raises(ConfigurationError, match="writes must be"):
        shop(writes="maybe")
    assert "createOrder(customer, items, status?, gift?) [ask]" in careful.describe()
    assert "(skipped POST /upload: it needs a file upload (file))" in careful.describe()


# ----------------------------------------------------------------------
# reading documents
# ----------------------------------------------------------------------
def test_a_document_is_a_file_text_a_url_or_a_mapping(tmp_path):
    as_json = tmp_path / "shop.json"
    as_json.write_text(json.dumps(SHOP))
    import yaml

    as_yaml = tmp_path / "shop.yaml"
    as_yaml.write_text(yaml.safe_dump(SHOP))
    for source in (as_json, str(as_yaml), json.dumps(SHOP), yaml.safe_dump(SHOP)):
        assert len(openapi_tools(source)) == 9

    def host(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/docs/openapi.json":
            return httpx.Response(200, json={**SHOP, "servers": [{"url": "/api"}]})
        return httpx.Response(404)

    fetched = openapi_tools("https://shop.test/docs/openapi.json",
                            transport=httpx.MockTransport(host))
    assert fetched.base_url == "https://shop.test/api"       # relative to the document

    for bad, message in (
            (tmp_path / "missing.json", "no OpenAPI document at"),
            ("{not json", "neither JSON nor YAML"),
            ({"info": {}}, "this is not an OpenAPI document"),
            ({"openapi": "3.0.0", "paths": {}}, "does not say where the API is"),
            (42, "is a file, a URL, its text, or a mapping"),
            ("https://shop.test/nope", "returned HTTP 404")):
        with pytest.raises(ConfigurationError, match=message):
            openapi_tools(bad, transport=httpx.MockTransport(host))
    broken = {"openapi": "3.0.0", "servers": [{"url": "https://a.test"}], "paths": {
        "/x": {"get": {"parameters": [{"$ref": "#/components/parameters/Gone"}]}}}}
    with pytest.raises(ConfigurationError, match="which is not in it"):
        openapi_tools(broken)
    assert openapi_tools({"openapi": "3.0.0", "paths": {}},
                         base_url="https://a.test/").base_url == "https://a.test"


async def test_swagger_2_is_read_too():
    document = {
        "swagger": "2.0", "info": {"title": "Pets"}, "host": "pets.test",
        "basePath": "/v2", "schemes": ["https"], "consumes": ["application/json"],
        "securityDefinitions": {"key": {"type": "apiKey", "in": "query", "name": "api_key"},
                                "login": {"type": "basic"}},
        "paths": {
            "/pets": {
                "get": {"operationId": "findPets", "parameters": [
                    {"name": "tags", "in": "query", "type": "array",
                     "items": {"type": "string"}, "collectionFormat": "csv"},
                    {"name": "api_key", "in": "query", "type": "string"}]},
                "post": {"operationId": "addPet", "parameters": [
                    {"name": "pet", "in": "body", "required": True,
                     "schema": {"$ref": "#/definitions/Pet"}}]},
            },
            "/pets/{id}": {"post": {"operationId": "updatePet",
                                    "consumes": ["application/x-www-form-urlencoded"],
                                    "parameters": [
                {"name": "id", "in": "path", "required": True, "type": "integer"},
                {"name": "name", "in": "formData", "type": "string", "required": True},
                {"name": "photo", "in": "formData", "type": "file"}]}},
        },
        "definitions": {"Pet": {"type": "object", "required": ["name"], "properties": {
            "name": {"type": "string"}, "age": {"type": "integer"}}}},
    }
    server = Server()
    api = openapi_tools(document, api_key="k-1", transport=httpx.MockTransport(server))
    assert api.base_url == "https://pets.test/v2"
    assert api.get("findPets").parameters["properties"] == {
        "tags": {"type": "array", "items": {"type": "string"}}}
    assert api.get("addPet").parameters["required"] == ["name"]
    assert list(api.get("updatePet").parameters["properties"]) == ["id", "name"]

    await api.call("findPets", tags=["cat", "small"])
    assert server.last.url.params.multi_items() == [("tags", "cat,small"),
                                                    ("api_key", "k-1")]
    await api.call("addPet", name="Tom", age=3)
    assert json.loads(server.last.content) == {"name": "Tom", "age": 3}
    await api.call("updatePet", id=7, name="Tim")
    assert server.last.url.path == "/v2/pets/7" and server.last.content == b"name=Tim"


# ----------------------------------------------------------------------
# an agent using them
# ----------------------------------------------------------------------
async def test_an_agent_calls_the_api_through_its_tools():
    server = Server({"GET /v1/orders/4182": {"id": "4182", "status": "shipped"},
                     "POST /v1/orders": httpx.Response(201, json={"id": "4183"})})
    api, _ = shop(server, token="tok-1")
    harness = Harness.testing()
    agent = Agent("support", "Help with orders.", tools=api, harness=harness,
                  memory=False, provider=FakeProvider([
                      tool_call("getOrder", orderId="4182"),
                      tool_call("createOrder", customer="c_1",
                                items=[{"sku": "tea", "quantity": 1}]),
                      tool_call("getOrder"),
                      "Order 4182 has shipped; I placed 4183."]))

    result = await agent.run("Where is 4182? Then reorder the tea.")

    assert result.output == "Order 4182 has shipped; I placed 4183."
    results = [b.content for m in result.messages for b in m.content
               if getattr(b, "type", "") == "tool_result"]
    assert json.loads(results[0]) == {"id": "4182", "status": "shipped"}
    assert json.loads(results[1]) == {"id": "4183"}
    assert results[2] == "Error: getOrder needs orderId"       # the model is told
    # The schema the model was given is the document's.
    offered = {t.name: t for t in agent.provider.requests[0].tools}
    assert offered["getOrder"].parameters["required"] == ["orderId"]
    assert server.requests[1].headers["authorization"] == "Bearer tok-1"
    # Every call is on the audit trail, like any other tool's.
    assert [e.target for e in harness.audit.entries if e.action == "tool_call"] == [
        "getOrder", "createOrder", "getOrder"]


async def test_the_rails_apply_to_api_tools():
    api, server = shop(writes="ask")
    harness = Harness.testing()
    agent = Agent("support", tools=api, harness=harness, memory=False,
                  provider=FakeProvider([tool_call("cancel_order", orderId="1"), "No."]))
    result = await agent.run("Cancel order 1.")
    refused = result.messages[2].content[0].content
    assert "needs approval but no approver is configured" in refused
    assert not server.requests

    agent.policy = PolicyGate("allow", approver=lambda tool, args, reason: True)
    agent.provider.queue(tool_call("cancel_order", orderId="1"), "Done.")
    await agent.run("Cancel order 1.")
    assert server.last.method == "DELETE"

    cached, server = shop(cache_reads=True)
    agent = Agent("support", tools=cached, harness=Harness.testing(), memory=False,
                  provider=FakeProvider([tool_call("getOrder", orderId="1"),
                                         tool_call("getOrder", orderId="1"), "Ok."]))
    await agent.run("Twice.")
    assert len(server.requests) == 1


# ----------------------------------------------------------------------
# declared in a file
# ----------------------------------------------------------------------
async def test_a_blueprint_and_a_workflow_can_declare_an_api(tmp_path, monkeypatch):
    (tmp_path / "shop.json").write_text(json.dumps(SHOP))
    monkeypatch.setenv("SHOP_TOKEN", "tok-7")
    file = tmp_path / "agents.yaml"
    file.write_text("""
openapi:
  shop:
    spec: shop.json
    token_env: SHOP_TOKEN
    include: [getOrder, listOrders]
agents:
  support: {instructions: Help., memory: false}
  reader: {instructions: Read., memory: false, tools: [getOrder]}
workflows:
  lookup:
    inputs: {order: {required: true, type: string}}
    steps:
      - {id: found, tool: getOrder, args: {orderId: "{{ inputs.order }}"}}
""")
    blueprint = Blueprint.from_file(file)
    harness = Harness.testing()
    assert blueprint.build("support", harness=harness).tools.names == [
        "getOrder", "listOrders"]
    assert blueprint.build("reader", harness=harness).tools.names == ["getOrder"]
    toolkit = blueprint.api_toolkits()[0]
    assert toolkit.name == "shop" and toolkit._headers["Authorization"] == "Bearer tok-7"

    server = Server({"GET /v1/orders/4182": {"id": "4182"}})
    toolkit._http = httpx.AsyncClient(transport=httpx.MockTransport(server))
    result = await blueprint.workflow("lookup", harness=harness).run({"order": "4182"})
    assert result.output == {"id": "4182"}

    # A workflow file of its own can declare one as well.
    flow = tmp_path / "flow.yaml"
    flow.write_text("""
openapi: {shop: shop.json}
steps:
  - {tool: listOrders, args: {limit: 2}}
""")
    workflow = Workflow.from_file(flow, harness=harness)
    workflow.apis[0]._http = httpx.AsyncClient(transport=httpx.MockTransport(server))
    assert (await workflow.run()).ok and server.last.url.params["limit"] == "2"

    monkeypatch.delenv("SHOP_TOKEN")
    with pytest.raises(ConfigurationError, match="SHOP_TOKEN is not set"):
        Blueprint.from_file(file).build("support")
    with pytest.raises(ConfigurationError, match="openapi.shop needs a `spec`"):
        Blueprint.from_dict({"openapi": {"shop": {}}, "agents": {"a": {}}}).build("a")
    with pytest.raises(ConfigurationError, match="unexpected keyword argument 'colour'"):
        Blueprint.from_dict({"openapi": {"shop": {"spec": SHOP, "colour": "red"}},
                             "agents": {"a": {}}}).build("a")


# ----------------------------------------------------------------------
# over a real socket, and from the command line
# ----------------------------------------------------------------------
class _Api(BaseHTTPRequestHandler):
    seen: list[tuple[str, str, dict[str, str], bytes]] = []

    def _reply(self, status: int, body: Any) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - the name http.server looks for
        _Api.seen.append(("GET", self.path, dict(self.headers), b""))
        if self.path == "/openapi.json":
            self._reply(200, {**SHOP, "servers": [{"url": "/v1"}]})
        elif self.path.startswith("/v1/orders/"):
            self._reply(200, {"id": self.path.rsplit("/", 1)[-1], "status": "shipped"})
        else:
            self._reply(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        body = self.rfile.read(int(self.headers.get("content-length") or 0))
        _Api.seen.append(("POST", self.path, dict(self.headers), body))
        self._reply(201, {"id": "4183", **json.loads(body)})

    def log_message(self, *args: Any) -> None:
        return None


@pytest.fixture
def live():
    _Api.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Api)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    thread.join(5)


async def test_a_document_fetched_from_a_live_api_calls_that_api(live):
    api = openapi_tools(f"{live}/openapi.json", api_key="k-1")
    assert api.base_url == f"{live}/v1"
    agent = Agent("support", tools=api, harness=Harness.testing(), memory=False,
                  provider=FakeProvider([
                      tool_call("getOrder", orderId="4182"),
                      tool_call("createOrder", customer="c_1", items=[]),
                      "Shipped, and reordered."]))
    result = await agent.run("Check 4182 and reorder.")
    await api.aclose()

    assert result.ok and result.output == "Shipped, and reordered."
    method, path, headers, _ = _Api.seen[1]
    assert (method, path, headers["X-API-Key"]) == ("GET", "/v1/orders/4182", "k-1")
    assert json.loads(_Api.seen[2][3]) == {"customer": "c_1", "items": []}
    with pytest.raises(ToolError, match="returned 404"):
        await openapi_tools(f"{live}/openapi.json").call("getTree")


def test_the_command_line_lists_the_tools_and_calls_one(live, tmp_path, capsys):
    file = tmp_path / "shop.json"
    file.write_text(json.dumps(SHOP))

    assert main(["openapi", str(file)]) == 0
    listed = capsys.readouterr().out
    assert "Shop API — https://eu.shop.test/v1 — 9 tools" in listed
    assert "getOrder(orderId)" in listed and "GET /orders/{orderId}" in listed

    assert main(["openapi", str(file), "--json"]) == 0
    tools = json.loads(capsys.readouterr().out)
    assert tools[2]["name"] == "getOrder" and tools[2]["method"] == "GET"

    assert main(["openapi", f"{live}/openapi.json", "--call", "getOrder",
                 "--arg", "orderId=4182", "--token", "tok-1"]) == 0
    assert json.loads(capsys.readouterr().out) == {"id": "4182", "status": "shipped"}
    assert _Api.seen[-1][2]["Authorization"] == "Bearer tok-1"

    assert main(["openapi", f"{live}/openapi.json", "--call", "getOrder"]) == 1
    assert "getOrder needs orderId" in capsys.readouterr().err


def test_an_agent_on_the_command_line_can_be_given_an_api(tmp_path, monkeypatch):
    from agent_harness.cli import _agent, build_parser

    file = tmp_path / "shop.json"
    file.write_text(json.dumps(SHOP))
    monkeypatch.setenv("OPENAPI_TOKEN", "tok-env")
    args = build_parser().parse_args(["run", "Where is 4182?", "--openapi", str(file),
                                      "--no-memory", "--approve"])
    agent = _agent(args, Harness.testing())
    assert "getOrder" in agent.tools and "createOrder" in agent.tools
    # --approve asks before anything that writes.
    assert agent.tools.get("createOrder").permission == "ask"
    assert agent.tools.get("getOrder").permission == "allow"
