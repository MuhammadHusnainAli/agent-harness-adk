"""The AI inventory: every agent, model, provider, tool and MCP server in use.

Registries of AI systems are asked for by ISO 42001, DORA's register of ICT
third parties, Colorado and the EU AI Act's deployer duties — and an agent's
tools are its supply chain (OWASP ASI04). So each tool's schema is hashed when
first seen: `pin()` freezes the set, and a tool whose schema changes afterwards
— an MCP server that quietly altered what a tool does — is reported as drift
and can be refused by policy (`tool_drift: deny`).

    gov.inventory.pin()                     # after review
    gov.inventory.drift()                   # what changed since
    gov.inventory.to_cyclonedx()            # an ML-BOM for your SBOM tooling
"""

from __future__ import annotations

import hashlib
import json
import time
import weakref
from pathlib import Path
from typing import Any

__all__ = ["AIInventory", "tool_fingerprint"]


def tool_fingerprint(tool: Any) -> str:
    """sha256 over what the model is told about the tool: name, description, schema."""
    schema = tool.to_schema().model_dump() if hasattr(tool, "to_schema") else {}
    payload = json.dumps(schema, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


class AIInventory:
    """Built as agents register; pins persist to a file when given a path."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.agents: dict[str, dict[str, Any]] = {}
        self.pins: dict[str, str] = {}
        self._live: weakref.WeakValueDictionary[str, Any] = weakref.WeakValueDictionary()
        if self.path and self.path.exists():
            self.pins = json.loads(self.path.read_text(encoding="utf-8")).get("pins", {})

    def register(self, agent: Any, *, identity: Any, region: Any = None) -> dict[str, Any]:
        provider = getattr(agent, "_provider", None)
        provider_name = (provider if isinstance(provider, str)
                         else getattr(provider, "name", None)) or "auto"
        tools = []
        for tool in agent.tools:
            tools.append({
                "name": tool.name, "tags": sorted(getattr(tool, "tags", ()) or ()),
                "permission": getattr(tool, "permission", "allow"),
                "fingerprint": tool_fingerprint(tool),
            })
        entry = {
            "name": agent.name,
            "identity": identity.model_dump(mode="json"),
            "model": agent.model,
            "provider": provider_name,
            "region": getattr(region, "jurisdiction", None) or "unknown",
            "version": getattr(agent, "version", ""),
            "tools": tools,
            "skills": sorted(agent.skills.names) if agent.skills else [],
            "subagents": sorted(agent.subagents),
            "registered": time.time(),
        }
        self.agents[agent.name] = entry
        self._live[agent.name] = agent
        return entry

    # ---- supply chain ---------------------------------------------------------
    def fingerprints(self) -> dict[str, str]:
        """Tool fingerprints as they are *now* — recomputed from live agents."""
        out: dict[str, str] = {}
        for name, entry in self.agents.items():
            agent = self._live.get(name)
            if agent is not None:
                for tool in agent.tools:
                    out[f"{name}/{tool.name}"] = tool_fingerprint(tool)
            else:
                for tool in entry["tools"]:
                    out[f"{name}/{tool['name']}"] = tool["fingerprint"]
        return out

    def pin(self) -> dict[str, str]:
        """Freeze the current tool fingerprints as the approved baseline."""
        self.pins = self.fingerprints()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"pins": self.pins, "pinned": time.time()},
                                            indent=2), encoding="utf-8")
        return dict(self.pins)

    def check_tool(self, agent: str, tool: Any) -> str | None:
        """None when fine; otherwise why this tool no longer matches its pin."""
        if not self.pins:
            return None
        key = f"{agent}/{tool.name}"
        pinned = self.pins.get(key)
        if pinned is None:
            return f"{tool.name} was not in the pinned inventory"
        if pinned != tool_fingerprint(tool):
            return f"{tool.name} has changed since it was pinned"
        return None

    def drift(self) -> dict[str, list[str]]:
        now = self.fingerprints()
        return {
            "added": sorted(k for k in now if k not in self.pins),
            "removed": sorted(k for k in self.pins if k not in now),
            "changed": sorted(k for k in now if k in self.pins and self.pins[k] != now[k]),
        }

    # ---- exports --------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        return {"agents": list(self.agents.values()), "pinned": bool(self.pins),
                "drift": self.drift() if self.pins else None}

    def third_parties(self) -> list[dict[str, Any]]:
        """Distinct model providers and where they process — DORA's register shape."""
        seen: dict[tuple[str, str], dict[str, Any]] = {}
        for entry in self.agents.values():
            key = (entry["provider"], entry["region"])
            row = seen.setdefault(key, {"provider": entry["provider"],
                                        "jurisdiction": entry["region"],
                                        "models": set(), "agents": set()})
            row["models"].add(entry["model"])
            row["agents"].add(entry["name"])
        return [{**r, "models": sorted(r["models"]), "agents": sorted(r["agents"])}
                for r in seen.values()]

    def to_cyclonedx(self) -> dict[str, Any]:
        """A CycloneDX 1.6 ML-BOM: models as components, providers as services."""
        components, services = [], []
        for entry in self.agents.values():
            components.append({
                "type": "machine-learning-model", "bom-ref": f"model:{entry['model']}",
                "name": entry["model"], "supplier": {"name": entry["provider"]},
            })
            services.append({
                "bom-ref": f"agent:{entry['name']}", "name": entry["name"],
                "description": entry["identity"].get("purpose", ""),
                "services": [{"bom-ref": f"tool:{entry['name']}/{t['name']}",
                              "name": t["name"],
                              "properties": [{"name": "sha256", "value": t["fingerprint"]}]}
                             for t in entry["tools"]],
                "properties": [{"name": "owner", "value": entry["identity"].get("owner", "")},
                               {"name": "jurisdiction", "value": entry["region"]}],
            })
        unique = {c["bom-ref"]: c for c in components}
        return {"bomFormat": "CycloneDX", "specVersion": "1.6", "version": 1,
                "metadata": {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             "tools": [{"name": "agent-harness-adk"}]},
                "components": list(unique.values()), "services": services}
