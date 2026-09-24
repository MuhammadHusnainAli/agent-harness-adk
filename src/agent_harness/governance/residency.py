"""Where does this model call actually go?

    RegionResolver().resolve(BedrockProvider(region="me-central-1"), "anthropic.claude-opus-5")
    # Region(jurisdiction='ae', location='me-central-1', source='aws region')

The answer is worked out from what the provider object knows at the moment of
the call — Bedrock's region, a cross-region inference profile in the model id
(``eu.anthropic...``), Vertex's region, the endpoint host — rather than from
what someone wrote in a spreadsheet. Where it cannot be known (an Azure
endpoint does not say which region it is in), you declare it once:

    RegionResolver({"azure": "eu", "https://llm.internal.acme.com": "on_prem"})

Jurisdictions are short codes: ``eu`` ``uk`` ``ch`` ``us`` ``ca`` ``ae`` ``sa``
``qa`` ``bh`` ``om`` ``kw`` ``sg`` ``in`` ``cn`` ``kr`` ``jp`` ``au`` ``vn``
``br`` ``il`` ``hk`` ``tw`` ``id`` ``my`` ``th`` ``za`` ``mx`` ``nz`` ``cl``,
plus ``on_prem``, ``apac`` / ``global`` for multi-region routing that cannot be
pinned to one jurisdiction, and ``unknown``.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Mapping
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any
from urllib.parse import urlparse

__all__ = ["Region", "RegionResolver", "resolve_region", "CLOUD_REGIONS", "HOSTS"]


@dataclass(frozen=True)
class Region:
    jurisdiction: str
    location: str = ""
    source: str = ""

    @property
    def known(self) -> bool:
        return self.jurisdiction not in ("unknown", "")


#: Cloud region names → jurisdiction. AWS, Google Cloud and Azure spellings.
CLOUD_REGIONS: dict[str, str] = {
    # --- AWS -------------------------------------------------------------
    "us-east-1": "us", "us-east-2": "us", "us-west-1": "us", "us-west-2": "us",
    "us-gov-west-1": "us", "us-gov-east-1": "us",
    "ca-central-1": "ca", "ca-west-1": "ca", "mx-central-1": "mx", "sa-east-1": "br",
    "eu-west-1": "eu", "eu-west-3": "eu", "eu-central-1": "eu", "eu-north-1": "eu",
    "eu-south-1": "eu", "eu-south-2": "eu", "eu-west-2": "uk", "eu-central-2": "ch",
    "me-central-1": "ae", "me-south-1": "bh", "il-central-1": "il", "af-south-1": "za",
    "ap-southeast-1": "sg", "ap-south-1": "in", "ap-south-2": "in",
    "ap-northeast-1": "jp", "ap-northeast-3": "jp", "ap-northeast-2": "kr",
    "ap-southeast-2": "au", "ap-southeast-4": "au", "ap-east-1": "hk",
    "ap-southeast-3": "id", "ap-southeast-5": "my", "ap-southeast-7": "th",
    "cn-north-1": "cn", "cn-northwest-1": "cn",
    # --- Google Cloud ----------------------------------------------------
    "us-central1": "us", "us-east1": "us", "us-east4": "us", "us-east5": "us",
    "us-south1": "us", "us-west1": "us", "us-west4": "us",
    "northamerica-northeast1": "ca", "northamerica-northeast2": "ca",
    "southamerica-east1": "br", "southamerica-west1": "cl",
    "europe-west1": "eu", "europe-west3": "eu", "europe-west4": "eu",
    "europe-west8": "eu", "europe-west9": "eu", "europe-west10": "eu",
    "europe-west12": "eu", "europe-north1": "eu", "europe-central2": "eu",
    "europe-southwest1": "eu", "europe-west2": "uk", "europe-west6": "ch",
    "me-central1": "qa", "me-central2": "sa", "me-west1": "il",
    "asia-southeast1": "sg", "asia-southeast2": "id", "asia-south1": "in",
    "asia-south2": "in", "asia-northeast1": "jp", "asia-northeast2": "jp",
    "asia-northeast3": "kr", "asia-east1": "tw", "asia-east2": "hk",
    "australia-southeast1": "au", "australia-southeast2": "au",
    "africa-south1": "za", "eu": "eu", "us": "us", "global": "global",
    # --- Azure -----------------------------------------------------------
    "westeurope": "eu", "northeurope": "eu", "swedencentral": "eu",
    "francecentral": "eu", "germanywestcentral": "eu", "italynorth": "eu",
    "spaincentral": "eu", "polandcentral": "eu", "uksouth": "uk", "ukwest": "uk",
    "switzerlandnorth": "ch", "uaenorth": "ae", "uaecentral": "ae",
    "qatarcentral": "qa", "saudiarabiacentral": "sa", "israelcentral": "il",
    "southeastasia": "sg", "centralindia": "in", "southindia": "in",
    "japaneast": "jp", "japanwest": "jp", "koreacentral": "kr",
    "australiaeast": "au", "canadacentral": "ca", "canadaeast": "ca",
    "brazilsouth": "br", "southafricanorth": "za", "eastasia": "hk",
    "eastus": "us", "eastus2": "us", "westus": "us", "westus2": "us", "westus3": "us",
    "centralus": "us", "northcentralus": "us", "southcentralus": "us",
}

#: Direct API hosts → where they process by default.
HOSTS: dict[str, str] = {
    "api.anthropic.com": "us",
    "api.openai.com": "us",
    "eu.api.openai.com": "eu",
    "generativelanguage.googleapis.com": "global",
    "api.mistral.ai": "eu",
    "api.deepseek.com": "cn",
    "api.groq.com": "us",
    "api.together.xyz": "us",
    "api.fireworks.ai": "us",
    "api.x.ai": "us",
    "api.cerebras.ai": "us",
    "openrouter.ai": "global",
}

_PROFILE_PREFIXES = {"eu": "eu", "us": "us", "apac": "apac", "global": "global",
                     "us-gov": "us", "jp": "jp", "au": "au", "ca": "ca"}


class RegionResolver:
    """Provider + model → `Region`. Overrides win over everything it infers.

    Override keys can be a provider name (``"azure"``), a host or URL
    (``"https://llm.internal"``), or ``"model:<glob>"``. Values are
    jurisdiction codes.
    """

    def __init__(self, overrides: Mapping[str, str] | None = None) -> None:
        self.overrides = dict(overrides or {})

    def resolve(self, provider: Any, model: str = "") -> Region:
        name = str(getattr(provider, "name", "") or "")
        base_url = str(getattr(provider, "base_url", "") or "")
        host = (urlparse(base_url).hostname or "").lower() if base_url else ""

        # 1. What the operator declared.
        for key, value in self.overrides.items():
            if key.startswith("model:") and model and fnmatch(model, key[6:]):
                return Region(value, model, "declared (model)")
        if name in self.overrides:
            return Region(self.overrides[name], name, "declared (provider)")
        for key, value in self.overrides.items():
            if "/" in key or "." in key:
                key_host = (urlparse(key).hostname or key).lower()
                if host and (host == key_host or host.endswith("." + key_host)):
                    return Region(value, host, "declared (host)")

        # 2. A cross-region inference profile in the model id pins the geography.
        if "." in model:
            prefix = model.split(".", 1)[0].lower()
            if prefix in _PROFILE_PREFIXES and name in ("bedrock", ""):
                return Region(_PROFILE_PREFIXES[prefix], model, "inference profile")

        # 3. The provider's own region setting.
        region = str(getattr(provider, "region", "") or getattr(provider, "location", "")
                     or "").lower()
        if region:
            if region in CLOUD_REGIONS:
                return Region(CLOUD_REGIONS[region], region, "cloud region")
            return Region("unknown", region, "unrecognised cloud region")

        # 4. The host.
        if host:
            if _private(host):
                return Region("on_prem", host, "private address")
            if host in HOSTS:
                return Region(HOSTS[host], host, "known endpoint")
            for suffix, jur in (("azure.com", "unknown"), (".cn", "cn")):
                if host.endswith(suffix):
                    return Region(jur, host, "endpoint host")
        return Region("unknown", host or name, "not resolvable")


def _private(host: str) -> bool:
    if host in ("localhost", "host.docker.internal") or host.endswith(".local"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_private or address.is_loopback


def resolve_region(provider: Any, model: str = "", *,
                   overrides: Mapping[str, str] | None = None) -> Region:
    """Where `provider` would send a call for `model`. See `RegionResolver`."""
    return RegionResolver(overrides).resolve(provider, model)
