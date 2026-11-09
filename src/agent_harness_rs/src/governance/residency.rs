//! Where does this model call actually go?
//!
//! ```
//! use agent_harness::governance::{ProviderInfo, RegionResolver};
//!
//! let bedrock = ProviderInfo { name: "bedrock".into(), region: "me-central-1".into(), ..Default::default() };
//! let region = RegionResolver::default().resolve(&bedrock, "anthropic.claude-opus-5");
//! assert_eq!((region.jurisdiction.as_str(), region.source.as_str()), ("ae", "cloud region"));
//! ```
//!
//! The answer is worked out from what the provider knows at the moment of the
//! call — Bedrock's region, a cross-region inference profile in the model id
//! (`eu.anthropic...`), Vertex's region, the endpoint host — rather than from
//! what someone wrote in a spreadsheet. Where it cannot be known (an Azure
//! endpoint does not say which region it is in), you declare it once:
//!
//! ```
//! # use agent_harness::governance::RegionResolver;
//! RegionResolver::new([("azure", "eu"), ("https://llm.internal.acme.com", "on_prem")]);
//! ```
//!
//! Jurisdictions are short codes: `eu` `uk` `ch` `us` `ca` `ae` `sa` `qa` `bh`
//! `om` `kw` `sg` `in` `cn` `kr` `jp` `au` `vn` `br` `il` `hk` `tw` `id` `my`
//! `th` `za` `mx` `nz` `cl`, plus `on_prem`, `apac` / `global` for multi-region
//! routing that cannot be pinned to one jurisdiction, and `unknown`.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Where one call goes: a jurisdiction code, the location it was read from,
/// and how it was worked out.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Region {
    pub jurisdiction: String,
    #[serde(default)]
    pub location: String,
    #[serde(default)]
    pub source: String,
}

impl Region {
    fn new(jurisdiction: &str, location: &str, source: &str) -> Self {
        Self { jurisdiction: jurisdiction.into(), location: location.into(), source: source.into() }
    }

    pub fn known(&self) -> bool {
        !matches!(self.jurisdiction.as_str(), "unknown" | "")
    }
}

/// What a provider knows about where it sends calls. Python reads these off the
/// provider object; here they are passed in — the `model_egress` hook gives the
/// provider's name, and may give a JSON object with `base_url` and `region` too.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderInfo {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub base_url: String,
    /// A cloud region (`region` or `location` on the provider).
    #[serde(default)]
    pub region: String,
}

impl ProviderInfo {
    pub fn named(name: &str) -> Self {
        Self { name: name.into(), ..Default::default() }
    }

    /// A provider name, or an object with `name`, `base_url`, `region`/`location`.
    pub fn from_value(value: &Value) -> Self {
        match value {
            Value::String(s) => Self::named(s),
            Value::Object(m) => {
                let s = |k: &str| m.get(k).and_then(Value::as_str).unwrap_or("").to_string();
                let region = Some(s("region")).filter(|r| !r.is_empty()).unwrap_or_else(|| s("location"));
                Self { name: s("name"), base_url: s("base_url"), region }
            }
            _ => Self::default(),
        }
    }
}

/// Cloud region names → jurisdiction. AWS, Google Cloud and Azure spellings.
pub const CLOUD_REGIONS: &[(&str, &str)] = &[
    // --- AWS -------------------------------------------------------------
    ("us-east-1", "us"),
    ("us-east-2", "us"),
    ("us-west-1", "us"),
    ("us-west-2", "us"),
    ("us-gov-west-1", "us"),
    ("us-gov-east-1", "us"),
    ("ca-central-1", "ca"),
    ("ca-west-1", "ca"),
    ("mx-central-1", "mx"),
    ("sa-east-1", "br"),
    ("eu-west-1", "eu"),
    ("eu-west-3", "eu"),
    ("eu-central-1", "eu"),
    ("eu-north-1", "eu"),
    ("eu-south-1", "eu"),
    ("eu-south-2", "eu"),
    ("eu-west-2", "uk"),
    ("eu-central-2", "ch"),
    ("me-central-1", "ae"),
    ("me-south-1", "bh"),
    ("il-central-1", "il"),
    ("af-south-1", "za"),
    ("ap-southeast-1", "sg"),
    ("ap-south-1", "in"),
    ("ap-south-2", "in"),
    ("ap-northeast-1", "jp"),
    ("ap-northeast-3", "jp"),
    ("ap-northeast-2", "kr"),
    ("ap-southeast-2", "au"),
    ("ap-southeast-4", "au"),
    ("ap-east-1", "hk"),
    ("ap-southeast-3", "id"),
    ("ap-southeast-5", "my"),
    ("ap-southeast-7", "th"),
    ("cn-north-1", "cn"),
    ("cn-northwest-1", "cn"),
    // --- Google Cloud ----------------------------------------------------
    ("us-central1", "us"),
    ("us-east1", "us"),
    ("us-east4", "us"),
    ("us-east5", "us"),
    ("us-south1", "us"),
    ("us-west1", "us"),
    ("us-west4", "us"),
    ("northamerica-northeast1", "ca"),
    ("northamerica-northeast2", "ca"),
    ("southamerica-east1", "br"),
    ("southamerica-west1", "cl"),
    ("europe-west1", "eu"),
    ("europe-west3", "eu"),
    ("europe-west4", "eu"),
    ("europe-west8", "eu"),
    ("europe-west9", "eu"),
    ("europe-west10", "eu"),
    ("europe-west12", "eu"),
    ("europe-north1", "eu"),
    ("europe-central2", "eu"),
    ("europe-southwest1", "eu"),
    ("europe-west2", "uk"),
    ("europe-west6", "ch"),
    ("me-central1", "qa"),
    ("me-central2", "sa"),
    ("me-west1", "il"),
    ("asia-southeast1", "sg"),
    ("asia-southeast2", "id"),
    ("asia-south1", "in"),
    ("asia-south2", "in"),
    ("asia-northeast1", "jp"),
    ("asia-northeast2", "jp"),
    ("asia-northeast3", "kr"),
    ("asia-east1", "tw"),
    ("asia-east2", "hk"),
    ("australia-southeast1", "au"),
    ("australia-southeast2", "au"),
    ("africa-south1", "za"),
    ("eu", "eu"),
    ("us", "us"),
    ("global", "global"),
    // --- Azure -----------------------------------------------------------
    ("westeurope", "eu"),
    ("northeurope", "eu"),
    ("swedencentral", "eu"),
    ("francecentral", "eu"),
    ("germanywestcentral", "eu"),
    ("italynorth", "eu"),
    ("spaincentral", "eu"),
    ("polandcentral", "eu"),
    ("uksouth", "uk"),
    ("ukwest", "uk"),
    ("switzerlandnorth", "ch"),
    ("uaenorth", "ae"),
    ("uaecentral", "ae"),
    ("qatarcentral", "qa"),
    ("saudiarabiacentral", "sa"),
    ("israelcentral", "il"),
    ("southeastasia", "sg"),
    ("centralindia", "in"),
    ("southindia", "in"),
    ("japaneast", "jp"),
    ("japanwest", "jp"),
    ("koreacentral", "kr"),
    ("australiaeast", "au"),
    ("canadacentral", "ca"),
    ("canadaeast", "ca"),
    ("brazilsouth", "br"),
    ("southafricanorth", "za"),
    ("eastasia", "hk"),
    ("eastus", "us"),
    ("eastus2", "us"),
    ("westus", "us"),
    ("westus2", "us"),
    ("westus3", "us"),
    ("centralus", "us"),
    ("northcentralus", "us"),
    ("southcentralus", "us"),
];

/// Direct API hosts → where they process by default.
pub const HOSTS: &[(&str, &str)] = &[
    ("api.anthropic.com", "us"),
    ("api.openai.com", "us"),
    ("eu.api.openai.com", "eu"),
    ("generativelanguage.googleapis.com", "global"),
    ("api.mistral.ai", "eu"),
    ("api.deepseek.com", "cn"),
    ("api.groq.com", "us"),
    ("api.together.xyz", "us"),
    ("api.fireworks.ai", "us"),
    ("api.x.ai", "us"),
    ("api.cerebras.ai", "us"),
    ("openrouter.ai", "global"),
];

const PROFILE_PREFIXES: &[(&str, &str)] =
    &[("eu", "eu"), ("us", "us"), ("apac", "apac"), ("global", "global"), ("us-gov", "us"), ("jp", "jp"), ("au", "au"), ("ca", "ca")];

fn lookup(table: &'static [(&'static str, &'static str)], key: &str) -> Option<&'static str> {
    table.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
}

fn lookup_str(table: &[(&str, &str)], key: &str) -> Option<String> {
    table.iter().find(|(k, _)| *k == key).map(|(_, v)| v.to_string())
}

/// `urlparse(url).hostname`: lowercased, without a port or IPv6 brackets.
fn hostname(url: &str) -> Option<String> {
    let parsed = reqwest::Url::parse(url).ok()?;
    let host = parsed.host_str()?;
    if host.is_empty() {
        return None;
    }
    Some(host.trim_start_matches('[').trim_end_matches(']').to_lowercase())
}

/// Provider + model → [`Region`]. Overrides win over everything it infers.
///
/// Override keys can be a provider name (`"azure"`), a host or URL
/// (`"https://llm.internal"`), or `"model:<glob>"`. Values are jurisdiction
/// codes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RegionResolver {
    pub overrides: IndexMap<String, String>,
}

impl RegionResolver {
    pub fn new<I: IntoIterator<Item = (K, V)>, K: Into<String>, V: Into<String>>(overrides: I) -> Self {
        Self { overrides: overrides.into_iter().map(|(k, v)| (k.into(), v.into())).collect() }
    }

    pub fn resolve(&self, provider: &ProviderInfo, model: &str) -> Region {
        let name = provider.name.as_str();
        let host = if provider.base_url.is_empty() { String::new() } else { hostname(&provider.base_url).unwrap_or_default() };

        // 1. What the operator declared.
        for (key, value) in &self.overrides {
            if let Some(glob) = key.strip_prefix("model:")
                && !model.is_empty()
                && super::fnmatch(model, glob)
            {
                return Region::new(value, model, "declared (model)");
            }
        }
        if let Some(value) = self.overrides.get(name) {
            return Region::new(value, name, "declared (provider)");
        }
        for (key, value) in &self.overrides {
            if key.contains('/') || key.contains('.') {
                let key_host = hostname(key).unwrap_or_else(|| key.to_lowercase());
                if !host.is_empty() && (host == key_host || host.ends_with(&format!(".{key_host}"))) {
                    return Region::new(value, &host, "declared (host)");
                }
            }
        }

        // 2. A cross-region inference profile in the model id pins the geography.
        if let Some((prefix, _)) = model.split_once('.') {
            let prefix = prefix.to_lowercase();
            if let Some(j) = lookup_str(PROFILE_PREFIXES, &prefix)
                && matches!(name, "bedrock" | "")
            {
                return Region::new(&j, model, "inference profile");
            }
        }

        // 3. The provider's own region setting.
        let region = provider.region.to_lowercase();
        if !region.is_empty() {
            return match lookup_str(CLOUD_REGIONS, &region) {
                Some(j) => Region::new(&j, &region, "cloud region"),
                None => Region::new("unknown", &region, "unrecognised cloud region"),
            };
        }

        // 4. The host.
        if !host.is_empty() {
            if private(&host) {
                return Region::new("on_prem", &host, "private address");
            }
            if let Some(j) = lookup_str(HOSTS, &host) {
                return Region::new(&j, &host, "known endpoint");
            }
            for (suffix, jur) in [("azure.com", "unknown"), (".cn", "cn")] {
                if host.ends_with(suffix) {
                    return Region::new(jur, &host, "endpoint host");
                }
            }
        }
        Region::new("unknown", if host.is_empty() { name } else { &host }, "not resolvable")
    }

    /// The jurisdiction a known cloud region name maps to.
    pub fn cloud_region(region: &str) -> Option<&'static str> {
        lookup(CLOUD_REGIONS, region)
    }
}

/// Python's `ipaddress` view of a private or loopback address.
fn private(host: &str) -> bool {
    if matches!(host, "localhost" | "host.docker.internal") || host.ends_with(".local") {
        return true;
    }
    match host.parse::<IpAddr>() {
        Ok(IpAddr::V4(v4)) => private_v4(v4),
        Ok(IpAddr::V6(v6)) => private_v6(v6),
        Err(_) => false,
    }
}

fn private_v4(ip: Ipv4Addr) -> bool {
    let o = ip.octets();
    ip.is_private()
        || ip.is_loopback()
        || ip.is_link_local()
        || o[0] == 0
        || o[0] >= 240
        || (o[0] == 192 && o[1] == 0 && o[2] == 0)
        || (o[0] == 192 && o[1] == 0 && o[2] == 2)
        || (o[0] == 198 && (o[1] == 18 || o[1] == 19))
        || (o[0] == 198 && o[1] == 51 && o[2] == 100)
        || (o[0] == 203 && o[1] == 0 && o[2] == 113)
}

fn private_v6(ip: Ipv6Addr) -> bool {
    if let Some(v4) = ip.to_ipv4_mapped() {
        return private_v4(v4);
    }
    let s = ip.segments();
    ip.is_loopback() || ip.is_unspecified() || (s[0] & 0xfe00) == 0xfc00 || (s[0] & 0xffc0) == 0xfe80 || (s[0] == 0x2001 && s[1] == 0x0db8)
}

/// Where `provider` would send a call for `model`. See [`RegionResolver`].
pub fn resolve_region(provider: &ProviderInfo, model: &str, overrides: Option<&IndexMap<String, String>>) -> Region {
    RegionResolver { overrides: overrides.cloned().unwrap_or_default() }.resolve(provider, model)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(name: &str, base_url: &str, region: &str) -> ProviderInfo {
        ProviderInfo { name: name.into(), base_url: base_url.into(), region: region.into() }
    }

    #[test]
    fn region_resolution_from_what_the_provider_knows() {
        let r = RegionResolver::new([("azure", "eu"), ("model:local-*", "on_prem")]);
        assert_eq!(r.resolve(&p("bedrock", "", "me-central-1"), "anthropic.x").jurisdiction, "ae");
        assert_eq!(r.resolve(&p("bedrock", "", "us-east-1"), "eu.anthropic.x").jurisdiction, "eu");
        assert_eq!(r.resolve(&p("vertex", "", "me-central2"), "").jurisdiction, "sa");
        assert_eq!(r.resolve(&p("openai", "https://api.openai.com/v1", ""), "").jurisdiction, "us");
        assert_eq!(r.resolve(&p("ollama", "http://localhost:11434", ""), "").jurisdiction, "on_prem");
        assert_eq!(r.resolve(&p("vllm", "http://10.0.0.5:8000", ""), "").jurisdiction, "on_prem");
        assert_eq!(r.resolve(&p("azure", "https://acme.openai.azure.com", ""), "").jurisdiction, "eu");
        assert_eq!(r.resolve(&p("x", "", ""), "local-llama").jurisdiction, "on_prem");
        assert!(!r.resolve(&p("fake", "http://fake.invalid", ""), "").known());
        assert_eq!(resolve_region(&p("bedrock", "", "me-south-1"), "", None).jurisdiction, "bh");
        let hosted = RegionResolver::new([("https://llm.internal.acme.com", "on_prem")]);
        assert_eq!(hosted.resolve(&p("x", "https://llm.internal.acme.com/v1", ""), "m").source, "declared (host)");
    }
}
