//! Is this address on the public internet? Asked before the harness fetches
//! anything a model named — a page, a reference image.

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::sync::Mutex;
use std::time::Duration;

/// Names that only ever mean this machine or the local network.
const LOCAL_NAMES: [&str; 5] = [".localhost", ".local", ".internal", ".lan", ".home.arpa"];

/// Does this name lead to this machine or a private network?
///
/// The name is resolved, so one that merely points at a private address is
/// caught too. A name that does not resolve is not private: it will not load.
/// `cache` remembers the answer per name, so a page of links is resolved once.
pub(crate) async fn is_private_host(host: &str, cache: Option<&Mutex<HashMap<String, bool>>>) -> bool {
    let host = host.to_lowercase().trim_end_matches('.').to_string();
    if let Some(cache) = cache
        && let Some(known) = cache.lock().unwrap_or_else(|p| p.into_inner()).get(&host)
    {
        return *known;
    }
    let mut private = host == "localhost" || LOCAL_NAMES.iter().any(|s| host.ends_with(s));
    if !private {
        let bare = host.trim_matches(|c| c == '[' || c == ']');
        let bare = bare.split('%').next().unwrap_or(bare);
        let addresses: Vec<IpAddr> = match bare.parse::<IpAddr>() {
            Ok(ip) => vec![ip],
            Err(_) => match tokio::time::timeout(Duration::from_secs(4), tokio::net::lookup_host((host.as_str(), 0))).await {
                Ok(Ok(found)) => found.map(|a| a.ip()).collect(),
                _ => Vec::new(),
            },
        };
        private = addresses.into_iter().any(is_private_ip);
    }
    if let Some(cache) = cache {
        cache.lock().unwrap_or_else(|p| p.into_inner()).insert(host, private);
    }
    private
}

/// Python's `ip.is_private or is_loopback or is_link_local or is_reserved or
/// is_multicast or is_unspecified`, for one address.
pub(crate) fn is_private_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => private_v4(v4),
        IpAddr::V6(v6) => private_v6(v6),
    }
}

fn in_v4(ip: Ipv4Addr, net: [u8; 4], bits: u32) -> bool {
    let mask = if bits == 0 { 0 } else { u32::MAX << (32 - bits) };
    (u32::from(ip) & mask) == (u32::from(Ipv4Addr::from(net)) & mask)
}

fn private_v4(ip: Ipv4Addr) -> bool {
    // `ipaddress`'s private networks, plus loopback, link-local, multicast,
    // reserved (240/4) and unspecified. 100.64/10 (carrier-grade NAT, also
    // where Tailscale lives) is refused too: it is never the public internet.
    const NETS: [([u8; 4], u32); 17] = [
        ([0, 0, 0, 0], 8),
        ([10, 0, 0, 0], 8),
        ([100, 64, 0, 0], 10),
        ([127, 0, 0, 0], 8),
        ([169, 254, 0, 0], 16),
        ([172, 16, 0, 0], 12),
        ([192, 0, 0, 0], 29),
        ([192, 0, 0, 170], 31),
        ([192, 0, 2, 0], 24),
        ([192, 168, 0, 0], 16),
        ([198, 18, 0, 0], 15),
        ([198, 51, 100, 0], 24),
        ([203, 0, 113, 0], 24),
        ([224, 0, 0, 0], 4),
        ([240, 0, 0, 0], 4),
        ([255, 255, 255, 255], 32),
        ([192, 88, 99, 0], 24),
    ];
    NETS.iter().any(|(net, bits)| in_v4(ip, *net, *bits))
}

fn in_v6(ip: Ipv6Addr, net: [u16; 8], bits: u32) -> bool {
    let mask = if bits == 0 { 0 } else { u128::MAX << (128 - bits) };
    (u128::from(ip) & mask) == (u128::from(Ipv6Addr::from(net)) & mask)
}

fn private_v6(ip: Ipv6Addr) -> bool {
    // An IPv4 address wearing IPv6 clothes is judged as the IPv4 address.
    if let Some(v4) = ip.to_ipv4_mapped() {
        return private_v4(v4);
    }
    // NAT64 (64:ff9b::/96) carries an IPv4 address in its last 32 bits.
    if in_v6(ip, [0x64, 0xff9b, 0, 0, 0, 0, 0, 0], 96) {
        let o = ip.octets();
        return private_v4(Ipv4Addr::new(o[12], o[13], o[14], o[15]));
    }
    // 6to4 (2002::/16) carries an IPv4 address in its second and third groups.
    if in_v6(ip, [0x2002, 0, 0, 0, 0, 0, 0, 0], 16) {
        let o = ip.octets();
        return private_v4(Ipv4Addr::new(o[2], o[3], o[4], o[5]));
    }
    // `ipaddress` reserves everything outside 2000::/3 (::/8, 100::/8, fc00::/7,
    // fe80::/10, ff00::/8, ...); inside it, the IETF and documentation blocks
    // are private.
    !in_v6(ip, [0x2000, 0, 0, 0, 0, 0, 0, 0], 3) || in_v6(ip, [0x2001, 0, 0, 0, 0, 0, 0, 0], 23) || in_v6(ip, [0x2001, 0xdb8, 0, 0, 0, 0, 0, 0], 32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn private_addresses_are_recognised() {
        for ip in [
            "127.0.0.1",
            "10.1.2.3",
            "172.20.0.1",
            "192.168.1.1",
            "169.254.169.254",
            "0.0.0.0",
            "224.0.0.1",
            "100.64.1.1",
            "::1",
            "::",
            "fe80::1",
            "fc00::1",
            "ff02::1",
            "::ffff:127.0.0.1",
            "64:ff9b::a00:1",
            "2002:c0a8:101::1",
            "2001:db8::1",
        ] {
            assert!(is_private_ip(ip.parse().unwrap()), "{ip}");
        }
        for ip in ["8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:4700:4700::1111", "::ffff:8.8.8.8"] {
            assert!(!is_private_ip(ip.parse().unwrap()), "{ip}");
        }
    }

    #[tokio::test]
    async fn names_that_lead_home_are_private() {
        let cache = Mutex::new(HashMap::new());
        assert!(is_private_host("localhost", Some(&cache)).await);
        assert!(is_private_host("LOCALHOST.", None).await);
        assert!(is_private_host("printer.local", None).await);
        assert!(is_private_host("db.internal", None).await);
        assert!(is_private_host("[::1]", None).await);
        assert!(is_private_host("127.0.0.1", Some(&cache)).await);
        assert!(!is_private_host("8.8.8.8", None).await);
        assert_eq!(cache.lock().unwrap().get("localhost"), Some(&true));
    }
}
