//! What kind of data is this? Classification, minimisation and pseudonymisation.
//!
//! ```
//! use agent_harness::governance::DataClassifier;
//!
//! let found = DataClassifier::default().scan("Mail bob@example.com, he is diabetic");
//! assert!(found.classes().contains("contact") && found.classes().contains("special_category"));
//! ```
//!
//! Everything the residency, purpose and redaction rules decide on starts here,
//! so the detectors are the exact kind where exactness is possible: card numbers
//! are Luhn-checked, IBANs mod-97-checked, and national identity numbers are
//! validated by their own check digits — Emirates ID (Luhn), Saudi national ID
//! and Iqama (Luhn), Aadhaar (Verhoeff), Singapore NRIC/FIN (weighted mod 11),
//! the Chinese resident ID (ISO 7064 mod 11-2). A sixteen-digit order number is
//! not reported as a card and a random twelve digits is not an Aadhaar.
//!
//! Special-category data (health, religion, ethnicity, ...) cannot be proven by a
//! pattern. Those detectors are keyword screens, report a lower confidence, and
//! say so; pair them with an `LLMGuard` if you need judgement rather than recall.
//! Redacting such a class removes the *sentence* the keyword sits in — masking
//! the word alone ("[health]") would leave the statement about the person intact.
//!
//! Spans are byte offsets into the text (Python's are character offsets); they
//! are only ever used to cut the same text.

use std::collections::{BTreeSet, HashMap};
use std::sync::{LazyLock, Mutex};

use blake2::Digest;
use indexmap::IndexMap;
use regex::Regex;
use serde_json::Value;

use crate::guardrails::detectors::{PIIDetector, SecretDetector};

/// The taxonomy. Keys are what policies and identities name.
pub const DATA_CLASSES: [(&str, &str); 16] = [
    ("contact", "email addresses, phone numbers"),
    ("financial", "card numbers, bank accounts (IBAN)"),
    ("government_id", "national identity numbers, social security numbers"),
    ("network_id", "IP addresses and device identifiers"),
    ("credentials", "API keys, tokens, private keys, passwords"),
    ("health", "diagnoses, conditions, treatment, medication"),
    ("biometric", "fingerprints, face and voice prints"),
    ("genetic", "DNA and genetic test results"),
    ("religion", "religious or philosophical belief"),
    ("ethnicity", "racial or ethnic origin"),
    ("political", "political opinion"),
    ("sexual_orientation", "sex life or sexual orientation"),
    ("trade_union", "trade union membership"),
    ("criminal", "criminal convictions and offences"),
    ("children", "data about minors"),
    ("location", "precise location"),
];

/// GDPR Art 9 (and its equivalents: PIPL "sensitive", DPDP, KSA PDPL "sensitive").
pub const SPECIAL_CATEGORIES: [&str; 9] =
    ["health", "biometric", "genetic", "religion", "ethnicity", "political", "sexual_orientation", "trade_union", "criminal"];

/// Everything that identifies or describes a person: every class but `credentials`.
pub static PERSONAL: LazyLock<BTreeSet<String>> =
    LazyLock::new(|| DATA_CLASSES.iter().map(|(c, _)| *c).filter(|c| *c != "credentials").map(str::to_string).collect());

pub(crate) fn is_personal(class: &str) -> bool {
    PERSONAL.contains(class)
}

pub(crate) fn is_special(class: &str) -> bool {
    SPECIAL_CATEGORIES.contains(&class)
}

/// Every data-class name a policy may use: the taxonomy, the umbrellas, yours.
pub fn known_classes<I: IntoIterator<Item = S>, S: Into<String>>(extra: I) -> BTreeSet<String> {
    let mut out: BTreeSet<String> = DATA_CLASSES.iter().map(|(c, _)| c.to_string()).collect();
    out.insert("personal".into());
    out.insert("special_category".into());
    out.extend(extra.into_iter().map(Into::into));
    out
}

/// Resolve the umbrella names `personal` and `special_category`.
pub fn expand_classes<I: IntoIterator<Item = S>, S: AsRef<str>>(names: I) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for name in names {
        match name.as_ref() {
            "personal" => out.extend(PERSONAL.iter().cloned()),
            "special_category" => out.extend(SPECIAL_CATEGORIES.iter().map(|s| s.to_string())),
            other => {
                out.insert(other.to_string());
            }
        }
    }
    out
}

// ---- check digits -----------------------------------------------------------

/// Luhn mod 10 at any length (the card detector's version insists on 12-19).
pub fn luhn(digits: &str) -> bool {
    let numbers: Vec<u32> = digits.chars().filter_map(|c| c.to_digit(10)).collect();
    if numbers.len() < 2 {
        return false;
    }
    let parity = numbers.len() % 2;
    let total: u32 = numbers.iter().enumerate().map(|(i, &d)| if i % 2 == parity { if d > 4 { d * 2 - 9 } else { d * 2 } } else { d }).sum();
    total.is_multiple_of(10)
}

const VERHOEFF_D: [[u8; 10]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
];
const VERHOEFF_P: [[u8; 10]; 8] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
];

/// The Verhoeff check Aadhaar numbers carry in their last digit.
pub fn verhoeff_valid(digits: &str) -> bool {
    let mut check = 0usize;
    for (index, c) in digits.chars().rev().enumerate() {
        let Some(d) = c.to_digit(10) else { return false };
        check = VERHOEFF_D[check][VERHOEFF_P[index % 8][d as usize] as usize] as usize;
    }
    check == 0
}

/// Singapore NRIC (S/T) and FIN (F/G): weighted sum mod 11 → letter.
pub fn nric_valid(value: &str) -> bool {
    let value: Vec<char> = value.to_uppercase().chars().collect();
    if value.len() != 9 || !"STFG".contains(value[0]) || !value[1..8].iter().all(|c| c.is_ascii_digit()) {
        return false;
    }
    let mut total: u32 = value[1..8].iter().zip([2, 7, 6, 5, 4, 3, 2]).map(|(d, w)| d.to_digit(10).unwrap_or(0) * w).sum();
    if "TG".contains(value[0]) {
        total += 4;
    }
    let table: Vec<char> = if "ST".contains(value[0]) { "JZIHGFEDCBA" } else { "XWUTRQPNMLK" }.chars().collect();
    table[(total % 11) as usize] == value[8]
}

/// China's 18-character resident identity number, ISO 7064 MOD 11-2.
pub fn cn_resident_id_valid(value: &str) -> bool {
    let value: Vec<char> = value.to_uppercase().chars().collect();
    if value.len() != 18 || !value[..17].iter().all(|c| c.is_ascii_digit()) {
        return false;
    }
    let weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2];
    let total: u32 = value[..17].iter().zip(weights).map(|(d, w)| d.to_digit(10).unwrap_or(0) * w).sum();
    let checks: Vec<char> = "10X98765432".chars().collect();
    if checks[(total % 11) as usize] != value[17] {
        return false;
    }
    let s: String = value.iter().collect();
    valid_date(&s[6..10], &s[10..12], &s[12..14])
}

fn valid_date(year: &str, month: &str, day: &str) -> bool {
    let (Ok(y), Ok(m), Ok(d)) = (year.parse::<i64>(), month.parse::<u32>(), day.parse::<u32>()) else { return false };
    if !(1..=9999).contains(&y) || !(1..=12).contains(&m) || d == 0 {
        return false;
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let days = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m as usize - 1];
    d <= days
}

/// Korean resident registration number: a real birth date and sex digit.
///
/// RRNs issued since October 2020 no longer carry a check digit, so the date
/// and century/sex digit are what can be verified.
fn kr_rrn_valid(value: &str) -> bool {
    let digits: String = value.replace('-', "");
    let century = match digits.chars().nth(6) {
        Some('1' | '2' | '5' | '6') => "19",
        Some('3' | '4' | '7' | '8') => "20",
        Some('9' | '0') => "18",
        _ => return false,
    };
    digits.len() >= 6 && valid_date(&format!("{century}{}", &digits[..2]), &digits[2..4], &digits[4..6])
}

fn only_digits(value: &str) -> String {
    value.chars().filter(char::is_ascii_digit).collect()
}

/// Italian partita IVA: eleven digits, Luhn-style check.
fn it_vat_valid(value: &str) -> bool {
    let digits = only_digits(value);
    digits.len() == 11 && luhn(&digits)
}

fn eu_vat_valid(value: &str) -> bool {
    if value.starts_with("IT") { it_vat_valid(value) } else { true }
}

fn uk_nino_valid(value: &str) -> bool {
    // Python: (?!BG|GB|NK|KN|TN|NT|ZZ) at the start. Rust has no look-ahead, so
    // the prefix is checked after the match.
    !["BG", "GB", "NK", "KN", "TN", "NT", "ZZ"].iter().any(|p| value.starts_with(p))
}

struct IdPattern {
    kind: &'static str,
    pattern: Regex,
    valid: fn(&str) -> bool,
}

fn re(pattern: &str) -> Regex {
    Regex::new(pattern).expect("built-in data pattern compiles")
}

/// EU VAT numbers, always with their country prefix — which is what keeps them
/// apart from any other run of digits. A sole trader's VAT number is personal data.
const EU_VAT: &str = concat!(
    r"\b(?:ATU\d{8}|BE[01]\d{9}|BG\d{9,10}|CY\d{8}[A-Z]|CZ\d{8,10}|DE\d{9}|DK\d{8}",
    r"|EE\d{9}|EL\d{9}|ES[A-Z0-9]\d{7}[A-Z0-9]|FI\d{8}|FR[A-HJ-NP-Z0-9]{2}\d{9}",
    r"|HR\d{11}|HU\d{8}|IE\d{7}[A-W][A-I]?|IE\d[A-Z+*]\d{5}[A-W]|IT\d{11}|LT(?:\d{9}|\d{12})",
    r"|LU\d{8}|LV\d{11}|MT\d{8}|NL\d{9}B\d{2}|PL\d{10}|PT\d{9}|RO\d{2,10}|SE\d{12}",
    r"|SI\d{8}|SK\d{10})\b"
);

static NATIONAL_IDS: LazyLock<Vec<IdPattern>> = LazyLock::new(|| {
    vec![
        IdPattern { kind: "emirates_id", pattern: re(r"\b784[- ]?\d{4}[- ]?\d{7}[- ]?\d\b"), valid: |v| luhn(&only_digits(v)) },
        IdPattern { kind: "cn_resident_id", pattern: re(r"\b\d{17}[\dXx]\b"), valid: cn_resident_id_valid },
        IdPattern { kind: "aadhaar", pattern: re(r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b"), valid: |v| verhoeff_valid(&only_digits(v)) },
        IdPattern { kind: "saudi_id", pattern: re(r"\b[12]\d{9}\b"), valid: luhn },
        IdPattern { kind: "sg_nric", pattern: re(r"\b[STFGstfg]\d{7}[A-Za-z]\b"), valid: nric_valid },
        IdPattern { kind: "kr_rrn", pattern: re(r"\b\d{6}-[1-8]\d{6}\b"), valid: kr_rrn_valid },
        IdPattern { kind: "eu_vat", pattern: re(EU_VAT), valid: eu_vat_valid },
        IdPattern { kind: "uk_nino", pattern: re(r"\b[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z] ?\d{2} ?\d{2} ?\d{2} ?[A-D]\b"), valid: uk_nino_valid },
    ]
});

/// Keyword screens for what no pattern can prove. Word-bounded, case-folded.
const KEYWORDS: [(&str, &[&str]); 10] = [
    (
        "health",
        &[
            "diagnosed",
            "diagnosis",
            "diabetes",
            "diabetic",
            "cancer",
            "chemotherapy",
            "hiv",
            "aids",
            "hepatitis",
            "depression",
            "anxiety disorder",
            "schizophrenia",
            "bipolar",
            "pregnant",
            "pregnancy",
            "prescription",
            "prescribed",
            "medication",
            "blood pressure",
            "blood test",
            "mri",
            "surgery",
            "disability",
            "therapy session",
            "medical record",
            "patient",
            "icd-10",
            "asthma",
            "dementia",
            "alzheimer",
        ],
    ),
    ("biometric", &["fingerprint", "face print", "faceprint", "voiceprint", "iris scan", "retina scan", "facial recognition template", "biometric"]),
    ("genetic", &["dna", "genome", "genetic test", "brca", "genotype"]),
    ("religion", &["muslim", "christian", "jewish", "hindu", "buddhist", "sikh", "atheist", "religion", "religious belief", "converted to"]),
    ("ethnicity", &["ethnicity", "ethnic origin", "racial origin", "race:"]),
    ("political", &["political party", "voted for", "party member", "political opinion", "political affiliation"]),
    ("sexual_orientation", &["gay", "lesbian", "bisexual", "transgender", "sexual orientation", "homosexual", "lgbt"]),
    ("trade_union", &["trade union", "union member", "labour union", "labor union"]),
    ("criminal", &["convicted", "conviction", "criminal record", "arrested", "sentenced to", "prison sentence", "offence", "felony"]),
    ("children", &["my son", "my daughter", "years old child", "minor aged", "under 18", "under 16", "pupil", "schoolchild", "kindergarten"]),
];

static KEYWORD_RE: LazyLock<Vec<(&'static str, Regex)>> = LazyLock::new(|| {
    KEYWORDS.iter().map(|(cls, words)| (*cls, re(&format!(r"(?i)\b(?:{})\b", words.iter().map(|w| regex::escape(w)).collect::<Vec<_>>().join("|"))))).collect()
});

fn pii_class(category: &str) -> &'static str {
    match category {
        "email" | "phone" => "contact",
        "credit_card" | "iban" => "financial",
        "ssn" => "government_id",
        "ip_address" => "network_id",
        _ => "personal",
    }
}

/// One thing found: its class, what kind of value, and where.
#[derive(Debug, Clone, PartialEq)]
pub struct DataFinding {
    pub data_class: String,
    /// e.g. "email", "emirates_id", "keyword".
    pub kind: String,
    /// Byte range in the scanned text.
    pub span: (usize, usize),
    pub confidence: f64,
}

impl DataFinding {
    pub fn value<'a>(&self, text: &'a str) -> &'a str {
        text.get(self.span.0..self.span.1).unwrap_or("")
    }
}

/// What a scan found, plus any classes declared for the text.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Classification {
    pub findings: Vec<DataFinding>,
    pub declared: BTreeSet<String>,
}

impl Classification {
    /// Detected classes plus the umbrella names `personal` / `special_category`.
    pub fn classes(&self) -> BTreeSet<String> {
        let mut found: BTreeSet<String> = self.findings.iter().map(|f| f.data_class.clone()).collect();
        found.extend(self.declared.iter().cloned());
        if found.iter().any(|c| is_personal(c)) {
            found.insert("personal".into());
        }
        if found.iter().any(|c| is_special(c)) {
            found.insert("special_category".into());
        }
        found
    }

    pub fn personal(&self) -> bool {
        self.classes().contains("personal")
    }

    pub fn special(&self) -> bool {
        self.classes().contains("special_category")
    }

    /// Both together (Python's `a | b`).
    pub fn union(mut self, other: Classification) -> Classification {
        self.findings.extend(other.findings);
        self.declared.extend(other.declared);
        self
    }

    pub fn summary(&self) -> IndexMap<String, usize> {
        let mut counts = IndexMap::new();
        for f in &self.findings {
            *counts.entry(f.data_class.clone()).or_insert(0) += 1;
        }
        counts
    }
}

/// Deterministic classification with a content-hash cache.
///
/// `extra` adds your own: `{"employee_id": ("government_id", r"EMP-\d{6}")}` —
/// kind → (class, pattern).
pub struct DataClassifier {
    pub keywords: bool,
    pub extra: IndexMap<String, (String, String)>,
    pub cache_size: usize,
    compiled: Vec<(String, String, Regex)>,
    pii: PIIDetector,
    secrets: SecretDetector,
    cache: Mutex<IndexMap<String, Classification>>,
}

impl std::fmt::Debug for DataClassifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DataClassifier").field("keywords", &self.keywords).field("extra", &self.extra).finish()
    }
}

impl Default for DataClassifier {
    fn default() -> Self {
        Self::new(true, IndexMap::new()).expect("no extra patterns to compile")
    }
}

impl DataClassifier {
    /// `extra` patterns are compiled here; a bad one is a configuration error.
    pub fn new(keywords: bool, extra: IndexMap<String, (String, String)>) -> crate::Result<Self> {
        let compiled = extra
            .iter()
            .map(|(kind, (cls, p))| {
                Regex::new(p).map(|r| (kind.clone(), cls.clone(), r)).map_err(|e| crate::Error::config(format!("data class pattern {kind}: {e}")))
            })
            .collect::<crate::Result<Vec<_>>>()?;
        Ok(Self {
            keywords,
            extra,
            cache_size: 4096,
            compiled,
            pii: PIIDetector::default(),
            secrets: SecretDetector::default(),
            cache: Mutex::new(IndexMap::new()),
        })
    }

    /// Add your own class: findings of `pattern` are reported as `class`, of kind `kind`.
    pub fn with_extra(kind: &str, class: &str, pattern: &str) -> crate::Result<Self> {
        Self::new(true, IndexMap::from([(kind.to_string(), (class.to_string(), pattern.to_string()))]))
    }

    /// The classes your `extra` patterns report.
    pub fn extra_classes(&self) -> Vec<String> {
        self.extra.values().map(|(c, _)| c.clone()).collect()
    }

    pub fn scan(&self, text: &str) -> Classification {
        if text.is_empty() {
            return Classification::default();
        }
        let key = super::hex(&blake2::Blake2b::<blake2::digest::consts::U16>::digest(text.as_bytes()));
        {
            let mut cache = self.cache.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(hit) = cache.shift_remove(&key) {
                cache.insert(key, hit.clone());
                return hit;
            }
        }
        let result = Classification { findings: self.scan_uncached(text), declared: BTreeSet::new() };
        let mut cache = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        cache.insert(key, result.clone());
        if cache.len() > self.cache_size {
            cache.shift_remove_index(0);
        }
        result
    }

    fn scan_uncached(&self, text: &str) -> Vec<DataFinding> {
        let mut found = Vec::new();
        let mut claimed: Vec<(usize, usize)> = Vec::new();
        let mut claim = |span: (usize, usize)| {
            if claimed.iter().any(|&(s, e)| span.0 < e && s < span.1) {
                return false;
            }
            claimed.push(span);
            true
        };
        let finding = |class: &str, kind: &str, span: (usize, usize)| DataFinding { data_class: class.into(), kind: kind.into(), span, confidence: 1.0 };
        // Most specific first: a national id is not also "a phone number".
        for id in NATIONAL_IDS.iter() {
            for m in id.pattern.find_iter(text) {
                let span = (m.start(), m.end());
                if (id.valid)(m.as_str()) && claim(span) {
                    found.push(finding("government_id", id.kind, span));
                }
            }
        }
        for (kind, cls, regex) in &self.compiled {
            for m in regex.find_iter(text) {
                if claim((m.start(), m.end())) {
                    found.push(finding(cls, kind, (m.start(), m.end())));
                }
            }
        }
        for f in self.pii.scan(text) {
            let cls = pii_class(&f.category);
            for &span in &f.spans {
                if claim(span) {
                    found.push(finding(cls, &f.category, span));
                }
            }
        }
        for f in self.secrets.scan(text) {
            for &span in &f.spans {
                if claim(span) {
                    found.push(finding("credentials", &f.category, span));
                }
            }
        }
        if self.keywords {
            for (cls, regex) in KEYWORD_RE.iter() {
                for m in regex.find_iter(text) {
                    found.push(DataFinding { data_class: cls.to_string(), kind: "keyword".into(), span: (m.start(), m.end()), confidence: 0.6 });
                }
            }
        }
        found
    }

    /// Take the named classes out.
    ///
    /// Identifiers (an email, an Emirates ID) are masked, or pseudonymised
    /// when a `pseudonymizer` is given. A class found only by keyword — health,
    /// religion, the other special categories — has no identifier to mask: the
    /// word "diabetic" is the signal, the statement around it is the data. So
    /// the whole sentence goes, replaced by `[health information removed]`.
    pub fn redact<S: AsRef<str>>(&self, text: &str, classes: &[S], pseudonymizer: Option<&Pseudonymizer>, subject: &str) -> String {
        let wanted = expand_classes(classes.iter().map(|c| c.as_ref()));
        let findings: Vec<DataFinding> = self.scan(text).findings.into_iter().filter(|f| wanted.contains(&f.data_class)).collect();
        if findings.is_empty() {
            return text.to_string();
        }
        let mut pieces: Vec<(usize, usize, String)> = Vec::new();
        for f in &findings {
            if f.kind == "keyword" {
                let (start, end) = sentence(text, f.span);
                pieces.push((start, end, format!("[{} information removed]", f.data_class.replace('_', " "))));
            } else if let Some(p) = pseudonymizer {
                pieces.push((f.span.0, f.span.1, p.token(f.value(text), &f.kind, subject)));
            } else {
                pieces.push((f.span.0, f.span.1, format!("[{}]", f.kind)));
            }
        }
        // Widest first at each position, so a removed sentence swallows the
        // identifiers inside it rather than being cut up by them.
        pieces.sort_by(|a, b| a.0.cmp(&b.0).then((b.1 - b.0).cmp(&(a.1 - a.0))));
        let mut out = String::with_capacity(text.len());
        let mut cursor = 0;
        for (start, end, replacement) in pieces {
            if start < cursor {
                continue;
            }
            out.push_str(&text[cursor..start]);
            out.push_str(&replacement);
            cursor = end;
        }
        out.push_str(&text[cursor..]);
        out
    }
}

fn is_sentence_end(b: u8) -> bool {
    matches!(b, b'.' | b'!' | b'?' | b'\n')
}

/// The bounds of the sentence containing `span`, trailing punctuation included.
fn sentence(text: &str, span: (usize, usize)) -> (usize, usize) {
    let bytes = text.as_bytes();
    let mut start = span.0;
    while start > 0 && !is_sentence_end(bytes[start - 1]) {
        start -= 1;
    }
    while start < span.0 && bytes[start] == b' ' {
        start += 1;
    }
    let end = bytes[span.1..].iter().position(|&b| is_sentence_end(b)).map_or(text.len(), |i| span.1 + i + 1);
    (start, end)
}

const OPEN: char = '\u{27e8}';
const CLOSE: char = '\u{27e9}';

static TOKEN: LazyLock<Regex> = LazyLock::new(|| re("\u{27e8}([a-z_]+):([0-9a-f]{12})\u{27e9}"));
static NOT_KIND: LazyLock<Regex> = LazyLock::new(|| re("[^a-z_]"));

#[derive(Default)]
struct Held {
    vault: HashMap<String, String>,
    by_subject: HashMap<String, BTreeSet<String>>,
}

/// Reversible tokens: `alice@acme.com` → `⟨email:3f9a1c0b2d4e⟩`.
///
/// The same value always gets the same token under one key, so a model can
/// still tell two people apart and refer back to one of them — and the token
/// is swapped back for the value before a tool sees it. Values are held in a
/// vault per subject; `forget(subject)` drops them, after which the tokens in
/// any log are meaningless (crypto-shredding).
pub struct Pseudonymizer {
    key: Vec<u8>,
    held: Mutex<Held>,
}

impl std::fmt::Debug for Pseudonymizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<Pseudonymizer {} tokens>", self.len())
    }
}

impl Default for Pseudonymizer {
    fn default() -> Self {
        Self::new(None)
    }
}

impl Pseudonymizer {
    /// Without a key, a random one: tokens are stable for this process only.
    pub fn new(key: Option<&[u8]>) -> Self {
        let key = key.filter(|k| !k.is_empty()).map_or_else(|| super::random_bytes(32), <[u8]>::to_vec);
        Self { key, held: Mutex::new(Held::default()) }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Held> {
        self.held.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn token(&self, value: &str, kind: &str, subject: &str) -> String {
        let digest = super::hmac_hex(&self.key, format!("{subject}\0{value}").as_bytes());
        let kind = NOT_KIND.replace_all(&kind.to_lowercase(), "_").into_owned();
        let kind = if kind.is_empty() { "value".to_string() } else { kind };
        let token = format!("{OPEN}{kind}:{}{CLOSE}", &digest[..12]);
        let mut held = self.lock();
        held.vault.insert(token.clone(), value.to_string());
        held.by_subject.entry(subject.to_string()).or_default().insert(token.clone());
        token
    }

    pub fn restore(&self, text: &str) -> String {
        if !text.contains(OPEN) {
            return text.to_string();
        }
        let held = self.lock();
        TOKEN.replace_all(text, |c: &regex::Captures<'_>| held.vault.get(&c[0]).cloned().unwrap_or_else(|| c[0].to_string())).into_owned()
    }

    /// Restore tokens anywhere inside a JSON value (tool arguments).
    pub fn restore_value(&self, value: &Value) -> Value {
        match value {
            Value::String(s) => Value::String(self.restore(s)),
            Value::Object(m) => Value::Object(m.iter().map(|(k, v)| (k.clone(), self.restore_value(v))).collect()),
            Value::Array(a) => Value::Array(a.iter().map(|v| self.restore_value(v)).collect()),
            other => other.clone(),
        }
    }

    /// Drop everything held for `subject`. Returns how many tokens went.
    pub fn forget(&self, subject: &str) -> usize {
        let mut held = self.lock();
        let tokens = held.by_subject.remove(subject).unwrap_or_default();
        for t in &tokens {
            held.vault.remove(t);
        }
        tokens.len()
    }

    pub fn len(&self) -> usize {
        self.lock().vault.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::guardrails::detectors::luhn_valid;

    /// Append the one check digit that makes `prefix` valid.
    pub(crate) fn complete(prefix: &str, valid: impl Fn(&str) -> bool) -> String {
        (0..10).map(|d| format!("{prefix}{d}")).find(|c| valid(c)).expect("a check digit exists")
    }

    pub(crate) fn emirates_id() -> String {
        let raw = complete("78419901234567", luhn_valid);
        format!("{}-{}-{}-{}", &raw[..3], &raw[3..7], &raw[7..14], &raw[14..])
    }

    #[test]
    fn national_ids_are_checked_by_their_check_digits() {
        assert!(nric_valid("S1234567D") && !nric_valid("S1234567A"));
        assert!(cn_resident_id_valid("11010519491231002X"));
        assert!(!cn_resident_id_valid("110105194912310021"));
        let aadhaar = complete("23412341234", verhoeff_valid);
        let last = aadhaar.chars().last().unwrap().to_digit(10).unwrap();
        assert!(verhoeff_valid(&aadhaar) && !verhoeff_valid(&format!("{}{}", &aadhaar[..11], (last + 1) % 10)));

        let classifier = DataClassifier::default();
        let eid = emirates_id();
        let found = classifier.scan(&format!("Emirates ID {eid}, Aadhaar {aadhaar}, NRIC S1234567D"));
        let kinds: BTreeSet<String> = found.findings.iter().filter(|f| f.data_class == "government_id").map(|f| f.kind.clone()).collect();
        assert_eq!(kinds, BTreeSet::from(["emirates_id".into(), "aadhaar".into(), "sg_nric".into()]));
        let last = eid.chars().last().unwrap().to_digit(10).unwrap();
        let bad = format!("{}{}", &eid[..eid.len() - 1], (last + 1) % 10);
        assert!(!classifier.scan(&format!("ref {bad}")).classes().contains("government_id"));
    }

    #[test]
    fn classes_roll_up_and_special_categories_are_screened() {
        let found = DataClassifier::default().scan("Mail bob@example.com \u{2014} he was diagnosed with diabetes");
        for c in ["contact", "health", "personal", "special_category"] {
            assert!(found.classes().contains(c), "{c}");
        }
        assert!(!DataClassifier::default().scan("the order shipped on 2026-09-01").classes().contains("personal"));
    }

    #[test]
    fn redaction_masks_identifiers_and_pseudonyms_round_trip() {
        let classifier = DataClassifier::default();
        let text = "Write to bob@example.com, card 4111 1111 1111 1111";
        let masked = classifier.redact(text, &["contact"], None, "");
        assert!(!masked.contains("bob@example.com") && masked.contains("[email]") && masked.contains("4111"));

        let vault = Pseudonymizer::new(Some(&[b'k'; 32]));
        let tokenised = classifier.redact(text, &["personal"], Some(&vault), "s1");
        assert!(!tokenised.contains("bob@example.com") && tokenised.contains("\u{27e8}email:"));
        // hmac.new(b"k"*32, b"s1\x00bob@example.com", sha256).hexdigest()[:12]
        assert!(tokenised.contains("\u{27e8}email:f418a93d2642\u{27e9}"), "{tokenised}");
        assert_eq!(vault.restore(&tokenised), text);
        assert_eq!(classifier.redact(text, &["personal"], Some(&vault), "s1"), tokenised);
        assert_eq!(vault.forget("s1"), 2);
        assert_eq!(vault.restore(&tokenised), tokenised);
    }

    #[test]
    fn special_categories_lose_the_whole_sentence() {
        let out = DataClassifier::default().redact("Refund order 12 to bob@example.com. I was diagnosed with cancer. Thanks!", &["health"], None, "");
        assert_eq!(out, "Refund order 12 to bob@example.com. [health information removed] Thanks!");
    }

    #[test]
    fn eu_vat_numbers_with_their_country_prefix() {
        let classifier = DataClassifier::default();
        let kinds: BTreeSet<String> = classifier.scan("VAT DE123456789, NL123456789B01").findings.iter().map(|f| f.kind.clone()).collect();
        assert_eq!(kinds, BTreeSet::from(["eu_vat".to_string()]));
        let bad_it = "IT12345678901";
        let good_it = complete("IT1234567890", |v| luhn(&v[2..]));
        let found = classifier.scan(&format!("{bad_it} {good_it}"));
        let n = found.findings.iter().filter(|f| f.kind == "eu_vat").count();
        assert_eq!(n, usize::from(luhn(&bad_it[2..])) + 1);
    }

    #[test]
    fn your_own_classes() {
        let classifier = DataClassifier::with_extra("emp", "employee_id", r"EMP-\d{6}").unwrap();
        assert!(classifier.scan("badge EMP-123456").classes().contains("employee_id"));
        assert_eq!(classifier.extra_classes(), vec!["employee_id"]);
    }
}
