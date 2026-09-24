"""What kind of data is this? Classification, minimisation and pseudonymisation.

    classifier = DataClassifier()
    found = classifier.scan("Ahmed, Emirates ID 784-1990-1234567-1, is diabetic")
    found.classes          # {'government_id', 'health', 'personal', 'special_category'}

Everything the residency, purpose and redaction rules decide on starts here,
so the detectors are the exact kind where exactness is possible: card numbers
are Luhn-checked, IBANs mod-97-checked, and national identity numbers are
validated by their own check digits — Emirates ID (Luhn), Saudi national ID
and Iqama (Luhn), Aadhaar (Verhoeff), Singapore NRIC/FIN (weighted mod 11),
the Chinese resident ID (ISO 7064 mod 11-2). A sixteen-digit order number is
not reported as a card and a random twelve digits is not an Aadhaar.

Special-category data (health, religion, ethnicity, ...) cannot be proven by a
pattern. Those detectors are keyword screens, report a lower confidence, and
say so; pair them with an `LLMGuard` if you need judgement rather than recall.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from ..guardrails.detectors import PIIDetector, SecretDetector

__all__ = [
    "DATA_CLASSES", "SPECIAL_CATEGORIES", "PERSONAL", "expand_classes",
    "DataFinding", "Classification", "DataClassifier", "Pseudonymizer",
    "luhn", "verhoeff_valid", "nric_valid", "cn_resident_id_valid",
]

#: The taxonomy. Keys are what policies and identities name.
DATA_CLASSES: dict[str, str] = {
    "contact": "email addresses, phone numbers",
    "financial": "card numbers, bank accounts (IBAN)",
    "government_id": "national identity numbers, social security numbers",
    "network_id": "IP addresses and device identifiers",
    "credentials": "API keys, tokens, private keys, passwords",
    "health": "diagnoses, conditions, treatment, medication",
    "biometric": "fingerprints, face and voice prints",
    "genetic": "DNA and genetic test results",
    "religion": "religious or philosophical belief",
    "ethnicity": "racial or ethnic origin",
    "political": "political opinion",
    "sexual_orientation": "sex life or sexual orientation",
    "trade_union": "trade union membership",
    "criminal": "criminal convictions and offences",
    "children": "data about minors",
    "location": "precise location",
}

#: GDPR Art 9 (and its equivalents: PIPL "sensitive", DPDP, KSA PDPL "sensitive").
SPECIAL_CATEGORIES: frozenset[str] = frozenset({
    "health", "biometric", "genetic", "religion", "ethnicity", "political",
    "sexual_orientation", "trade_union", "criminal",
})
#: Everything that identifies or describes a person.
PERSONAL: frozenset[str] = frozenset(set(DATA_CLASSES) - {"credentials"})


def expand_classes(names: Iterable[str]) -> set[str]:
    """Resolve the umbrella names `personal` and `special_category`."""
    out: set[str] = set()
    for name in names:
        if name == "personal":
            out |= PERSONAL
        elif name == "special_category":
            out |= SPECIAL_CATEGORIES
        else:
            out.add(name)
    return out


# ---- check digits -----------------------------------------------------------

def luhn(digits: str) -> bool:
    """Luhn mod 10 at any length (the card detector's version insists on 12-19)."""
    numbers = [int(c) for c in digits if c.isdigit()]
    if len(numbers) < 2:
        return False
    total, parity = 0, len(numbers) % 2
    for index, digit in enumerate(numbers):
        if index % 2 == parity:
            digit = digit * 2 - 9 if digit > 4 else digit * 2
        total += digit
    return total % 10 == 0


_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6), (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8), (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2), (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4), (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2), (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0), (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5), (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def verhoeff_valid(digits: str) -> bool:
    """The Verhoeff check Aadhaar numbers carry in their last digit."""
    check = 0
    for index, char in enumerate(reversed(digits)):
        check = _VERHOEFF_D[check][_VERHOEFF_P[index % 8][int(char)]]
    return check == 0


def nric_valid(value: str) -> bool:
    """Singapore NRIC (S/T) and FIN (F/G): weighted sum mod 11 → letter."""
    value = value.upper()
    if len(value) != 9 or value[0] not in "STFG" or not value[1:8].isdigit():
        return False
    total = sum(int(d) * w for d, w in zip(value[1:8], (2, 7, 6, 5, 4, 3, 2), strict=True))
    if value[0] in "TG":
        total += 4
    table = "JZIHGFEDCBA" if value[0] in "ST" else "XWUTRQPNMLK"
    return table[total % 11] == value[8]


def cn_resident_id_valid(value: str) -> bool:
    """China's 18-character resident identity number, ISO 7064 MOD 11-2."""
    value = value.upper()
    if len(value) != 18 or not value[:17].isdigit():
        return False
    weights = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
    total = sum(int(d) * w for d, w in zip(value[:17], weights, strict=True))
    if "10X98765432"[total % 11] != value[17]:
        return False
    return _valid_date(value[6:10], value[10:12], value[12:14])


def _valid_date(year: str, month: str, day: str) -> bool:
    try:
        date(int(year), int(month), int(day))
    except ValueError:
        return False
    return True


def _kr_rrn_valid(value: str) -> bool:
    """Korean resident registration number: a real birth date and sex digit.

    RRNs issued since October 2020 no longer carry a check digit, so the date
    and century/sex digit are what can be verified.
    """
    digits = value.replace("-", "")
    century = {"1": "19", "2": "19", "3": "20", "4": "20", "5": "19", "6": "19",
               "7": "20", "8": "20", "9": "18", "0": "18"}.get(digits[6])
    return century is not None and _valid_date(century + digits[:2], digits[2:4],
                                               digits[4:6])


@dataclass(frozen=True)
class _IdPattern:
    kind: str
    pattern: re.Pattern[str]
    valid: Callable[[str], bool]


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value)


_NATIONAL_IDS: tuple[_IdPattern, ...] = (
    _IdPattern("emirates_id", re.compile(r"\b784[- ]?\d{4}[- ]?\d{7}[- ]?\d\b"),
               lambda v: luhn(_digits(v))),
    _IdPattern("cn_resident_id", re.compile(r"\b\d{17}[\dXx]\b"), cn_resident_id_valid),
    _IdPattern("aadhaar", re.compile(r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b"),
               lambda v: verhoeff_valid(_digits(v))),
    _IdPattern("saudi_id", re.compile(r"\b[12]\d{9}\b"), luhn),
    _IdPattern("sg_nric", re.compile(r"\b[STFGstfg]\d{7}[A-Za-z]\b"), nric_valid),
    _IdPattern("kr_rrn", re.compile(r"\b\d{6}-[1-8]\d{6}\b"), _kr_rrn_valid),
    _IdPattern("uk_nino", re.compile(
        r"\b(?!BG|GB|NK|KN|TN|NT|ZZ)[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z] ?\d{2} ?\d{2} ?\d{2} ?[A-D]\b"),
        lambda v: True),
)

#: Keyword screens for what no pattern can prove. Word-bounded, case-folded.
_KEYWORDS: dict[str, tuple[str, ...]] = {
    "health": (
        "diagnosed", "diagnosis", "diabetes", "diabetic", "cancer", "chemotherapy",
        "hiv", "aids", "hepatitis", "depression", "anxiety disorder", "schizophrenia",
        "bipolar", "pregnant", "pregnancy", "prescription", "prescribed", "medication",
        "blood pressure", "blood test", "mri", "surgery", "disability", "therapy session",
        "medical record", "patient", "icd-10", "asthma", "dementia", "alzheimer",
    ),
    "biometric": ("fingerprint", "face print", "faceprint", "voiceprint", "iris scan",
                  "retina scan", "facial recognition template", "biometric"),
    "genetic": ("dna", "genome", "genetic test", "brca", "genotype"),
    "religion": ("muslim", "christian", "jewish", "hindu", "buddhist", "sikh",
                 "atheist", "religion", "religious belief", "converted to"),
    "ethnicity": ("ethnicity", "ethnic origin", "racial origin", "race:"),
    "political": ("political party", "voted for", "party member", "political opinion",
                  "political affiliation"),
    "sexual_orientation": ("gay", "lesbian", "bisexual", "transgender", "sexual orientation",
                           "homosexual", "lgbt"),
    "trade_union": ("trade union", "union member", "labour union", "labor union"),
    "criminal": ("convicted", "conviction", "criminal record", "arrested", "sentenced to",
                 "prison sentence", "offence", "felony"),
    "children": ("my son", "my daughter", "years old child", "minor aged", "under 18",
                 "under 16", "pupil", "schoolchild", "kindergarten"),
}
_KEYWORD_RE: dict[str, re.Pattern[str]] = {
    cls: re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b", re.I)
    for cls, words in _KEYWORDS.items()
}
_PII_TO_CLASS = {"email": "contact", "phone": "contact", "credit_card": "financial",
                 "iban": "financial", "ssn": "government_id", "ip_address": "network_id"}


@dataclass(frozen=True)
class DataFinding:
    data_class: str
    kind: str                     # e.g. "email", "emirates_id", "keyword"
    span: tuple[int, int]
    confidence: float = 1.0

    def value(self, text: str) -> str:
        return text[self.span[0]:self.span[1]]


@dataclass(frozen=True)
class Classification:
    findings: tuple[DataFinding, ...] = ()
    declared: frozenset[str] = frozenset()

    @property
    def classes(self) -> frozenset[str]:
        """Detected classes plus the umbrella names `personal` / `special_category`."""
        found = {f.data_class for f in self.findings} | set(self.declared)
        if found & PERSONAL:
            found.add("personal")
        if found & SPECIAL_CATEGORIES:
            found.add("special_category")
        return frozenset(found)

    @property
    def personal(self) -> bool:
        return "personal" in self.classes

    @property
    def special(self) -> bool:
        return "special_category" in self.classes

    def __or__(self, other: Classification) -> Classification:
        return Classification(self.findings + other.findings,
                              self.declared | other.declared)

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in self.findings:
            counts[f.data_class] = counts.get(f.data_class, 0) + 1
        return counts


@dataclass
class DataClassifier:
    """Deterministic classification with a content-hash cache.

    `extra` adds your own: ``{"employee_id": ("government_id", r"EMP-\\d{6}")}``.
    """

    keywords: bool = True
    extra: dict[str, tuple[str, str]] = field(default_factory=dict)
    cache_size: int = 4096

    def __post_init__(self) -> None:
        self._pii = PIIDetector()
        self._secrets = SecretDetector()
        self._extra = [(kind, cls, re.compile(p)) for kind, (cls, p) in self.extra.items()]
        self._cache: OrderedDict[str, Classification] = OrderedDict()

    def scan(self, text: str) -> Classification:
        if not text:
            return Classification()
        key = hashlib.blake2b(text.encode("utf-8", "surrogatepass"),
                              digest_size=16).hexdigest()
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        result = Classification(tuple(self._scan(text)))
        self._cache[key] = result
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return result

    def _scan(self, text: str) -> list[DataFinding]:
        found: list[DataFinding] = []
        claimed: list[tuple[int, int]] = []

        def claim(span: tuple[int, int]) -> bool:
            if any(span[0] < end and start < span[1] for start, end in claimed):
                return False
            claimed.append(span)
            return True

        # Most specific first: a national id is not also "a phone number".
        for pattern in _NATIONAL_IDS:
            for match in pattern.pattern.finditer(text):
                if pattern.valid(match.group(0)) and claim(match.span()):
                    found.append(DataFinding("government_id", pattern.kind, match.span()))
        for kind, cls, regex in self._extra:
            for match in regex.finditer(text):
                if claim(match.span()):
                    found.append(DataFinding(cls, kind, match.span()))
        for finding in self._pii.scan(text):
            cls = _PII_TO_CLASS.get(finding.category, "personal")
            for span in finding.spans:
                if claim(span):
                    found.append(DataFinding(cls, finding.category, span))
        for finding in self._secrets.scan(text):
            for span in finding.spans:
                if claim(span):
                    found.append(DataFinding("credentials", finding.category, span))
        if self.keywords:
            for cls, regex in _KEYWORD_RE.items():
                for match in regex.finditer(text):
                    found.append(DataFinding(cls, "keyword", match.span(), confidence=0.6))
        return found

    def redact(self, text: str, classes: Iterable[str], *,
               pseudonymizer: Pseudonymizer | None = None,
               subject: str = "") -> str:
        """Take the named classes out. Keyword findings are left in place.

        A keyword ("diabetic") is the *signal* that a sentence is about health,
        not the identifier; masking the word would mangle the text and protect
        nobody. Redact the identifiers and the sentence stops being about a
        person.
        """
        wanted = expand_classes(classes)
        spans = [f for f in self.scan(text).findings
                 if f.data_class in wanted and f.kind != "keyword"]
        if not spans:
            return text
        out, cursor = [], 0
        for finding in sorted(spans, key=lambda f: f.span[0]):
            start, end = finding.span
            if start < cursor:
                continue
            out.append(text[cursor:start])
            value = text[start:end]
            if pseudonymizer is not None:
                out.append(pseudonymizer.token(value, finding.kind, subject=subject))
            else:
                out.append(f"[{finding.kind}]")
            cursor = end
        out.append(text[cursor:])
        return "".join(out)


_TOKEN = re.compile(r"⟨([a-z_]+):([0-9a-f]{12})⟩")


class Pseudonymizer:
    """Reversible tokens: ``alice@acme.com`` → ``⟨email:3f9a1c0b2d4e⟩``.

    The same value always gets the same token under one key, so a model can
    still tell two people apart and refer back to one of them — and the token
    is swapped back for the value before a tool sees it. Values are held in a
    vault per subject; `forget(subject)` drops them, after which the tokens in
    any log are meaningless (crypto-shredding for the stdlib).
    """

    def __init__(self, key: bytes | str | None = None) -> None:
        if isinstance(key, str):
            key = key.encode()
        self._key = key or secrets.token_bytes(32)
        self._vault: dict[str, str] = {}
        self._by_subject: dict[str, set[str]] = {}

    def token(self, value: str, kind: str = "value", *, subject: str = "") -> str:
        digest = hmac.new(self._key, f"{subject}\x00{value}".encode(),
                          hashlib.sha256).hexdigest()[:12]
        kind = re.sub(r"[^a-z_]", "_", kind.lower()) or "value"
        token = f"⟨{kind}:{digest}⟩"
        self._vault[token] = value
        self._by_subject.setdefault(subject, set()).add(token)
        return token

    def restore(self, text: str) -> str:
        if "⟨" not in text:
            return text
        return _TOKEN.sub(lambda m: self._vault.get(m.group(0), m.group(0)), text)

    def restore_value(self, value: Any) -> Any:
        """Restore tokens anywhere inside a JSON-like value (tool arguments)."""
        if isinstance(value, str):
            return self.restore(value)
        if isinstance(value, dict):
            return {k: self.restore_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.restore_value(v) for v in value]
        return value

    def forget(self, subject: str) -> int:
        tokens = self._by_subject.pop(subject, set())
        for token in tokens:
            self._vault.pop(token, None)
        return len(tokens)

    def __len__(self) -> int:
        return len(self._vault)
