"""Algorithmic detectors: deterministic, fast, and free.

These run in microseconds and never call a model, so they belong on every path.
Each answers one question and says how sure it is, with the spans it matched —
so a caller can redact precisely rather than discarding a whole message.

Where a cheap check can be made exact, it is: a card number is validated with
Luhn and an IBAN with mod-97, so "4111 1111 1111 1111" is a card and
"4111 1111 1111 1112" is just digits. That is the difference between a detector
you can leave switched on and one everybody turns off.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

__all__ = [
    "Finding",
    "Detector",
    "PIIDetector",
    "SecretDetector",
    "InjectionDetector",
    "ToxicityDetector",
    "GroundednessDetector",
    "RepetitionDetector",
    "shannon_entropy",
    "luhn_valid",
    "iban_valid",
]

Severity = Literal["low", "medium", "high", "critical"]
_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2, "critical": 3}


@dataclass
class Finding:
    """One thing a detector found, and where."""

    detector: str
    category: str
    severity: Severity = "medium"
    score: float = 1.0                 # 0.0-1.0, how sure
    detail: str = ""
    spans: list[tuple[int, int]] = field(default_factory=list)
    samples: list[str] = field(default_factory=list)

    def at_least(self, severity: Severity) -> bool:
        return _ORDER[self.severity] >= _ORDER[severity]

    def line(self) -> str:
        return f"{self.category} ({self.severity}, {self.score:.0%}): {self.detail}"


@runtime_checkable
class Detector(Protocol):
    """Anything that inspects text and reports what it found."""

    name: str

    def scan(self, text: str, **context: Any) -> list[Finding]: ...


# --- helpers -------------------------------------------------------------------

def shannon_entropy(value: str) -> float:
    """Bits per character. High entropy is what a key looks like and prose does not."""
    if not value:
        return 0.0
    counts: dict[str, int] = {}
    for char in value:
        counts[char] = counts.get(char, 0) + 1
    length = len(value)
    return -sum((n / length) * math.log2(n / length) for n in counts.values())


def luhn_valid(digits: str) -> bool:
    """The checksum every real card number satisfies."""
    numbers = [int(c) for c in digits if c.isdigit()]
    if not 12 <= len(numbers) <= 19:
        return False
    total, parity = 0, len(numbers) % 2
    for index, digit in enumerate(numbers):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def iban_valid(value: str) -> bool:
    """IBAN mod-97: the check that separates an account number from noise."""
    cleaned = re.sub(r"\s+", "", value).upper()
    if not 15 <= len(cleaned) <= 34 or not cleaned[:2].isalpha():
        return False
    rotated = cleaned[4:] + cleaned[:4]
    try:
        numeric = "".join(str(int(c, 36)) for c in rotated)
    except ValueError:
        return False
    return int(numeric) % 97 == 1


def _redact(text: str, spans: Sequence[tuple[int, int]], token: str) -> str:
    """Replace spans back to front so earlier offsets stay valid."""
    out = text
    for start, end in sorted(spans, reverse=True):
        out = out[:start] + token + out[end:]
    return out


# --- personal data ----------------------------------------------------------------

@dataclass
class PIIDetector:
    """Personal data, with the cheap exact checks actually applied.

    Cards are Luhn-checked and IBANs mod-97-checked, so an order number that
    happens to be sixteen digits is not reported as a credit card.
    """

    name: str = "pii"
    #: Scanned in this order, and an earlier category claims the characters it
    #: matched. Without that a card number is also "a phone number", which makes
    #: every report noisy and every severity wrong.
    categories: tuple[str, ...] = (
        "credit_card", "iban", "ssn", "email", "ip_address", "phone")
    severity: Severity = "high"

    PATTERNS: dict[str, re.Pattern[str]] = field(default_factory=lambda: {
        "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b"),
        "phone": re.compile(
            r"(?<![\w.])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?"
            r"\d{3,4}[\s.-]\d{3,4}(?:[\s.-]\d{2,4})?(?![\w.])"),
        "credit_card": re.compile(r"\b(?:\d[ -]?){12,19}\b"),
        "iban": re.compile(r"\b[A-Z]{2}\d{2}[ ]?(?:[A-Z0-9]{4}[ ]?){2,7}[A-Z0-9]{1,4}\b"),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "ip_address": re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
    })

    #: A phone number has this many digits. Anything longer is an account
    #: number, an order id, or something else that is not a phone.
    phone_digits: tuple[int, int] = (7, 15)

    RUN_CHARS = frozenset(" -.0123456789")

    def _acceptable(self, category: str, value: str, text: str,
                    span: tuple[int, int]) -> bool:
        """The exact checks. Without them these categories are noise generators."""
        if category == "credit_card":
            return luhn_valid(value)
        if category == "iban":
            return iban_valid(value)
        if category == "phone":
            # Count the digits in the *whole* run this match sits inside, not
            # just the part the pattern happened to take. A 24-digit reference
            # contains something phone-shaped; it is not a phone number.
            start, end = span
            while start > 0 and text[start - 1] in self.RUN_CHARS:
                start -= 1
            while end < len(text) and text[end] in self.RUN_CHARS:
                end += 1
            low, high = self.phone_digits
            return low <= sum(c.isdigit() for c in text[start:end]) <= high
        return True

    def scan(self, text: str, **context: Any) -> list[Finding]:
        found: list[Finding] = []
        claimed: list[tuple[int, int]] = []
        for category in self.categories:
            pattern = self.PATTERNS.get(category)
            if pattern is None:
                continue
            spans, samples = [], []
            for match in pattern.finditer(text):
                value, span = match.group(0), match.span()
                if not self._acceptable(category, value, text, span):
                    continue
                # An earlier, more specific category already owns these characters.
                if any(span[0] < end and start < span[1] for start, end in claimed):
                    continue
                claimed.append(span)
                spans.append(span)
                samples.append(value[:4] + "…")
            if spans:
                found.append(Finding(
                    self.name, category,
                    severity="critical" if category in {"credit_card", "ssn", "iban"}
                    else self.severity,
                    score=1.0, spans=spans, samples=samples[:3],
                    detail=f"{len(spans)} {category.replace('_', ' ')} "
                           f"{'value' if len(spans) == 1 else 'values'} found"))
        return found

    def redact(self, text: str, **context: Any) -> str:
        for finding in self.scan(text):
            text = _redact(text, finding.spans, f"[{finding.category}]")
        return text


# --- credentials ---------------------------------------------------------------------

@dataclass
class SecretDetector:
    """Known key formats, plus anything that looks like a key by entropy."""

    name: str = "secret"
    min_entropy: float = 4.0
    min_length: int = 24
    severity: Severity = "critical"

    KNOWN: dict[str, re.Pattern[str]] = field(default_factory=lambda: {
        "anthropic": re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}"),
        "openai": re.compile(r"\bsk-(?!ant-)[A-Za-z0-9]{20,}\b"),
        "google": re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b"),
        "aws": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "github": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
        "slack": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
        "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\."
                          r"[A-Za-z0-9_-]{10,}\b"),
    })
    CANDIDATE: re.Pattern[str] = field(
        default_factory=lambda: re.compile(r"\b[A-Za-z0-9+/_\-]{24,}={0,2}\b"))

    def scan(self, text: str, **context: Any) -> list[Finding]:
        found: list[Finding] = []
        covered: list[tuple[int, int]] = []
        for label, pattern in self.KNOWN.items():
            spans = [m.span() for m in pattern.finditer(text)]
            if spans:
                covered += spans
                found.append(Finding(self.name, f"{label}_key", severity="critical",
                                     score=1.0, spans=spans,
                                     detail=f"a {label} credential is in the text"))

        # Anything else long and random enough to be a key we do not recognise.
        entropic: list[tuple[int, int]] = []
        for match in self.CANDIDATE.finditer(text):
            span, value = match.span(), match.group(0)
            if any(span[0] < end and start < span[1] for start, end in covered):
                continue
            if len(value) >= self.min_length and shannon_entropy(value) >= self.min_entropy:
                entropic.append(span)
        if entropic:
            found.append(Finding(
                self.name, "high_entropy", severity="medium",
                score=min(1.0, 0.4 + 0.2 * len(entropic)), spans=entropic,
                detail=f"{len(entropic)} high-entropy strings that look like keys"))
        return found

    def redact(self, text: str, **context: Any) -> str:
        for finding in self.scan(text):
            text = _redact(text, finding.spans, "[redacted]")
        return text


# --- prompt injection -------------------------------------------------------------------

@dataclass
class InjectionDetector:
    """Prompt injection, scored rather than matched.

    One suspicious phrase is weak evidence; three together is not. Signals are
    weighted and summed into a 0-1 score, which is reported honestly instead of
    being dressed up as certainty — no text-only detector can be certain, and
    the real defence is the tool allowlist and the policy gate.
    """

    name: str = "injection"
    threshold: float = 0.5
    severity: Severity = "medium"

    SIGNALS: tuple[tuple[str, str, float], ...] = (
        ("override", r"ignore (?:all |any )?(?:previous|prior|above|earlier) "
                     r"(?:instructions?|prompts?|rules?)", 0.45),
        ("override", r"disregard (?:everything|all|the) (?:above|previous|prior)", 0.45),
        ("role_change", r"you are now (?:a|an|the|my) [\w ]{3,40}", 0.3),
        ("role_change", r"(?:act|behave|pretend to be) as (?:if you are |a |an )", 0.25),
        ("exfiltration", r"(?:reveal|print|repeat|show|output|tell me) "
                         r"(?:me )?(?:your |the )?(?:system |initial )?"
                         r"(?:prompt|instructions|rules)", 0.5),
        ("exfiltration", r"what (?:were|are) your (?:original |initial )?instructions",
         0.45),
        ("delimiter", r"(?:^|\n)\s*(?:###|---|```)?\s*(?:system|assistant)\s*:", 0.3),
        ("delimiter", r"<\s*/?\s*(?:system|instructions?|prompt)\s*>", 0.35),
        ("jailbreak", r"\b(?:DAN|do anything now|developer mode|jailbreak)\b", 0.4),
        ("jailbreak", r"without (?:any )?(?:restrictions?|filters?|limitations?)", 0.3),
        ("tool_abuse", r"(?:call|run|execute|invoke) (?:the )?"
                       r"(?:shell|bash|rm|delete|drop table)", 0.4),
        ("encoding", r"\b(?:base64|rot13|hex)\s*(?:decode|encoded?)\b", 0.25),
    )

    def scan(self, text: str, **context: Any) -> list[Finding]:
        score = 0.0
        hit: dict[str, list[tuple[int, int]]] = {}
        for category, pattern, weight in self.SIGNALS:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                hit.setdefault(category, []).append(match.span())
                score += weight
                break                       # one hit per signal, not per repetition
        if not hit:
            return []
        score = min(1.0, score)
        severity: Severity = ("high" if score >= 0.8 else
                              "medium" if score >= self.threshold else "low")
        spans = [span for spans in hit.values() for span in spans]
        return [Finding(self.name, "prompt_injection", severity=severity, score=score,
                        spans=spans,
                        detail=f"{len(hit)} injection signals ({', '.join(sorted(hit))})")]

    def is_injection(self, text: str) -> bool:
        findings = self.scan(text)
        return bool(findings) and findings[0].score >= self.threshold


# --- tone ---------------------------------------------------------------------------------

@dataclass
class ToxicityDetector:
    """A screen, not a classifier.

    Catches obvious abuse in the text an agent is about to send, including the
    usual character substitutions. It will not catch everything — pair it with
    `LLMGuard` when the cost of missing something is real.
    """

    name: str = "toxicity"
    severity: Severity = "high"
    terms: tuple[str, ...] = (
        "idiot", "moron", "stupid bitch", "bastard", "scumbag",
        "kill yourself", "kys", "go die", "worthless piece",
    )
    SUBSTITUTIONS = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a",
                                   "5": "s", "7": "t", "@": "a", "$": "s"})

    def normalise(self, text: str) -> str:
        lowered = text.lower().translate(self.SUBSTITUTIONS)
        return re.sub(r"[^a-z\s]", "", lowered)

    def scan(self, text: str, **context: Any) -> list[Finding]:
        normalised = self.normalise(text)
        hits = [term for term in self.terms
                if re.search(rf"\b{re.escape(self.normalise(term))}\b", normalised)]
        if not hits:
            return []
        return [Finding(self.name, "toxicity", severity=self.severity,
                        score=min(1.0, 0.5 + 0.25 * len(hits)), samples=hits[:3],
                        detail=f"{len(hits)} abusive terms")]


# --- is the answer supported? ----------------------------------------------------------

@dataclass
class GroundednessDetector:
    """Does the answer stay inside what the sources actually said?

    Content words in the answer that appear nowhere in the sources are the ones
    worth looking at. A blunt measure, but it catches the answer that invented a
    number, and it costs nothing.
    """

    name: str = "groundedness"
    min_overlap: float = 0.6
    severity: Severity = "medium"
    STOPWORDS = frozenset(
        "the a an and or but if then than that this these those is are was were be "
        "been being to of in on at by for with from as it its they them their there "
        "here we you i he she his her our your not no yes can could would should may "
        "might will shall do does did done have has had about into over under more "
        "most some any each other same so such only own very just".split())

    def _content_words(self, text: str) -> set[str]:
        # Trailing punctuation has to go, or "Singapore." and "Singapore" read as
        # two different words and every sentence-final term looks unsupported.
        words = (w.strip(".'-") for w in
                 re.findall(r"[a-z0-9][a-z0-9'\-.]*", text.lower()))
        return {w for w in words if len(w) > 2 and w not in self.STOPWORDS}

    def scan(self, text: str, *, sources: Iterable[str] = (),
             **context: Any) -> list[Finding]:
        corpus = " ".join(sources)
        if not corpus.strip() or not text.strip():
            return []
        answer_words = self._content_words(text)
        if not answer_words:
            return []
        supported = answer_words & self._content_words(corpus)
        overlap = len(supported) / len(answer_words)
        if overlap >= self.min_overlap:
            return []
        unsupported = sorted(answer_words - supported)
        return [Finding(
            self.name, "unsupported_claim",
            severity="high" if overlap < self.min_overlap / 2 else self.severity,
            score=round(1.0 - overlap, 3), samples=unsupported[:6],
            detail=f"only {overlap:.0%} of the answer is supported by the sources")]


# --- is it stuck? ---------------------------------------------------------------------------

@dataclass
class RepetitionDetector:
    """A model looping on itself — the same phrase, over and over."""

    name: str = "repetition"
    window: int = 6              # n-gram size
    max_ratio: float = 0.3       # of n-grams, how many may be repeats
    severity: Severity = "low"

    def scan(self, text: str, **context: Any) -> list[Finding]:
        words = text.split()
        if len(words) < self.window * 3:
            return []
        grams = [" ".join(words[i:i + self.window])
                 for i in range(len(words) - self.window + 1)]
        unique = len(set(grams))
        repeated = 1.0 - (unique / len(grams))
        if repeated <= self.max_ratio:
            return []
        counts: dict[str, int] = {}
        for gram in grams:
            counts[gram] = counts.get(gram, 0) + 1
        worst = max(counts, key=lambda g: counts[g])
        return [Finding(self.name, "repetition", severity=self.severity,
                        score=round(repeated, 3), samples=[worst],
                        detail=f"{repeated:.0%} of the answer repeats itself")]
