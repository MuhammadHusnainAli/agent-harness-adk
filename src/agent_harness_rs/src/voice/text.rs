//! Turning a model's stream of tokens into things worth saying aloud.
//!
//! A voice agent cannot wait for the whole answer before it starts speaking,
//! and cannot speak half a word. [`SpeechChunker`] hands over text a sentence
//! at a time as it arrives — and the first piece sooner still, at a clause,
//! because the gap before the first sound is the one people notice.

use std::sync::LazyLock;

use regex::Regex;

// Python's patterns end in `(?=\s)`; Rust's regex has no look-ahead, so the
// character after a match is checked by hand.
static SENTENCE: LazyLock<Regex> = LazyLock::new(|| Regex::new("[.!?…]+[\"'”’)\\]]*|\n+").expect("valid"));
static CLAUSE: LazyLock<Regex> = LazyLock::new(|| Regex::new("[,;:—–]").expect("valid"));
/// A full stop after one of these is not the end of a sentence.
const ABBREVIATIONS: [&str; 16] = ["mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "no", "inc", "ltd", "approx"];

static LINK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[([^\]]+)\]\([^)]+\)").expect("valid"));
static URL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"https?://\S+").expect("valid"));
static CODE_FENCE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)```.*?```").expect("valid"));
static MARKS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[*_`#>|~]+").expect("valid"));
static BULLET: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?m)^\s*(?:[-•]|\d+[.)])\s+").expect("valid"));
static SPACE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+").expect("valid"));

/// Text with what cannot be said removed: markdown, code fences, raw URLs.
pub fn speakable(text: &str) -> String {
    let t = CODE_FENCE.replace_all(text, " ");
    let t = LINK.replace_all(&t, "$1");
    let t = URL.replace_all(&t, "the link");
    let t = BULLET.replace_all(&t, "");
    let t = MARKS.replace_all(&t, "");
    SPACE.replace_all(&t, " ").trim().to_string()
}

fn chars(s: &str) -> usize {
    s.chars().count()
}

/// The byte offset of character `n` (or the end).
fn byte_at(s: &str, n: usize) -> usize {
    s.char_indices().nth(n).map_or(s.len(), |(i, _)| i)
}

fn followed_by_space(text: &str, end: usize) -> bool {
    text[end..].chars().next().is_some_and(char::is_whitespace)
}

/// Feed it deltas; it gives back pieces ready for a voice.
///
/// - `first_min`: the first piece may end at a comma once it is this long.
/// - `min_chars`: a later piece shorter than this waits for the next sentence
///   — a voice that says "Yes." and stops sounds broken.
/// - `max_chars`: a sentence longer than this is cut at a space, so a model
///   that never uses a full stop still gets spoken.
#[derive(Debug, Clone)]
pub struct SpeechChunker {
    pub first_min: usize,
    pub min_chars: usize,
    pub max_chars: usize,
    buffer: String,
    emitted: usize,
}

impl Default for SpeechChunker {
    fn default() -> Self {
        Self::new(28, 12, 240)
    }
}

impl SpeechChunker {
    pub fn new(first_min: usize, min_chars: usize, max_chars: usize) -> Self {
        Self { first_min, min_chars, max_chars, buffer: String::new(), emitted: 0 }
    }

    pub fn feed(&mut self, delta: &str) -> Vec<String> {
        self.buffer.push_str(delta);
        let mut out = vec![];
        while let Some(cut) = self.cut() {
            let rest = self.buffer.split_off(cut);
            let piece = std::mem::replace(&mut self.buffer, rest);
            let said = speakable(&piece);
            if said.chars().any(char::is_alphanumeric) {
                out.push(said);
                self.emitted += 1;
            }
        }
        out
    }

    /// The stream has ended: say whatever is left.
    pub fn flush(&mut self) -> Vec<String> {
        let said = speakable(&std::mem::take(&mut self.buffer));
        if said.chars().any(char::is_alphanumeric) {
            self.emitted += 1;
            vec![said]
        } else {
            vec![]
        }
    }

    /// Where to cut the buffer, as a byte offset, if anything is ready.
    fn cut(&self) -> Option<usize> {
        let full = self.buffer.as_str();
        // Code is not spoken, and a fence is never split: it leaves as one
        // piece, which `speakable` then drops whole.
        let fence = full.find("```");
        let text = match fence {
            Some(0) => return full[3..].find("```").map(|c| c + 6),
            Some(f) => {
                let before = &full[..f];
                if before.trim().is_empty() {
                    return Some(f);
                }
                before
            }
            None if full.ends_with('`') => return None, // perhaps the start of a fence: wait
            None => full,
        };
        for m in SENTENCE.find_iter(text) {
            let newline = m.as_str().starts_with('\n');
            if !newline && !followed_by_space(text, m.end()) {
                continue;
            }
            if chars(&text[..m.end()]) < self.min_chars && !m.as_str().contains('\n') {
                continue;
            }
            let head = &text[..m.start()];
            let before = head.split_whitespace().last().unwrap_or("").to_lowercase();
            if m.as_str().starts_with('.') && (ABBREVIATIONS.contains(&before.trim_end_matches('.')) || chars(&before) == 1) {
                continue;
            }
            return Some(m.end());
        }
        if self.emitted == 0 && chars(text) >= self.first_min {
            for m in CLAUSE.find_iter(text) {
                if followed_by_space(text, m.end()) && chars(&text[..m.end()]) >= self.first_min {
                    return Some(m.end());
                }
            }
        }
        if chars(text) > self.max_chars {
            let limit = byte_at(text, self.max_chars);
            return Some(match text[..limit].rfind(' ') {
                Some(space) if space > 0 => space + 1,
                _ => limit,
            });
        }
        fence.filter(|f| *f > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(deltas: &[&str]) -> Vec<String> {
        let mut c = SpeechChunker::default();
        let mut out: Vec<String> = deltas.iter().flat_map(|d| c.feed(d)).collect();
        out.extend(c.flush());
        out
    }

    #[test]
    fn sentences_clauses_and_what_cannot_be_said() {
        assert_eq!(speakable("**Bold** [the docs](https://x.y) at https://a.b/c\n- item `code`"), "Bold the docs at the link item code");
        let text = "Sure, I found your booking for the twelfth. It is at Dr. Smith's office. See you then!";
        let pieces = chunk(&text.split_inclusive(' ').collect::<Vec<_>>());
        assert_eq!(pieces, vec!["Sure, I found your booking for the twelfth.", "It is at Dr. Smith's office.", "See you then!"]);
        // The first piece leaves at a clause once it is long enough.
        let first = chunk(&["Well, let me look that up for you, ", "it may take a moment"]);
        assert_eq!(first[0], "Well, let me look that up for you,");
        // A fence is never split and never spoken.
        assert_eq!(chunk(&["Here it is. ", "```py\nprint(1)\n", "```", " Done now, thanks."]), vec!["Here it is.", "Done now, thanks."]);
        // A model that never stops gets cut at a space.
        let long = "word ".repeat(80);
        let cut = chunk(&[long.as_str()]);
        assert_eq!(cut.iter().map(|c| c.chars().count()).collect::<Vec<_>>(), vec![239, 159], "as Python cuts it");
    }
}
