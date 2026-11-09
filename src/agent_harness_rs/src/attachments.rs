//! Getting an attachment to a model in a form the model can take.
//!
//! Where the model reads that kind of thing — Claude and GPT a PDF, Gemini a
//! recording or a video — it is sent as it is. Where it cannot, it is turned
//! into text first: a text file is read, a recording is transcribed if the
//! harness has been given a [`Transcriber`].
//!
//! Not yet ported: reading PDF / Word / spreadsheet *text* for a model that
//! cannot take the file natively (Python's `toolkits.documents`). Such a file
//! is refused with a clear error until then.

use std::sync::Arc;

use async_trait::async_trait;

use crate::errors::{Error, Result};
use crate::llm_providers::Provider;
use crate::types::{ContentBlock, MediaBlock, MediaKind};

/// The most that is sent inline. Past this a provider wants an upload.
pub const MAX_INLINE_BYTES: u64 = 20 * 1024 * 1024;

const TEXT_LIMIT: usize = 400_000;

/// Something that turns a recording into text.
#[async_trait]
pub trait Transcriber: Send + Sync {
    async fn transcribe(&self, audio: &[u8], media_type: &str) -> Result<String>;
}

fn label(kind: MediaKind, m: &MediaBlock) -> String {
    if !m.name.is_empty() { m.name.clone() } else { m.url.clone().unwrap_or_else(|| kind.as_str().into()) }
}

fn is_plain_text(m: &MediaBlock) -> bool {
    let t = m.media_type.as_str();
    t.starts_with("text/")
        || matches!(t, "application/json" | "application/xml" | "application/x-yaml" | "application/yaml" | "application/toml" | "application/javascript")
        || [".md", ".txt", ".csv", ".json", ".yaml", ".yml", ".toml", ".log", ".xml", ".html"].iter().any(|ext| m.name.to_lowercase().ends_with(ext))
}

/// The text of a file attachment, read on this machine.
pub fn document_text(m: &MediaBlock) -> Result<String> {
    let name = label(MediaKind::Document, m);
    if m.path.is_none() && m.data.is_empty() {
        return Err(Error::config(format!("attachment '{name}' is only a URL, and this model cannot fetch it — pass the file itself")));
    }
    if !is_plain_text(m) {
        return Err(Error::config(format!(
            "attachment '{name}' ({}) cannot be read as text yet in the Rust port — use a model that reads it natively",
            m.media_type
        )));
    }
    let bytes = m.read().map_err(|_| Error::config(format!("attachment '{name}' was not found at {}", m.path.as_deref().unwrap_or("?"))))?;
    let text = String::from_utf8_lossy(&bytes);
    Ok(text.chars().take(TEXT_LIMIT).collect())
}

/// Blocks ready to send to `model`: native where it takes them, text where it
/// does not. A configuration error for what cannot be done either way.
pub async fn prepare_attachments(
    items: &[ContentBlock],
    provider: &dyn Provider,
    model: &str,
    speech: Option<&Arc<dyn Transcriber>>,
) -> Result<Vec<ContentBlock>> {
    let mut out = Vec::new();
    for item in items {
        let Some((kind, block)) = item.as_media() else {
            out.push(item.clone());
            continue;
        };
        let name = label(kind, block);
        if let Some(path) = &block.path
            && !std::path::Path::new(path).is_file()
        {
            return Err(Error::config(format!("attachment '{name}' was not found at {path}")));
        }
        let size = block.size();
        if size > MAX_INLINE_BYTES {
            return Err(Error::config(format!(
                "attachment '{name}' is {:.0} MB; the most that can be sent inline is {} MB — send a smaller file, or a URL the provider can fetch",
                size as f64 / 1_048_576.0,
                MAX_INLINE_BYTES / 1_048_576
            )));
        }
        if provider.accepts(kind, block, model) {
            out.push(item.clone());
        } else if kind == MediaKind::Document {
            let text = document_text(block)?;
            out.push(ContentBlock::text(format!("[Attached file: {name}]\n{text}")));
        } else if kind == MediaKind::Audio {
            let Some(speech) = speech else {
                return Err(Error::config(format!(
                    "{model} cannot listen to '{name}'. Use a model that takes audio (Gemini, or an OpenAI audio model), or give the harness something to transcribe with: Harness::speech(...)"
                )));
            };
            if !block.inline() {
                return Err(Error::config(format!("attachment '{name}' is only a URL — pass the recording itself so it can be transcribed")));
            }
            let text = speech.transcribe(&block.read()?, &block.media_type).await?;
            out.push(ContentBlock::text(format!("[Transcript of the recording {name}]\n{text}")));
        } else {
            let hint = if kind == MediaKind::Video { " — Gemini models take video" } else { "" };
            return Err(Error::config(format!("{model} cannot take {} input, so '{name}' cannot be sent{hint}", kind.as_str())));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_providers::{AnthropicProvider, FakeProvider};
    use crate::types::{attach, attach_bytes};

    #[tokio::test]
    async fn native_text_and_refusals() {
        let claude = AnthropicProvider::new(Some("k".into()));
        let notes = attach_bytes(b"hello notes", "text/plain");
        let out = prepare_attachments(&[attach_bytes(b"png", "image/png"), notes], &claude, "claude-opus-5", None).await.unwrap();
        assert_eq!(out[0].type_name(), "image");
        match &out[1] {
            ContentBlock::Text(t) => assert_eq!(t.text, "[Attached file: document]\nhello notes"),
            other => panic!("{other:?}"),
        }
        let err = prepare_attachments(&[attach_bytes(b"x", "audio/wav")], &claude, "claude-opus-5", None).await.unwrap_err();
        assert!(err.to_string().starts_with("claude-opus-5 cannot listen"));
        let missing = prepare_attachments(&[attach("/no/such/file.png", None)], &claude, "m", None).await.unwrap_err();
        assert!(missing.to_string().contains("was not found"));
        let fake = FakeProvider::new(vec![]);
        assert_eq!(prepare_attachments(&[attach_bytes(b"x", "audio/wav")], &fake, "m", None).await.unwrap().len(), 1);
    }
}
