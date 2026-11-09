//! Document parsing and OCR: get text out of whatever the user dropped in.
//!
//! Text, Markdown, CSV, TSV, JSON, JSONL, HTML and XML parse natively. DOCX is
//! read natively too (the zip and its XML). PDF text comes from `pdf-extract`
//! behind the `pdf` feature (on by default). OCR of scanned images runs the
//! `tesseract` program when it is on PATH — as Python's `pytesseract` does.
//! When a format cannot be read here the error says exactly what unlocks it,
//! instead of failing halfway through a run.

use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};

use regex::Regex;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::errors::{Error, Result};
use crate::pyjson::{Dumps, dumps_with};
use crate::runtime::workspace::{Workspace, on_path};
use crate::tools::{Tool, ToolContext};

const TOOL: &str = "parse_document";

static TAGS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?is)<script[^>]*>.*?</script>|<style[^>]*>.*?</style>").expect("valid"));
static MARKUP: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<[^>]+>").expect("valid"));
static BLANKS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\n{3,}").expect("valid"));
static RUNS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[ \t]{2,}").expect("valid"));

/// Formats read with nothing but this crate.
pub const NATIVE: [&str; 23] = [
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".log",
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".ndjson",
    ".html",
    ".htm",
    ".xml",
    ".yaml",
    ".yml",
    ".py",
    ".js",
    ".ts",
    ".sql",
    ".sh",
    ".toml",
    ".ini",
    ".cfg",
];
const IMAGES: [&str; 7] = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"];
/// The formats that need something more, as Python lists them.
const OPTIONAL: [&str; 7] = [".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tif", ".tiff"];

fn error(message: impl Into<String>) -> Error {
    Error::tool(message, TOOL)
}

fn optional_available(suffix: &str) -> std::result::Result<(), String> {
    match suffix {
        ".pdf" if cfg!(feature = "pdf") => Ok(()),
        ".pdf" => Err("enable the `pdf` feature of agent-harness-adk".into()),
        ".docx" => Ok(()),
        _ if on_path("tesseract") => Ok(()),
        _ => Err("install tesseract and put it on PATH".into()),
    }
}

/// Which formats work right now, and which need an install first:
/// `{"native": [...], "available": [...], "needs_install": [...]}`.
pub fn supported_formats() -> Value {
    let mut native: Vec<&str> = NATIVE.to_vec();
    native.sort_unstable();
    let mut available = Vec::new();
    let mut missing = Vec::new();
    for suffix in OPTIONAL {
        match optional_available(suffix) {
            Ok(()) => available.push(suffix.to_string()),
            Err(how) => missing.push(format!("{suffix} ({how})")),
        }
    }
    available.sort();
    missing.sort();
    json!({"native": native, "available": available, "needs_install": missing})
}

fn html_to_text(body: &str) -> String {
    let stripped = TAGS.replace_all(body, " ");
    let mut stripped = MARKUP.replace_all(&stripped, " ").into_owned();
    for (entity, c) in [("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", "\""), ("&#39;", "'")] {
        stripped = stripped.replace(entity, c);
    }
    BLANKS.replace_all(&RUNS.replace_all(&stripped, " "), "\n\n").trim().to_string()
}

/// Rows of a CSV, as Python's `csv.reader` reads them: quoted fields, doubled
/// quotes, delimiters and newlines inside quotes; a blank line is an empty row.
fn csv_rows(body: &str, delimiter: char) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut started = false; // anything seen on this row
    let mut chars = body.chars().peekable();
    while let Some(c) = chars.next() {
        if quoted {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    quoted = false;
                }
            } else {
                field.push(c);
            }
            continue;
        }
        match c {
            '"' if field.is_empty() => {
                quoted = true;
                started = true;
            }
            c if c == delimiter => {
                row.push(std::mem::take(&mut field));
                started = true;
            }
            '\r' | '\n' => {
                if c == '\r' && chars.peek() == Some(&'\n') {
                    chars.next();
                }
                if started || !field.is_empty() {
                    row.push(std::mem::take(&mut field));
                }
                rows.push(std::mem::take(&mut row));
                started = false;
            }
            c => {
                field.push(c);
                started = true;
            }
        }
    }
    if started || !field.is_empty() {
        row.push(field);
        rows.push(row);
    }
    rows
}

fn csv_to_text(body: &str, delimiter: char, max_rows: usize) -> String {
    let rows = csv_rows(body, delimiter);
    let Some((header, data)) = rows.split_first() else { return String::new() };
    let mut lines = vec![header.join(" | ")];
    lines.extend(data.iter().take(max_rows).map(|r| r.join(" | ")));
    if data.len() > max_rows {
        lines.push(format!("... [{} more rows]", data.len() - max_rows));
    }
    lines.push(format!("\n({} rows, {} columns)", data.len(), header.len()));
    lines.join("\n")
}

/// Parse content already in memory. `suffix` (`.csv`, `.json`, ...) says how to read it.
pub fn parse_text(content: &str, suffix: &str) -> Result<String> {
    let suffix = suffix.to_lowercase();
    let pretty = Dumps { ensure_ascii: false, ..Default::default() };
    Ok(match suffix.as_str() {
        ".html" | ".htm" | ".xml" => html_to_text(content),
        ".csv" => csv_to_text(content, ',', 500),
        ".tsv" => csv_to_text(content, '\t', 500),
        ".json" => {
            let value: Value = serde_json::from_str(content).map_err(|e| error(format!("invalid JSON: {e}")))?;
            indent_json(&value, 0)
        }
        ".jsonl" | ".ndjson" => content
            .lines()
            .enumerate()
            .filter(|(_, line)| !line.trim().is_empty())
            .map(|(i, line)| serde_json::from_str::<Value>(line).map_or_else(|_| format!("[line {}: not valid JSON]", i + 1), |v| dumps_with(&v, pretty)))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => content.to_string(),
    })
}

/// `json.dumps(value, indent=2, ensure_ascii=False)`.
fn indent_json(value: &Value, depth: usize) -> String {
    let pad = "  ".repeat(depth + 1);
    let end = "  ".repeat(depth);
    match value {
        Value::Array(items) if !items.is_empty() => {
            let inner: Vec<String> = items.iter().map(|v| format!("{pad}{}", indent_json(v, depth + 1))).collect();
            format!("[\n{}\n{end}]", inner.join(",\n"))
        }
        Value::Object(map) if !map.is_empty() => {
            let inner: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{pad}{}: {}", dumps_with(&json!(k), Dumps { ensure_ascii: false, ..Default::default() }), indent_json(v, depth + 1)))
                .collect();
            format!("{{\n{}\n{end}}}", inner.join(",\n"))
        }
        other => dumps_with(other, Dumps { ensure_ascii: false, ..Default::default() }),
    }
}

#[cfg(feature = "pdf")]
fn read_pdf(bytes: Vec<u8>, max_pages: usize) -> Result<(String, Map<String, Value>)> {
    use pdf_extract::{Document, PlainTextOutput, output_doc_page};
    // The parser can panic on a damaged file: a panic is a file that could not be read.
    let outcome = std::panic::catch_unwind(move || -> std::result::Result<(usize, Vec<String>), String> {
        let mut doc = Document::load_mem(&bytes).map_err(|e| e.to_string())?;
        if doc.is_encrypted() {
            doc.decrypt("").map_err(|e| e.to_string())?;
        }
        let numbers: Vec<u32> = doc.get_pages().keys().copied().collect();
        let mut pages = Vec::new();
        for number in numbers.iter().take(max_pages) {
            let mut text = String::new();
            {
                let mut output = PlainTextOutput::new(&mut text);
                // A page that cannot be read reads as empty, as pypdf's does.
                let _ = output_doc_page(&doc, &mut output, *number);
            }
            pages.push(text);
        }
        Ok((numbers.len(), pages))
    });
    let (total, pages) = match outcome {
        Ok(Ok(found)) => found,
        Ok(Err(e)) => return Err(error(format!("parse_document failed: {e}"))),
        Err(_) => return Err(error("parse_document failed: the PDF could not be read")),
    };
    let mut text = pages.iter().enumerate().map(|(i, p)| format!("--- page {} ---\n{}", i + 1, p.trim())).collect::<Vec<_>>().join("\n\n");
    if text.trim().is_empty() {
        text = "[no extractable text \u{2014} this looks like a scanned PDF. Convert the pages to images and use OCR.]".into();
    }
    let mut meta = Map::new();
    meta.insert("pages".into(), json!(total));
    meta.insert("pages_read".into(), json!(pages.len()));
    Ok((text, meta))
}

#[cfg(not(feature = "pdf"))]
fn read_pdf(_bytes: Vec<u8>, _max_pages: usize) -> Result<(String, Map<String, Value>)> {
    Err(error("reading PDFs needs the `pdf` feature \u{2014} `agent-harness-adk = { features = [\"pdf\"] }`"))
}

/// The text of a `w:p` or a table cell: runs, tabs and breaks, as python-docx reads them.
fn docx_text(node: roxmltree::Node) -> String {
    let mut out = String::new();
    for d in node.descendants() {
        if d.tag_name().namespace() != Some(W) {
            continue;
        }
        match d.tag_name().name() {
            "t" => out.push_str(d.text().unwrap_or("")),
            "tab" => out.push('\t'),
            "br" | "cr" => out.push('\n'),
            _ => {}
        }
    }
    out
}

const W: &str = "http://schemas.openxmlformats.org/wordprocessingml/2006/main";

fn read_docx(bytes: &[u8]) -> Result<(String, Map<String, Value>)> {
    let failed = |e: String| error(format!("parse_document failed: {e}"));
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes)).map_err(|e| failed(e.to_string()))?;
    let mut xml = String::new();
    archive.by_name("word/document.xml").map_err(|e| failed(e.to_string()))?.read_to_string(&mut xml).map_err(|e| failed(e.to_string()))?;
    let doc = roxmltree::Document::parse(&xml).map_err(|e| failed(e.to_string()))?;
    let is = |n: &roxmltree::Node, name: &str| n.is_element() && n.tag_name().namespace() == Some(W) && n.tag_name().name() == name;
    let Some(body) = doc.root_element().children().find(|n| is(n, "body")) else {
        return Err(failed("no document body".into()));
    };
    let paragraphs: Vec<String> = body.children().filter(|n| is(n, "p")).map(docx_text).collect();
    let tables: Vec<roxmltree::Node> = body.children().filter(|n| is(n, "tbl")).collect();
    let mut parts: Vec<String> = paragraphs.iter().filter(|p| !p.trim().is_empty()).cloned().collect();
    for table in &tables {
        for row in table.children().filter(|n| is(n, "tr")) {
            let cells: Vec<String> = row
                .children()
                .filter(|n| is(n, "tc"))
                .map(|cell| cell.children().filter(|n| is(n, "p")).map(docx_text).collect::<Vec<_>>().join("\n").trim().to_string())
                .collect();
            parts.push(cells.join(" | "));
        }
    }
    let mut meta = Map::new();
    meta.insert("paragraphs".into(), json!(paragraphs.len()));
    meta.insert("tables".into(), json!(tables.len()));
    Ok((parts.join("\n"), meta))
}

async fn read_image(path: &Path, language: &str) -> Result<(String, Map<String, Value>)> {
    if !on_path("tesseract") {
        return Err(error("OCR needs tesseract \u{2014} install it and put the `tesseract` program on PATH"));
    }
    let out = tokio::process::Command::new("tesseract")
        .arg(path)
        .arg("stdout")
        .arg("-l")
        .arg(language)
        .output()
        .await
        .map_err(|e| error(format!("parse_document failed: could not start tesseract: {e}")))?;
    if !out.status.success() {
        return Err(error(format!("parse_document failed: {}", String::from_utf8_lossy(&out.stderr).trim())));
    }
    let mut meta = Map::new();
    if let Ok(size) = imagesize::size(path) {
        meta.insert("size".into(), json!(format!("{}x{}", size.width, size.height)));
    }
    meta.insert("ocr".into(), json!(true));
    Ok((String::from_utf8_lossy(&out.stdout).trim().to_string(), meta))
}

/// How [`parse_document`] reads a file.
#[derive(Debug, Clone)]
pub struct ParseOptions {
    pub max_chars: usize,
    /// For PDFs, how many pages to read.
    pub max_pages: usize,
    /// The OCR language (tesseract's code).
    pub language: String,
}

impl Default for ParseOptions {
    fn default() -> Self {
        Self { max_chars: 200_000, max_pages: 50, language: "eng".into() }
    }
}

/// Read a document and return its text plus what was learned about it:
/// `{"text", "path", "format", "bytes", ..., "characters", "truncated"?}`.
pub async fn parse_document(path: impl AsRef<Path>, options: &ParseOptions) -> Result<Value> {
    let file = path.as_ref();
    if !file.is_file() {
        return Err(error(format!("no such file: {}", file.display())));
    }
    let suffix = file.extension().map(|e| format!(".{}", e.to_string_lossy().to_lowercase())).filter(|s| s.len() > 1).unwrap_or_default();
    let size = std::fs::metadata(file).map(|m| m.len()).unwrap_or(0);
    let mut meta = Map::new();
    meta.insert("path".into(), json!(file.display().to_string()));
    meta.insert("format".into(), json!(if suffix.is_empty() { "unknown" } else { &suffix }));
    meta.insert("bytes".into(), json!(size));
    let read = |file: &Path| std::fs::read(file).map_err(|e| error(format!("parse_document failed: {e}")));

    let (text, extra) = if suffix == ".pdf" {
        let bytes = read(file)?;
        let max_pages = options.max_pages;
        tokio::task::spawn_blocking(move || read_pdf(bytes, max_pages)).await.map_err(|e| error(format!("parse_document failed: {e}")))??
    } else if suffix == ".docx" {
        read_docx(&read(file)?)?
    } else if IMAGES.contains(&suffix.as_str()) {
        read_image(file, &options.language).await?
    } else if suffix.is_empty() || NATIVE.contains(&suffix.as_str()) {
        let raw = String::from_utf8_lossy(&read(file)?).replace("\r\n", "\n").replace('\r', "\n");
        (parse_text(&raw, if suffix.is_empty() { ".txt" } else { &suffix })?, Map::new())
    } else {
        let mut native: Vec<&str> = NATIVE.to_vec();
        native.sort_unstable();
        let formats = supported_formats();
        let missing: Vec<&str> = formats["needs_install"].as_array().map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
        let needs = if missing.is_empty() { String::new() } else { format!("Needs an install: {}", missing.join("; ")) };
        let what = if suffix.is_empty() { "this file" } else { &suffix };
        return Err(error(format!("cannot read {what}. Native: {}... {needs}", native[..8].join(", "))));
    };

    meta.extend(extra);
    let characters = text.chars().count();
    meta.insert("characters".into(), json!(characters));
    let text = match text.char_indices().nth(options.max_chars) {
        Some((cut, _)) => {
            meta.insert("truncated".into(), json!(true));
            format!("{}\n... [truncated at {} characters]", &text[..cut], options.max_chars)
        }
        None => text,
    };
    let mut out = Map::new();
    out.insert("text".into(), json!(text));
    out.extend(meta);
    Ok(Value::Object(out))
}

fn fifty() -> usize {
    50
}

#[derive(Deserialize, JsonSchema)]
struct DocumentArgs {
    /// the file to read.
    path: String,
    /// for PDFs, how many pages to read.
    #[serde(default = "fifty")]
    max_pages: usize,
}

/// A document-parsing tool, named `parse_document`. With a workspace, paths
/// resolve inside it (a file in a remote sandbox is brought down first);
/// without one, a path is read where it lies.
pub fn make_document_tool(workspace: Option<Arc<dyn Workspace>>, name: &str) -> Tool {
    Tool::new(name, move |args: DocumentArgs, _ctx: ToolContext| {
        let workspace = workspace.clone();
        async move {
            let target = match &workspace {
                Some(ws) => ws.materialize(&args.path).await?,
                None => PathBuf::from(&args.path),
            };
            parse_document(target, &ParseOptions { max_pages: args.max_pages, ..Default::default() }).await
        }
    })
    .describe(
        "Read a document and return its text.\n\nHandles text, Markdown, CSV, TSV, JSON, JSONL, HTML and XML directly;\nPDF, DOCX and scanned images if the optional libraries are installed.",
    )
    .tags(["builtin", "documents"])
    .cacheable(true)
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;

    use super::*;
    use crate::runtime::workspace::{LocalWorkspace, WorkspaceOptions};

    fn dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ah-docs-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    async fn parse(path: &Path) -> Result<Value> {
        parse_document(path, &ParseOptions::default()).await
    }

    #[tokio::test]
    async fn plain_text_markdown_csv_and_json() {
        let d = dir("text");
        std::fs::write(d.join("notes.md"), "# Title\r\n\r\nSome body text.").unwrap();
        let parsed = parse(&d.join("notes.md")).await.unwrap();
        assert_eq!(parsed["text"], "# Title\n\nSome body text.");
        assert_eq!(parsed["format"], ".md");
        assert_eq!(parsed["characters"], 24);
        assert_eq!(parsed.as_object().unwrap().keys().collect::<Vec<_>>(), ["text", "path", "format", "bytes", "characters"]);

        std::fs::write(d.join("data.csv"), "name,amount\nada,10\n\"grace, h\",20\n").unwrap();
        let text = parse(&d.join("data.csv")).await.unwrap()["text"].as_str().unwrap().to_string();
        assert_eq!(text, "name | amount\nada | 10\ngrace, h | 20\n\n(2 rows, 2 columns)");

        std::fs::write(d.join("d.json"), r#"{"b":2,"a":[1,{"c":"é"}],"e":{}}"#).unwrap();
        let text = parse(&d.join("d.json")).await.unwrap()["text"].as_str().unwrap().to_string();
        assert_eq!(text, "{\n  \"b\": 2,\n  \"a\": [\n    1,\n    {\n      \"c\": \"é\"\n    }\n  ],\n  \"e\": {}\n}");
        std::fs::write(d.join("bad.json"), "{not json").unwrap();
        assert!(parse(&d.join("bad.json")).await.unwrap_err().to_string().starts_with("invalid JSON"));

        std::fs::write(d.join("page.html"), "<html><body><h1>Title</h1><script>evil()</script><p>Body&nbsp;text</p></body></html>").unwrap();
        let text = parse(&d.join("page.html")).await.unwrap()["text"].as_str().unwrap().to_string();
        assert!(text.contains("Title") && text.contains("Body text") && !text.contains("evil()") && !text.contains("<p>"), "{text}");
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn parse_text_handles_tsv_big_csv_and_jsonl() {
        let body = format!("a,b\n{}", (0..2000).map(|i| format!("{i},{i}")).collect::<Vec<_>>().join("\n"));
        let text = parse_text(&body, ".csv").unwrap();
        assert!(text.contains("... [1500 more rows]") && text.ends_with("(2000 rows, 2 columns)"));
        assert!(parse_text("x\ty\na\tb\n", ".tsv").unwrap().contains("a | b"));
        assert_eq!(parse_text("{\"a\":1}\nnot json\n\n{\"b\":2}", ".jsonl").unwrap(), "{\"a\": 1}\n[line 2: not valid JSON]\n{\"b\": 2}");
        assert_eq!(parse_text("plain", ".TXT").unwrap(), "plain");
    }

    #[tokio::test]
    async fn missing_unknown_and_truncated_files_are_clear() {
        let d = dir("odd");
        let e = parse(&d.join("nope.txt")).await.unwrap_err().to_string();
        assert!(e.starts_with("no such file: "), "{e}");
        std::fs::write(d.join("thing.xyz"), "x").unwrap();
        let e = parse(&d.join("thing.xyz")).await.unwrap_err().to_string();
        assert!(e.starts_with("cannot read .xyz. Native: .cfg, .csv, .htm, .html, .ini, .js, .json, .jsonl... "), "{e}");
        std::fs::write(d.join("big.txt"), "x".repeat(5000)).unwrap();
        let parsed = parse_document(d.join("big.txt"), &ParseOptions { max_chars: 1000, ..Default::default() }).await.unwrap();
        assert_eq!(parsed["truncated"], true);
        assert_eq!(parsed["characters"], 5000);
        assert!(parsed["text"].as_str().unwrap().ends_with("\n... [truncated at 1000 characters]"));
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn the_format_list_separates_native_from_needs_install() {
        let formats = supported_formats();
        let native: Vec<&str> = formats["native"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
        assert!(native.contains(&".csv") && native.contains(&".json"));
        let available: Vec<&str> = formats["available"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
        assert!(available.contains(&".docx"));
        assert_eq!(available.contains(&".pdf"), cfg!(feature = "pdf"));
    }

    /// A one-page PDF saying `text`, with a correct cross-reference table.
    fn tiny_pdf(text: &str) -> Vec<u8> {
        let stream = format!("BT /F1 24 Tf 72 720 Td ({text}) Tj ET");
        let objects = [
            "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>".to_string(),
            format!("<< /Length {} >>\nstream\n{stream}\nendstream", stream.len()),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>".to_string(),
        ];
        let mut out = b"%PDF-1.4\n".to_vec();
        let mut offsets = Vec::new();
        for (i, body) in objects.iter().enumerate() {
            offsets.push(out.len());
            out.extend_from_slice(format!("{} 0 obj\n{body}\nendobj\n", i + 1).as_bytes());
        }
        let xref = out.len();
        out.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).as_bytes());
        for o in offsets {
            out.extend_from_slice(format!("{o:010} 00000 n \n").as_bytes());
        }
        out.extend_from_slice(format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n", objects.len() + 1).as_bytes());
        out
    }

    #[tokio::test]
    async fn pdfs_are_read_page_by_page() {
        let d = dir("pdf");
        std::fs::write(d.join("doc.pdf"), tiny_pdf("Hello from a PDF")).unwrap();
        std::fs::write(d.join("broken.pdf"), b"%PDF-1.4 not really a pdf").unwrap();
        if cfg!(feature = "pdf") {
            let parsed = parse(&d.join("doc.pdf")).await.unwrap();
            let text = parsed["text"].as_str().unwrap();
            assert!(text.starts_with("--- page 1 ---\n") && text.contains("Hello from a PDF"), "{text}");
            assert_eq!((parsed["pages"].clone(), parsed["pages_read"].clone()), (json!(1), json!(1)));
            let e = parse(&d.join("broken.pdf")).await.unwrap_err().to_string();
            assert!(e.starts_with("parse_document failed"), "{e}");
        } else {
            let e = parse(&d.join("doc.pdf")).await.unwrap_err().to_string();
            assert!(e.contains("`pdf` feature"), "{e}");
        }
        let _ = std::fs::remove_dir_all(d);
    }

    fn tiny_docx() -> Vec<u8> {
        let xml = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:r><w:t>First </w:t></w:r><w:r><w:t>paragraph</w:t><w:tab/><w:t>tabbed</w:t></w:r></w:p>
<w:p></w:p>
<w:tbl><w:tr><w:tc><w:p><w:r><w:t> a </w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>b</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
<w:p><w:r><w:t>Last</w:t></w:r></w:p>
</w:body></w:document>"#;
        let mut buffer = std::io::Cursor::new(Vec::new());
        {
            let mut zip = zip::ZipWriter::new(&mut buffer);
            zip.start_file("word/document.xml", zip::write::SimpleFileOptions::default()).unwrap();
            zip.write_all(xml.as_bytes()).unwrap();
            zip.finish().unwrap();
        }
        buffer.into_inner()
    }

    #[tokio::test]
    async fn docx_paragraphs_then_tables() {
        let d = dir("docx");
        std::fs::write(d.join("memo.docx"), tiny_docx()).unwrap();
        let parsed = parse(&d.join("memo.docx")).await.unwrap();
        assert_eq!(parsed["text"], "First paragraph\ttabbed\nLast\na | b");
        assert_eq!((parsed["paragraphs"].clone(), parsed["tables"].clone()), (json!(3), json!(1)));
        let _ = std::fs::remove_dir_all(d);
    }

    #[tokio::test]
    async fn the_document_tool_resolves_inside_the_workspace() {
        let ws: Arc<dyn Workspace> = Arc::new(LocalWorkspace::new(dir("ws"), WorkspaceOptions { ephemeral: true, ..Default::default() }).unwrap());
        ws.write("report.md", b"the contents", false).await.unwrap();
        let entry = make_document_tool(Some(ws.clone()), "parse_document");
        assert!(entry.cacheable);
        assert_eq!(entry.parameters["properties"]["max_pages"]["default"], 50);
        let out = entry.invoke(json!({"path": "report.md"}).as_object().cloned().unwrap(), ToolContext::default()).await.unwrap();
        assert!(out["text"].as_str().unwrap().contains("the contents"));
        let e = entry.invoke(json!({"path": "../../etc/passwd"}).as_object().cloned().unwrap(), ToolContext::default()).await.unwrap_err();
        assert!(e.to_string().contains("outside the workspace"), "{e}");
        ws.close().await;
    }
}
