//! Native tools that ship with the harness.

pub mod basics;
pub mod charts;
pub mod compute;
pub mod documents;
pub mod images;
pub(crate) mod net;
pub mod openapi;
pub mod search;
pub mod web;

pub use basics::{basic_tools, calculate, make_corpus_search, make_corpus_search_with, now, wait};
pub use charts::{ChartOptions, bar_chart, line_chart, make_chart_tool, make_report_tool, markdown_table, render_report};
pub use compute::{make_python_tool, make_python_tool_with};
pub use documents::{ParseOptions, make_document_tool, parse_document, parse_text, supported_formats};
pub use search::{
    EngineRequest, EngineSettings, EngineStats, SearchEngine, SearchFailure, SearchQuery, SearchResult, WebSearch, WebSearchBuilder, engine_named,
    make_search_tool, search_engines, web_search,
};
pub use web::{FetchOptions, http_fetch, make_fetch_tool, make_http_tool};

pub use images::{
    FakeImages, GeminiImages, GenerateOptions, GeneratedImage, ImageEngine, ImageFailure, ImageGenerator, ImageJob, ImageOptions, ImageProgress, ImageRequest,
    ImageToolOptions, OpenAIImages, Reference, ReplicateImages, image_engines, image_tools,
};
pub use openapi::{OpenAPIOptions, OpenAPISpec, OpenAPIToolkit, Secret, openapi_tools, toolkits_from_config};
