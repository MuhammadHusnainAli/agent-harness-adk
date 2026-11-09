//! The browser, as the tools an agent calls.

use std::future::Future;
use std::sync::Arc;

use schemars::JsonSchema;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use crate::browser::browser::Browser;
use crate::errors::{Error, Result};
use crate::tools::{NoArgs, Permission, Tool};
use crate::types::ContentBlock;

/// How [`browser_tools`] builds the tools.
///
/// `prefix` is what the tool names start with. `screenshots` offers
/// `browser_screenshot`, for a model that can see; `vision` sends a screenshot
/// back with every action, not only on request. `Permission::Ask` has every
/// action approved first.
#[derive(Debug, Clone)]
pub struct BrowserToolsOptions {
    pub prefix: String,
    pub screenshots: bool,
    pub vision: bool,
    pub permission: Permission,
}

impl Default for BrowserToolsOptions {
    fn default() -> Self {
        Self { prefix: "browser_".into(), screenshots: true, vision: false, permission: Permission::Allow }
    }
}

/// Open a web page, and read what is on it.
#[derive(Deserialize, JsonSchema)]
struct Navigate {
    /// the address to open.
    url: String,
}

/// Click an element of the page.
#[derive(Deserialize, JsonSchema)]
struct Click {
    /// the element's number in the page listing, e.g. 12 for [12].
    #[serde(rename = "ref")]
    #[schemars(rename = "ref")]
    reference: i64,
}

/// Type into a field, replacing what was in it.
#[derive(Deserialize, JsonSchema)]
struct TypeInto {
    /// the field's number in the page listing.
    #[serde(rename = "ref")]
    #[schemars(rename = "ref")]
    reference: i64,
    /// what to type.
    text: String,
    /// press Enter afterwards, to send a form or run a search.
    #[serde(default)]
    submit: bool,
}

/// Choose an option from a drop-down list.
#[derive(Deserialize, JsonSchema)]
struct Select {
    /// the list's number in the page listing.
    #[serde(rename = "ref")]
    #[schemars(rename = "ref")]
    reference: i64,
    /// the text of the option to choose.
    option: String,
}

/// Press a key: Enter, Tab, Escape, ArrowDown, PageDown, or a combination like ctrl+a.
#[derive(Deserialize, JsonSchema)]
struct Press {
    /// the key, or keys joined with +.
    key: String,
}

fn down() -> String {
    "down".into()
}
fn one() -> f64 {
    1.0
}
fn eight_thousand() -> i64 {
    8000
}
fn two() -> f64 {
    2.0
}
fn list() -> String {
    "list".into()
}

/// Scroll the page.
#[derive(Deserialize, JsonSchema)]
struct Scroll {
    /// up, down, left, right, top or bottom.
    #[serde(default = "down")]
    direction: String,
    /// how many screens to move.
    #[serde(default = "one")]
    amount: f64,
}

/// Read the text of the page — all of it, where the listing shows only the start.
#[derive(Deserialize, JsonSchema)]
struct Read {
    /// the character to start from, to read on.
    #[serde(default)]
    offset: i64,
    /// how much to return.
    #[serde(default = "eight_thousand")]
    max_chars: i64,
}

/// Wait for the page: a number of seconds, or until some text appears.
#[derive(Deserialize, JsonSchema)]
struct Wait {
    /// how long to wait, at most 30.
    #[serde(default = "two")]
    seconds: f64,
    /// stop waiting as soon as the page shows this.
    #[serde(default)]
    text: String,
}

/// List the open tabs, switch to one, open a new one, or close one.
#[derive(Deserialize, JsonSchema)]
struct Tabs {
    /// list, switch, new or close.
    #[serde(default = "list")]
    action: String,
    /// the tab's number, for switch and close.
    #[serde(default)]
    index: i64,
    /// the page to open, for new.
    #[serde(default)]
    url: String,
}

struct Maker<'a> {
    browser: &'a Browser,
    options: &'a BrowserToolsOptions,
}

impl Maker<'_> {
    fn made<A, F, Fut>(&self, name: &str, description: &str, f: F) -> Tool
    where
        A: DeserializeOwned + JsonSchema + Send + 'static,
        F: Fn(Browser, A) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value>> + Send + 'static,
    {
        let full = format!("{}{name}", self.options.prefix);
        let vision = self.options.vision && !matches!(name, "read" | "tabs");
        let (browser, f, tool_name) = (self.browser.clone(), Arc::new(f), full.clone());
        Tool::new(full, move |args: A, _ctx| {
            let (b, f, tool_name) = (browser.clone(), f.clone(), tool_name.clone());
            async move {
                let said = f(b.clone(), args).await.map_err(|e| match e {
                    Error::Cdp { message, .. } => Error::tool(format!("the browser could not do that: {message}"), &tool_name),
                    other => other,
                })?;
                if vision && said.is_string() {
                    return Ok(match b.screenshot().await {
                        Ok(shot) => json!([said, ContentBlock::Image(shot)]),
                        Err(_) => said,
                    });
                }
                Ok(said)
            }
        })
        .describe(description)
        .tags(["builtin", "browser", "web"])
        .permission(self.options.permission)
    }
}

/// The tools that drive a browser: navigate, snapshot, click, type, select,
/// press, scroll, back, read, wait, tabs — and screenshot.
pub fn browser_tools(browser: &Browser, options: &BrowserToolsOptions) -> Vec<Tool> {
    let m = Maker { browser, options };
    let prefix = options.prefix.clone();
    let mut out = vec![
        m.made("navigate", "Open a web page, and read what is on it.", |b, a: Navigate| async move { b.goto(&a.url).await.map(Value::from) }),
        m.made("snapshot", "Read the page again: its numbered elements and its text.", |b, _: NoArgs| async move { b.snapshot().await.map(Value::from) }),
        m.made("click", "Click an element of the page.", |b, a: Click| async move { b.click(a.reference).await.map(Value::from) }),
        m.made("type", "Type into a field, replacing what was in it.", |b, a: TypeInto| async move { b.type_into(a.reference, &a.text, a.submit, true).await.map(Value::from) }),
        m.made("select", "Choose an option from a drop-down list.", |b, a: Select| async move { b.select(a.reference, &a.option).await.map(Value::from) }),
        m.made("press", "Press a key: Enter, Tab, Escape, ArrowDown, PageDown, or a combination like ctrl+a.", |b, a: Press| async move { b.press(&a.key).await.map(Value::from) }),
        m.made("scroll", "Scroll the page.", |b, a: Scroll| async move { b.scroll(&a.direction, a.amount).await.map(Value::from) }),
        m.made("back", "Go back to the previous page.", |b, _: NoArgs| async move { b.back().await.map(Value::from) }),
        m.made("read", "Read the text of the page — all of it, where the listing shows only the start.", |b, a: Read| async move { b.read(a.offset, a.max_chars).await.map(Value::from) }),
        m.made("wait", "Wait for the page: a number of seconds, or until some text appears.", |b, a: Wait| async move { b.wait(a.seconds, &a.text).await.map(Value::from) }),
        m.made("tabs", "List the open tabs, switch to one, open a new one, or close one.", move |b, a: Tabs| {
            let tool = format!("{prefix}tabs");
            async move {
                let said = match a.action.trim().to_lowercase().as_str() {
                    "" | "list" => b.tabs().await?,
                    "switch" => b.switch_tab(a.index).await?,
                    "new" => b.new_tab(&a.url).await?,
                    "close" => b.close_tab(Some(a.index)).await?,
                    _ => return Err(Error::tool("action is one of list, switch, new, close", &tool)),
                };
                Ok(Value::from(said))
            }
        }),
    ];
    if options.screenshots || options.vision {
        out.push(m.made("screenshot", "Take a picture of the page as it is shown, to look at it.", |b, _: NoArgs| async move {
            let shot = b.screenshot().await?;
            Ok(json!(["A screenshot of the page.", ContentBlock::Image(shot)]))
        }));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tools_and_their_shapes() {
        let b = Browser::default();
        let tools = browser_tools(&b, &BrowserToolsOptions::default());
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "browser_navigate",
                "browser_snapshot",
                "browser_click",
                "browser_type",
                "browser_select",
                "browser_press",
                "browser_scroll",
                "browser_back",
                "browser_read",
                "browser_wait",
                "browser_tabs",
                "browser_screenshot"
            ]
        );
        let click = &tools[2];
        assert_eq!(click.parameters["required"], json!(["ref"]));
        assert!(click.parameters["properties"]["ref"]["description"].as_str().unwrap().contains("e.g. 12 for [12]"));
        assert!(click.tags.contains("browser"));
        let quiet = browser_tools(&b, &BrowserToolsOptions { prefix: "web_".into(), screenshots: false, permission: Permission::Ask, ..Default::default() });
        assert_eq!(quiet.len(), 11);
        assert_eq!(quiet[0].name, "web_navigate");
        assert_eq!(quiet[0].permission, Permission::Ask);
    }
}
