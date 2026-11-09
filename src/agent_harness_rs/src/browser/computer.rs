//! Computer use: a screen an agent looks at, and a mouse and keyboard it works.
//!
//! ```no_run
//! # async fn demo() -> agent_harness::Result<()> {
//! use std::sync::Arc;
//! use agent_harness::Agent;
//! use agent_harness::browser::{Browser, ComputerToolOptions, computer_tool};
//! let browser = Browser::default();
//! let screen = Arc::new(browser.computer());
//! let agent = Agent::builder("operator").tool(computer_tool(screen, &ComputerToolOptions::default())).build()?;
//! agent.run("Open example.com and tell me what the heading says.").await?;
//! # Ok(()) }
//! ```
//!
//! The agent calls one tool, `computer`, with an action — `screenshot`,
//! `click`, `type`, `key`, `scroll`, `drag` — and is shown the screen after
//! each one. It needs a model that can see. Any such model will do: the tool is
//! an ordinary one, not a vendor's.
//!
//! Two screens ship. [`Browser::computer`] is a browser page: the place for
//! anything on the web that the numbered listing of `browser.tools()` cannot
//! reach — a canvas, a map, a game. [`DesktopComputer`] is a Linux desktop
//! driven with `xdotool`, on this machine or — better — inside a sandbox.
//! Any other screen is an implementation of [`Computer`].

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine;
use serde_json::{Map, Value, json};

use crate::browser::browser::Browser;
use crate::errors::{Error, Result};
use crate::sandboxes::Sandbox;
use crate::toolkits::search::fmt_g;
use crate::tools::{Permission, Tool};
use crate::types::{ContentBlock, MediaBlock};

pub const ACTIONS: [&str; 11] = ["screenshot", "click", "double_click", "right_click", "middle_click", "move", "drag", "type", "key", "scroll", "wait"];

/// A screen with a pointer and a keyboard.
///
/// Coordinates are pixels from the top-left corner of the screenshot.
/// Implement it for a screen that is not here — a VNC session, a phone, a
/// remote desktop — and [`computer_tool`] gives an agent the use of it.
#[async_trait]
pub trait Computer: Send + Sync {
    /// The screen, in pixels. Known once `start` has run.
    fn width(&self) -> u32;
    fn height(&self) -> u32;
    /// What the model is told this is.
    fn kind(&self) -> &str {
        "computer"
    }
    /// Does `open(url)` do something here?
    fn can_open(&self) -> bool {
        false
    }
    /// Is this the desktop of the machine the harness runs on?
    fn local(&self) -> bool {
        false
    }
    /// Get ready: connect, measure the screen. Called before the first action.
    async fn start(&self) -> Result<()> {
        Ok(())
    }
    /// Let go of the screen.
    async fn close(&self) {}
    /// The screen as it is now.
    async fn screenshot(&self) -> Result<MediaBlock>;
    async fn click(&self, x: i64, y: i64, button: &str, count: i64) -> Result<()>;
    async fn move_to(&self, x: i64, y: i64) -> Result<()>;
    async fn drag(&self, x: i64, y: i64, to_x: i64, to_y: i64) -> Result<()>;
    async fn type_text(&self, text: &str) -> Result<()>;
    /// Press a key or a combination: `Enter`, `ctrl+a`.
    async fn key(&self, keys: &str) -> Result<()>;
    /// Turn the wheel `amount` notches at a point.
    async fn scroll(&self, x: i64, y: i64, direction: &str, amount: i64) -> Result<()>;
    async fn open(&self, _url: &str) -> Result<()> {
        Err(Error::tool(format!("this {} cannot be told an address — use its own address bar", self.kind()), "computer"))
    }
}

/// A browser page as the screen. The mouse and keys are the real thing: the
/// page cannot tell them from a person's.
#[derive(Debug, Clone)]
pub struct BrowserComputer {
    pub browser: Browser,
}

impl BrowserComputer {
    pub fn new(browser: Browser) -> Self {
        Self { browser }
    }
}

impl BrowserComputer {
    async fn locked<F, Fut>(&self, action: F) -> Result<()>
    where
        F: FnOnce(Browser) -> Fut,
        Fut: std::future::Future<Output = Result<()>>,
    {
        let b = &self.browser;
        let _held = b.locked().await;
        b.guarded().await?;
        action(b.clone()).await
    }
}

#[async_trait]
impl Computer for BrowserComputer {
    fn width(&self) -> u32 {
        self.browser.viewport().0
    }
    fn height(&self) -> u32 {
        self.browser.viewport().1
    }
    fn kind(&self) -> &str {
        "browser"
    }
    fn can_open(&self) -> bool {
        true
    }
    async fn start(&self) -> Result<()> {
        self.browser.start().await.map(drop)
    }
    async fn close(&self) {
        self.browser.close().await;
    }
    async fn screenshot(&self) -> Result<MediaBlock> {
        self.browser.screenshot().await
    }
    async fn click(&self, x: i64, y: i64, button: &str, count: i64) -> Result<()> {
        let button = button.to_string();
        self.locked(|b| async move { b.click_at(x as f64, y as f64, &button, count).await }).await
    }
    async fn move_to(&self, x: i64, y: i64) -> Result<()> {
        self.locked(|b| async move { b.mouse("mouseMoved", x as f64, y as f64, "none", 0, 0, None).await }).await
    }
    async fn drag(&self, x: i64, y: i64, to_x: i64, to_y: i64) -> Result<()> {
        self.locked(|b| async move {
            let (x, y, tx, ty) = (x as f64, y as f64, to_x as f64, to_y as f64);
            b.mouse("mouseMoved", x, y, "none", 0, 0, None).await?;
            b.mouse("mousePressed", x, y, "left", 1, 1, None).await?;
            // In steps: a page that follows the pointer needs to see it travel.
            for step in 1..=10 {
                let f = f64::from(step) / 10.0;
                b.mouse("mouseMoved", x + (tx - x) * f, y + (ty - y) * f, "left", 0, 1, None).await?;
            }
            b.mouse("mouseReleased", tx, ty, "left", 1, 0, None).await
        })
        .await
    }
    async fn type_text(&self, text: &str) -> Result<()> {
        let text = text.to_string();
        self.locked(|b| async move { b.type_text(&text).await }).await
    }
    async fn key(&self, keys: &str) -> Result<()> {
        let keys = keys.to_string();
        self.locked(|b| async move { b.key(&keys).await.map(drop) }).await
    }
    async fn scroll(&self, x: i64, y: i64, direction: &str, amount: i64) -> Result<()> {
        let step = 100.0 * amount as f64;
        let (dx, dy) = match direction {
            "up" => (0.0, -step),
            "left" => (-step, 0.0),
            "right" => (step, 0.0),
            _ => (0.0, step),
        };
        self.locked(|b| async move { b.wheel(x as f64, y as f64, dx, dy).await }).await
    }
    async fn open(&self, url: &str) -> Result<()> {
        self.browser.goto(url).await.map(drop)
    }
}

/// The names a model uses, as xdotool knows them.
fn xkey(name: &str) -> Option<&'static str> {
    Some(match name {
        "enter" | "return" => "Return",
        "esc" | "escape" => "Escape",
        "backspace" => "BackSpace",
        "delete" | "del" => "Delete",
        "tab" => "Tab",
        "space" => "space",
        "pageup" | "pgup" => "Prior",
        "pagedown" | "pgdn" => "Next",
        "arrowup" | "up" => "Up",
        "arrowdown" | "down" => "Down",
        "arrowleft" | "left" => "Left",
        "arrowright" | "right" => "Right",
        "home" => "Home",
        "end" => "End",
        "insert" => "Insert",
        "ctrl" | "control" => "ctrl",
        "alt" | "option" => "alt",
        "shift" => "shift",
        "meta" | "cmd" | "command" | "super" | "win" => "super",
        _ => return None,
    })
}

const SHOOTERS: [(&str, &str); 4] = [
    ("scrot", "scrot -o -z {path}"),
    ("import", "import -window root {path}"),
    ("gnome-screenshot", "gnome-screenshot -f {path}"),
    ("xfce4-screenshooter", "xfce4-screenshooter -f -s {path}"),
];

fn quote(s: &str) -> String {
    if !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || "@%+=:,./-_".contains(c)) {
        return s.into();
    }
    format!("'{}'", s.replace('\'', "'\"'\"'"))
}

/// A Linux desktop (X11), driven with `xdotool`.
///
/// `sandbox` is the sandbox whose desktop this is — a container with a
/// display, `xdotool` and `scrot` in it. `None` drives *this* machine's
/// desktop, which is why the tool then asks before every action. `display` is
/// the X display, e.g. `":1"`; the default is the environment's.
///
/// Wayland does not let one program drive another, so this needs X11 — which
/// is what a virtual display (Xvfb) in a container is.
pub struct DesktopComputer {
    pub sandbox: Option<Arc<Sandbox>>,
    pub display: Option<String>,
    pub timeout: f64,
    shot: Mutex<String>,
    path: String,
    width: AtomicU32,
    height: AtomicU32,
}

impl std::fmt::Debug for DesktopComputer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<DesktopComputer {}>", if self.sandbox.is_some() { "in a sandbox" } else { "local" })
    }
}

impl DesktopComputer {
    pub fn new(sandbox: Option<Arc<Sandbox>>) -> Self {
        Self { sandbox, display: None, timeout: 20.0, shot: Mutex::default(), path: "/tmp/agent-harness-screen.png".into(), width: AtomicU32::new(0), height: AtomicU32::new(0) }
    }

    pub fn display(mut self, display: &str) -> Self {
        self.display = Some(display.into());
        self
    }

    pub fn timeout(mut self, seconds: f64) -> Self {
        self.timeout = seconds;
        self
    }

    async fn run(&self, command: &str) -> Result<(i32, String, String)> {
        let command = match &self.display {
            Some(d) => format!("DISPLAY={} {command}", quote(d)),
            None => command.to_string(),
        };
        if let Some(sandbox) = &self.sandbox {
            let r = sandbox.exec(&command, None, None, Some(self.timeout)).await?;
            return Ok((r.returncode, r.stdout, r.stderr));
        }
        match crate::sandboxes::base::run_process(vec!["sh".into(), "-c".into(), command], None, self.timeout).await? {
            Some((code, out, err)) => Ok((code, String::from_utf8_lossy(&out).into_owned(), String::from_utf8_lossy(&err).into_owned())),
            None => Ok((124, String::new(), format!("timed out after {}s", fmt_g(self.timeout)))),
        }
    }

    async fn x(&self, arguments: &str, what: &str) -> Result<String> {
        let (code, out, err) = self.run(&format!("xdotool {arguments}")).await?;
        if code != 0 {
            let said: String = if err.is_empty() { &out } else { &err }.trim().chars().take(300).collect();
            return Err(Error::tool(format!("could not {what}: {}", if said.is_empty() { code.to_string() } else { said }), "computer"));
        }
        Ok(out)
    }

    fn where_(&self) -> &str {
        if self.sandbox.is_some() { "the sandbox" } else { "this machine" }
    }
}

#[async_trait]
impl Computer for DesktopComputer {
    fn width(&self) -> u32 {
        self.width.load(Ordering::SeqCst)
    }
    fn height(&self) -> u32 {
        self.height.load(Ordering::SeqCst)
    }
    fn kind(&self) -> &str {
        "desktop"
    }
    fn local(&self) -> bool {
        self.sandbox.is_none()
    }

    async fn start(&self) -> Result<()> {
        if !self.shot.lock().unwrap_or_else(|p| p.into_inner()).is_empty() {
            return Ok(());
        }
        let has: Vec<String> = SHOOTERS.iter().map(|(name, _)| format!("command -v {name} >/dev/null 2>&1 && echo HAS-{name}")).collect();
        let (code, out, err) = self
            .run(&format!(
                "command -v xdotool >/dev/null 2>&1 || {{ echo NO-XDOTOOL; exit 0; }}; xdotool getdisplaygeometry || {{ echo NO-DISPLAY; exit 0; }}; {}; true",
                has.join("; ")
            ))
            .await?;
        let where_ = self.where_();
        if out.contains("NO-XDOTOOL") {
            return Err(Error::config(format!("xdotool is not installed on {where_} — a desktop is driven with it (apt install xdotool scrot)")));
        }
        if out.contains("NO-DISPLAY") || code != 0 {
            let said = if err.trim().is_empty() { out.trim() } else { err.trim() };
            let said = if said.is_empty() { String::new() } else { format!(" ({})", said.chars().take(200).collect::<String>()) };
            return Err(Error::config(format!("no X display can be reached on {where_}{said} — start one (Xvfb :1 -screen 0 1280x800x24) and pass display ':1'")));
        }
        let size: Vec<&str> = out.split('\n').next().unwrap_or("").split_whitespace().collect();
        match (size.first().and_then(|w| w.parse().ok()), size.get(1).and_then(|h| h.parse().ok())) {
            (Some(w), Some(h)) => {
                self.width.store(w, Ordering::SeqCst);
                self.height.store(h, Ordering::SeqCst);
            }
            _ => return Err(Error::config(format!("the size of the display could not be read: {}", crate::toolkits::openapi::py_repr(&out.trim().chars().take(200).collect::<String>())))),
        }
        for (name, command) in SHOOTERS {
            if out.contains(&format!("HAS-{name}")) {
                *self.shot.lock().unwrap_or_else(|p| p.into_inner()) = command.replace("{path}", &self.path);
                return Ok(());
            }
        }
        Err(Error::config(format!("nothing on {where_} can take a screenshot — install one of: {}", SHOOTERS.map(|(n, _)| n).join(", "))))
    }

    async fn screenshot(&self) -> Result<MediaBlock> {
        self.start().await?;
        let shot = self.shot.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let (_, out, err) = self.run(&format!("rm -f {p}; {shot} >/dev/null 2>&1; base64 < {p} 2>/dev/null", p = self.path)).await?;
        let data = base64::engine::general_purpose::STANDARD.decode(out.split_whitespace().collect::<String>()).unwrap_or_default();
        if !data.starts_with(b"\x89PNG") {
            let said = if err.trim().is_empty() { String::new() } else { format!(": {}", err.trim().chars().take(200).collect::<String>()) };
            return Err(Error::tool(format!("the screenshot could not be taken{said}"), "computer"));
        }
        let mut block = MediaBlock::from_bytes(&data, "image/png");
        block.name = "screenshot".into();
        Ok(block)
    }

    async fn click(&self, x: i64, y: i64, button: &str, count: i64) -> Result<()> {
        let number = match button {
            "middle" => 2,
            "right" => 3,
            _ => 1,
        };
        let repeat = if count > 1 { format!("--repeat {count} --delay 80 ") } else { String::new() };
        self.x(&format!("mousemove {x} {y} click {repeat}{number}"), "click").await.map(drop)
    }

    async fn move_to(&self, x: i64, y: i64) -> Result<()> {
        self.x(&format!("mousemove {x} {y}"), "move the pointer").await.map(drop)
    }

    async fn drag(&self, x: i64, y: i64, to_x: i64, to_y: i64) -> Result<()> {
        self.x(&format!("mousemove {x} {y} mousedown 1 sleep 0.1 mousemove {to_x} {to_y} sleep 0.1 mouseup 1"), "drag").await.map(drop)
    }

    async fn type_text(&self, text: &str) -> Result<()> {
        self.x(&format!("type --delay 12 -- {}", quote(text)), "type").await.map(drop)
    }

    async fn key(&self, keys: &str) -> Result<()> {
        let parts: Vec<String> = keys.replace(' ', "").split('+').filter(|p| !p.is_empty()).map(|p| xkey(&p.to_lowercase()).map_or_else(|| p.to_string(), str::to_string)).collect();
        if parts.is_empty() {
            return Err(Error::tool("no key was named", "computer"));
        }
        self.x(&format!("key -- {}", quote(&parts.join("+"))), &format!("press {keys}")).await.map(drop)
    }

    async fn scroll(&self, x: i64, y: i64, direction: &str, amount: i64) -> Result<()> {
        let button = match direction {
            "up" => 4,
            "left" => 6,
            "right" => 7,
            _ => 5,
        };
        self.x(&format!("mousemove {x} {y} click --repeat {amount} --delay 20 {button}"), "scroll").await.map(drop)
    }
}

/// How [`computer_tool`] builds the tool.
///
/// `permission` `Some(Ask)` approves every action; the default (`None`) asks
/// only when the screen is this machine's own desktop. `settle` is seconds to
/// let the screen react before it is looked at again.
#[derive(Debug, Clone)]
pub struct ComputerToolOptions {
    pub name: String,
    pub permission: Option<Permission>,
    pub settle: f64,
}

impl Default for ComputerToolOptions {
    fn default() -> Self {
        Self { name: "computer".into(), permission: None, settle: 0.8 }
    }
}

fn number(args: &Map<String, Value>, key: &str) -> Option<f64> {
    args.get(key).and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.trim().parse().ok())))
}

/// The tool an agent uses a computer with.
pub fn computer_tool(screen: Arc<dyn Computer>, options: &ComputerToolOptions) -> Tool {
    let name = options.name.clone();
    let permission = options.permission.unwrap_or(if screen.local() { Permission::Ask } else { Permission::Allow });
    let mut actions: Vec<&str> = ACTIONS.to_vec();
    if screen.can_open() {
        actions.push("open");
    }
    let opens = screen.can_open();
    let properties = json!({
        "action": {"type": "string", "enum": actions, "description": format!(
            "screenshot: look at the screen. click, double_click, right_click, middle_click, move: at x, y. drag: from x, y to to_x, to_y. type: the text, into whatever has the focus. key: press the key or combination in text (Enter, Tab, Escape, ctrl+a). scroll: turn the wheel in direction, at x, y. wait: for amount seconds.{}",
            if opens { " open: go to the address in text." } else { "" })},
        "x": {"type": "number", "description": "pixels from the left edge of the screenshot."},
        "y": {"type": "number", "description": "pixels from the top edge of the screenshot."},
        "text": {"type": "string", "description": format!("what to type, the key to press{}", if opens { ", or the address to open." } else { "." })},
        "to_x": {"type": "number", "description": "where a drag ends."},
        "to_y": {"type": "number", "description": "where a drag ends."},
        "direction": {"type": "string", "enum": ["up", "down", "left", "right"], "description": "which way to scroll."},
        "amount": {"type": "integer", "description": "wheel notches to scroll, or seconds to wait."},
    });
    let size = if screen.width() > 0 { format!(" The screen is {}x{} pixels.", screen.width(), screen.height()) } else { String::new() };
    let description = format!(
        "Use a {} the way a person does: look at the screen, move the mouse, click, type. Every action answers with a screenshot of what the screen shows afterwards — look at it before the next action, and take a screenshot first to see where things are. Click the middle of what you mean to hit.{size}",
        screen.kind()
    );
    let actions: Vec<String> = actions.into_iter().map(str::to_string).collect();
    let started = Arc::new(tokio::sync::Mutex::new(false));
    let settle = options.settle;
    let tool_name = name.clone();
    Tool::raw(name, description, json!({"type": "object", "properties": properties, "required": ["action"]}), move |args, _ctx| {
        let (screen, actions, started, name) = (screen.clone(), actions.clone(), started.clone(), tool_name.clone());
        async move {
            let err = |m: String| Error::tool(m, &name);
            let given = args.get("action").and_then(Value::as_str).unwrap_or("").trim().to_lowercase().replace('-', "_");
            let action = match given.as_str() {
                "left_click" => "click",
                "mouse_move" => "move",
                "press" | "keypress" => "key",
                "left_click_drag" => "drag",
                "navigate" | "goto" => "open",
                other => other,
            }
            .to_string();
            if !actions.contains(&action) {
                return Err(err(format!("action is one of {} — got {}", actions.join(", "), crate::toolkits::openapi::py_repr(&action))));
            }
            let text = args.get("text").map(|t| t.as_str().map_or_else(|| t.to_string(), str::to_string)).filter(|t| !t.is_empty() && t != "null").unwrap_or_default();
            // One action at a time: a screen can only do one thing at once.
            let mut ready = started.lock().await;
            let outcome: Result<(String, MediaBlock)> = async {
                if !*ready {
                    screen.start().await?;
                    *ready = true;
                }
                let (w, h) = (i64::from(screen.width()), i64::from(screen.height()));
                let point = |x: Option<f64>, y: Option<f64>, what: &str| -> Result<(i64, i64)> {
                    let (Some(x), Some(y)) = (x, y) else { return Err(err(format!("{what} needs x and y — a point on the screenshot"))) };
                    let (px, py) = (x.round_ties_even() as i64, y.round_ties_even() as i64);
                    if !(0 <= px && px < w && 0 <= py && py < h) {
                        return Err(err(format!("({px}, {py}) is off the screen, which is {w}x{h}")));
                    }
                    Ok((px, py))
                };
                let (x, y) = (number(&args, "x"), number(&args, "y"));
                let mut did = "Here is the screen.".to_string();
                match action.as_str() {
                    "click" | "double_click" | "right_click" | "middle_click" => {
                        let (px, py) = point(x, y, &action)?;
                        let button = match action.as_str() {
                            "right_click" => "right",
                            "middle_click" => "middle",
                            _ => "left",
                        };
                        screen.click(px, py, button, if action == "double_click" { 2 } else { 1 }).await?;
                        let words = action.replace('_', " ");
                        let mut c = words.chars();
                        let cap = c.next().map(|f| f.to_uppercase().chain(c).collect::<String>()).unwrap_or_default();
                        did = format!("{cap}ed at ({px}, {py}).");
                    }
                    "move" => {
                        let (px, py) = point(x, y, "move")?;
                        screen.move_to(px, py).await?;
                        did = format!("Moved the pointer to ({px}, {py}).");
                    }
                    "drag" => {
                        let (px, py) = point(x, y, "drag")?;
                        let (qx, qy) = point(number(&args, "to_x"), number(&args, "to_y"), "drag (to_x, to_y)")?;
                        screen.drag(px, py, qx, qy).await?;
                        did = format!("Dragged from ({px}, {py}) to ({qx}, {qy}).");
                    }
                    "type" => {
                        if text.is_empty() {
                            return Err(err("type needs text".into()));
                        }
                        screen.type_text(&text).await?;
                        did = format!("Typed {} characters.", text.chars().count());
                    }
                    "key" => {
                        if text.is_empty() {
                            return Err(err("key needs text: the key to press, e.g. Enter or ctrl+a".into()));
                        }
                        screen.key(&text).await?;
                        did = format!("Pressed {text}.");
                    }
                    "scroll" => {
                        let way = args.get("direction").and_then(Value::as_str).unwrap_or("down").trim().to_lowercase();
                        let way = if way.is_empty() { "down".to_string() } else { way };
                        if !matches!(way.as_str(), "up" | "down" | "left" | "right") {
                            return Err(err("direction is one of up, down, left, right".into()));
                        }
                        let (px, py) = if x.is_some() && y.is_some() { point(x, y, "scroll")? } else { (w / 2, h / 2) };
                        let notches = number(&args, "amount").map_or(3, |a| if a == 0.0 { 3 } else { a as i64 }).clamp(1, 30);
                        screen.scroll(px, py, &way, notches).await?;
                        did = format!("Scrolled {way}.");
                    }
                    "wait" => {
                        let seconds = number(&args, "amount").map_or(1.0, |a| if a == 0.0 { 1.0 } else { a }).clamp(0.0, 30.0);
                        tokio::time::sleep(Duration::from_secs_f64(seconds)).await;
                        did = "Waited.".into();
                    }
                    "open" => {
                        if text.is_empty() {
                            return Err(err("open needs text: the address".into()));
                        }
                        screen.open(&text).await?;
                        did = format!("Opened {text}.");
                    }
                    _ => {}
                }
                if !matches!(action.as_str(), "screenshot" | "wait" | "open") {
                    tokio::time::sleep(Duration::from_secs_f64(settle.max(0.0))).await;
                }
                Ok((did, screen.screenshot().await?))
            }
            .await;
            drop(ready);
            let (did, shot) = outcome.map_err(|e| match e {
                Error::Cdp { message, .. } => err(format!("the {} could not do that: {message}", screen.kind())),
                other => other,
            })?;
            Ok(json!([format!("{did} The screen is {}x{}.", screen.width(), screen.height()), ContentBlock::Image(shot)]))
        }
    })
    .tags(["builtin", "computer"])
    .permission(permission)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolContext;

    /// A screen that only writes down what was done to it.
    #[derive(Default)]
    struct Paper {
        done: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl Computer for Paper {
        fn width(&self) -> u32 {
            800
        }
        fn height(&self) -> u32 {
            600
        }
        async fn screenshot(&self) -> Result<MediaBlock> {
            Ok(MediaBlock::from_bytes(b"\x89PNG", "image/png"))
        }
        async fn click(&self, x: i64, y: i64, button: &str, count: i64) -> Result<()> {
            self.done.lock().unwrap().push(format!("click {x} {y} {button} {count}"));
            Ok(())
        }
        async fn move_to(&self, x: i64, y: i64) -> Result<()> {
            self.done.lock().unwrap().push(format!("move {x} {y}"));
            Ok(())
        }
        async fn drag(&self, x: i64, y: i64, to_x: i64, to_y: i64) -> Result<()> {
            self.done.lock().unwrap().push(format!("drag {x} {y} {to_x} {to_y}"));
            Ok(())
        }
        async fn type_text(&self, text: &str) -> Result<()> {
            self.done.lock().unwrap().push(format!("type {text}"));
            Ok(())
        }
        async fn key(&self, keys: &str) -> Result<()> {
            self.done.lock().unwrap().push(format!("key {keys}"));
            Ok(())
        }
        async fn scroll(&self, x: i64, y: i64, direction: &str, amount: i64) -> Result<()> {
            self.done.lock().unwrap().push(format!("scroll {x} {y} {direction} {amount}"));
            Ok(())
        }
    }

    #[tokio::test]
    async fn every_action_answers_with_the_screen() {
        let paper = Arc::new(Paper::default());
        let tool = computer_tool(paper.clone(), &ComputerToolOptions { settle: 0.0, ..Default::default() });
        assert!(tool.description.ends_with("The screen is 800x600 pixels."));
        assert_eq!(tool.permission, Permission::Allow);
        assert!(!tool.parameters["properties"]["action"]["enum"].as_array().unwrap().contains(&json!("open")));
        let run = |args: Value| {
            let tool = tool.clone();
            async move { tool.run("c", args.as_object().unwrap().clone(), ToolContext::default()).await }
        };
        let out = run(json!({"action": "double_click", "x": 10.5, "y": 20})).await;
        assert_eq!(out.content, "Double clicked at (10, 20). The screen is 800x600.");
        run(json!({"action": "left_click_drag", "x": 1, "y": 2, "to_x": 3, "to_y": 4})).await;
        run(json!({"action": "scroll", "direction": "up"})).await;
        run(json!({"action": "press", "text": "ctrl+a"})).await;
        assert!(run(json!({"action": "click", "x": 900, "y": 1})).await.content.contains("(900, 1) is off the screen, which is 800x600"));
        assert!(run(json!({"action": "fly"})).await.content.contains("action is one of screenshot, click"));
        assert!(run(json!({"action": "open", "text": "x"})).await.content.contains("got 'open'"));
        assert_eq!(paper.done.lock().unwrap().clone(), vec!["click 10 20 left 2", "drag 1 2 3 4", "scroll 400 300 up 3", "key ctrl+a"]);
        assert_eq!(quote("it's"), "'it'\"'\"'s'");
        let desktop = DesktopComputer::new(None);
        assert!(desktop.local());
        assert_eq!(computer_tool(Arc::new(desktop), &ComputerToolOptions::default()).permission, Permission::Ask);
    }
}
