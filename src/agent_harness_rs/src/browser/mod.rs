//! Browser and computer use: a web browser, or a whole screen, for an agent.
//!
//! ```no_run
//! # fn demo() -> agent_harness::Result<()> {
//! use agent_harness::{Agent, browser::Browser};
//! let browser = Browser::default();
//! let agent = Agent::builder("researcher").tools(browser.tools()).build()?;
//! # Ok(()) }
//! ```
//!
//! See [`Browser`] for the browser, [`computer_tool`] for a mouse and keyboard.

#[allow(clippy::module_inception)]
pub mod browser;
pub mod cdp;
pub mod computer;
pub mod tools;

pub use browser::{BROWSER_ENV, Browser, BrowserOptions, find_browser};
pub use cdp::{Cdp, Listener, cdp_error};
pub use computer::{ACTIONS, BrowserComputer, Computer, ComputerToolOptions, DesktopComputer, computer_tool};
pub use tools::{BrowserToolsOptions, browser_tools};
