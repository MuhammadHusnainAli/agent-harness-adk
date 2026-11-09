//! Sandboxed compute: run code, get the answer, without it touching your machine.
//!
//! The agent writes Python, it runs inside the workspace, and only what it
//! printed comes back. In a sandbox ([`crate::sandboxes`]) or the Docker
//! workspace the process is contained. With the local backend it is a plain
//! subprocess in a jailed directory, which is weaker: use a sandbox for
//! anything you did not write.
//!
//! On this machine the tool asks for approval before every run. That is
//! deliberate. Inside a sandbox, which is its own machine, it does not.

use std::sync::Arc;

use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::errors::Error;
use crate::runtime::workspace::Workspace;
use crate::tools::{Permission, Tool, ToolContext};

/// Imports made available to every snippet.
pub const PREAMBLE: &str = "import json, math, statistics, re, csv, datetime\nfrom pathlib import Path\n";

#[derive(Deserialize, JsonSchema)]
struct PythonArgs {
    /// the Python to run.
    code: String,
    /// seconds before it is killed.
    #[serde(default)]
    timeout_s: Option<f64>,
}

fn last_chars(text: &str, n: usize) -> String {
    let count = text.chars().count();
    text.chars().skip(count.saturating_sub(n)).collect()
}

/// A sandboxed Python tool bound to a workspace, named `run_python`, with a
/// 60-second limit and the standard preamble. See [`make_python_tool_with`].
pub fn make_python_tool(workspace: Arc<dyn Workspace>) -> Tool {
    make_python_tool_with(workspace, "run_python", 60.0, PREAMBLE)
}

/// A sandboxed Python tool bound to a workspace.
///
/// - `workspace`: the workspace the code runs inside.
/// - `name`: the tool name the model sees.
/// - `timeout`: seconds before the process is killed, unless the model asks for another.
/// - `preamble`: imports made available to the snippet.
///
/// The snippet is written to `_snippet.py` in the workspace and run there with
/// the workspace's interpreter. The tool asks for approval unless the
/// workspace is isolated (its own machine).
pub fn make_python_tool_with(workspace: Arc<dyn Workspace>, name: &str, timeout: f64, preamble: &str) -> Tool {
    let permission = if workspace.isolated() { Permission::Allow } else { Permission::Ask };
    let preamble = preamble.to_string();
    let tool_name = name.to_string();
    let mut tool = Tool::new(name, move |args: PythonArgs, _ctx: ToolContext| {
        let (workspace, preamble, name) = (workspace.clone(), preamble.clone(), tool_name.clone());
        async move {
            if args.code.trim().is_empty() {
                return Err(Error::tool("no code to run", &name));
            }
            let script = format!("{preamble}\n{}\n", args.code);
            workspace.write("_snippet.py", script.as_bytes(), false).await?;
            let python = workspace.python();
            let python = if python.contains(' ') && !python.starts_with('"') { format!("\"{python}\"") } else { python };
            let result = workspace.shell(&format!("{python} _snippet.py"), Some(args.timeout_s.unwrap_or(timeout))).await?;
            let text = |key: &str| result.get(key).and_then(Value::as_str).unwrap_or("").to_string();
            let code = result.get("returncode").cloned().unwrap_or(json!(-1));
            if code != json!(0) {
                let stderr = last_chars(&text("stderr"), 4000);
                return Ok(json!({
                    "ok": false,
                    "stdout": text("stdout"),
                    "error": if stderr.is_empty() { "the snippet exited non-zero".to_string() } else { stderr },
                    "returncode": code,
                }));
            }
            let stderr = last_chars(&text("stderr"), 2000);
            Ok(json!({"ok": true, "stdout": text("stdout"), "stderr": if stderr.is_empty() { Value::Null } else { Value::String(stderr) }}))
        }
    })
    .describe(
        "Run Python inside the workspace and return what it printed.\n\nThe snippet starts in the workspace directory with json, math,\nstatistics, re, csv, datetime and pathlib already imported. Print what\nyou want back \u{2014} nothing else is returned.",
    )
    .tags(["builtin", "compute"])
    .permission(permission);
    if let Some(prop) = tool.parameters.pointer_mut("/properties/timeout_s").and_then(Value::as_object_mut) {
        prop.insert("type".into(), json!("number"));
        prop.insert("default".into(), json!(timeout));
    }
    tool
}

#[cfg(test)]
mod tests {
    use serde_json::Map;

    use super::*;
    use crate::runtime::workspace::{LocalWorkspace, WorkspaceOptions};

    fn args(v: Value) -> Map<String, Value> {
        v.as_object().cloned().unwrap()
    }

    /// A shell-enabled workspace, or None when no Python is on PATH.
    fn workspace(tag: &str) -> Option<Arc<dyn Workspace>> {
        let has_python =
            std::process::Command::new(if cfg!(windows) { "python" } else { "python3" }).arg("--version").output().is_ok_and(|o| o.status.success());
        if !has_python {
            return None;
        }
        let root = std::env::temp_dir().join(format!("ah-compute-{tag}-{}", uuid::Uuid::new_v4()));
        let ws = LocalWorkspace::new(root, WorkspaceOptions { allow_shell: true, ephemeral: true, ..Default::default() }).unwrap();
        Some(Arc::new(ws))
    }

    #[tokio::test]
    async fn python_runs_in_the_workspace_and_returns_what_it_printed() {
        let Some(ws) = workspace("run") else { return };
        ws.write("data.txt", b"hello from the workspace", false).await.unwrap();
        let entry = make_python_tool(ws.clone());
        assert_eq!(entry.permission, Permission::Ask);
        assert_eq!(entry.parameters["properties"]["timeout_s"]["default"], json!(60.0));
        let out = entry.invoke(args(json!({"code": "print(sum(range(10)))"})), ToolContext::default()).await.unwrap();
        assert_eq!(out["ok"], true);
        assert_eq!(out["stdout"].as_str().unwrap().trim(), "45");
        assert_eq!(out["stderr"], Value::Null);

        let out = entry.invoke(args(json!({"code": "print(json.dumps({'root': math.sqrt(16)}))"})), ToolContext::default()).await.unwrap();
        let parsed: Value = serde_json::from_str(out["stdout"].as_str().unwrap().trim()).unwrap();
        assert_eq!(parsed["root"], json!(4.0));

        let out = entry.invoke(args(json!({"code": "print(Path('data.txt').read_text())"})), ToolContext::default()).await.unwrap();
        assert!(out["stdout"].as_str().unwrap().contains("hello from the workspace"));

        let out = entry.invoke(args(json!({"code": "raise ValueError('nope')"})), ToolContext::default()).await.unwrap();
        assert_eq!(out["ok"], false);
        let error = out["error"].as_str().unwrap();
        assert!(error.contains("ValueError") && error.contains("nope"), "{error}");
        assert_eq!(out["returncode"], 1);
        ws.close().await;
    }

    #[tokio::test]
    async fn a_runaway_snippet_is_killed_and_empty_code_refused() {
        let Some(ws) = workspace("slow") else { return };
        let entry = make_python_tool(ws.clone());
        let e = entry.invoke(args(json!({"code": "import time; time.sleep(30)", "timeout_s": 0.3})), ToolContext::default()).await.unwrap_err();
        assert!(e.to_string().contains("timed out"), "{e}");
        let e = entry.invoke(args(json!({"code": "   "})), ToolContext::default()).await.unwrap_err();
        assert_eq!(e.to_string(), "no code to run");
        ws.close().await;
    }
}
