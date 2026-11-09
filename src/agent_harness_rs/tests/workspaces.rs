//! Workspaces and sandboxes, end to end through an agent.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use agent_harness::errors::{Error, Result};
use agent_harness::llm_providers::{FakeProvider, Script, tool_call};
use agent_harness::modes;
use agent_harness::runtime::workspace::on_path;
use agent_harness::runtime::{LocalWorkspace, WorkspaceOptions};
use agent_harness::sandboxes::{ExecResult, SandboxBackend, register_sandbox, run_process};
use agent_harness::types::ContentBlock;
use agent_harness::{Agent, Harness, RunOptions};
use async_trait::async_trait;
use serde_json::{Map, Value, json};

fn fake(script: Vec<Script>) -> Arc<FakeProvider> {
    Arc::new(FakeProvider::new(script))
}

fn temp(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("ah_{tag}_{}", uuid::Uuid::new_v4().simple()))
}

#[tokio::test]
async fn cowork_hands_back_the_files_it_wrote() {
    let dir = temp("cowork");
    let ws = Arc::new(LocalWorkspace::new(&dir, WorkspaceOptions::default()).unwrap());
    let p = fake(vec![
        tool_call("todo_write", json!({"todos": [{"content": "write the plan", "status": "done"}]})),
        tool_call("fs_write", json!({"path": "plan.md", "content": "# Plan\n1. ship"})),
        "Wrote plan.md.".into(),
    ]);
    let a = Agent::builder("colleague").provider(p.clone()).memory(false).mode(modes::cowork(Some("fast"), None).unwrap()).workspace(ws).build().unwrap();
    assert!(a.tools().contains("fs_write") && !a.tools().contains("shell"));
    let r = a.run("write a plan").await.unwrap();
    assert!(r.ok(), "{:?}", r.error);
    let plan = r.artifacts.iter().find(|a| a.name == "plan.md").expect("the file it wrote");
    assert_eq!(plan.content, "# Plan\n1. ship");
    assert_eq!(plan.produced_by, "colleague");
    assert!(a.harness().deliverables.contains("plan.md"));
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn a_folder_from_the_broker_and_tools_run_under_the_scheduler() {
    let calls = Arc::new(AtomicU32::new(0));
    let p = fake(vec![
        Script::Blocks(vec![
            ContentBlock::ToolUse(agent_harness::types::ToolUseBlock {
                id: "a".into(),
                name: "fs_write".into(),
                input: json!({"path": "a.txt", "content": "1"}).as_object().cloned().unwrap(),
                thought_signature: None,
            }),
            ContentBlock::ToolUse(agent_harness::types::ToolUseBlock {
                id: "b".into(),
                name: "fs_write".into(),
                input: json!({"path": "b.txt", "content": "2"}).as_object().cloned().unwrap(),
                thought_signature: None,
            }),
        ]),
        "done".into(),
    ]);
    let harness = Harness::testing(Some(p.clone()));
    let a = Agent::builder("w").harness(harness.clone()).memory(false).workspace_spec(json!(true)).build().unwrap();
    let _ = calls.load(Ordering::SeqCst);
    a.run("write two files").await.unwrap();
    let ws = a.workspace().unwrap();
    assert_eq!(ws.listdir(".").await.unwrap(), vec!["a.txt", "b.txt"]);
    assert_eq!(harness.scheduler.stats()["completed"], 2);
    harness.close().await;
}

/// A "sandbox" that is a folder on this machine run through `sh` — enough to
/// exercise the sandbox contract where a POSIX shell exists. Ids it was given
/// before it was "restarted" are gone, the way an expired cloud sandbox is.
struct FolderSandbox {
    root: PathBuf,
    live: std::sync::Mutex<Vec<String>>,
}

#[async_trait]
impl SandboxBackend for FolderSandbox {
    fn name(&self) -> &str {
        "folder"
    }

    async fn start(&self, attach: Option<&str>) -> Result<String> {
        let mut live = self.live.lock().unwrap();
        if let Some(id) = attach {
            if !live.iter().any(|l| l == id) {
                return Err(Error::Io(format!("sandbox {id} does not exist")));
            }
            return Ok(id.into());
        }
        let id = format!("fold_{}", live.len() + 1);
        std::fs::create_dir_all(self.root.join(&id)).unwrap();
        live.push(id.clone());
        Ok(id)
    }

    async fn exec(&self, command: &str, cwd: Option<&str>, env: &BTreeMap<String, String>, timeout: f64) -> Result<ExecResult> {
        let base = self.root.to_string_lossy().replace('\\', "/");
        let body = agent_harness::sandboxes::base::script(command, cwd.or(Some(&base)), env);
        match run_process(vec!["sh".into(), "-c".into(), body], None, timeout).await? {
            Some((code, out, err)) => Ok(ExecResult::new(code, String::from_utf8_lossy(&out), String::from_utf8_lossy(&err))),
            None => Ok(ExecResult { returncode: 124, timed_out: true, ..Default::default() }),
        }
    }
}

#[tokio::test]
async fn a_conversation_remembers_its_sandbox_and_survives_losing_it() {
    if !(on_path("sh") && on_path("base64") && on_path("stat") && on_path("find")) {
        return; // needs a POSIX userland (Git for Windows has one)
    }
    let dir = temp("sbx");
    let root = dir.clone();
    let backend = Arc::new(FolderSandbox { root: root.clone(), live: Default::default() });
    let shared = backend.clone();
    register_sandbox("folder", Arc::new(move |_: &Map<String, Value>| Ok(shared.clone() as Arc<dyn SandboxBackend>)), "", &[], &[], "a folder, for tests");
    let workdir = format!("{}/work", root.to_string_lossy().replace('\\', "/"));

    let harness = Harness::testing(None);
    let p = fake(vec![tool_call("fs_write", json!({"path": "notes.md", "content": "keep me"})), "noted".into()]);
    let first = Agent::builder("colleague")
        .provider(p)
        .harness(harness.clone())
        .memory(false)
        .mode(modes::cowork(Some("fast"), None).unwrap())
        .workspace_spec(json!({"sandbox": "folder", "workdir": workdir}))
        .build()
        .unwrap();
    assert!(first.tools().contains("shell"), "a sandbox's shell is its own, and on by default");
    let r = first.run("take a note").await.unwrap();
    assert!(r.ok(), "{:?}", r.error);
    assert_eq!(r.sandbox_id, "fold_1");
    let held = harness.sessions.load(&r.session_id).await.unwrap();
    assert_eq!(held.metadata["sandbox"]["id"], "fold_1");
    assert_eq!(held.metadata["sandbox_files"]["notes.md"], "keep me");
    first.workspace().unwrap().close().await;

    // The sandbox expires; a new agent picks the conversation up anyway.
    backend.live.lock().unwrap().clear();
    let _ = std::fs::remove_dir_all(&workdir);
    let p2 = fake(vec!["it is still there".into()]);
    let second = Agent::builder("colleague")
        .provider(p2.clone())
        .harness(harness.clone())
        .memory(false)
        .mode(modes::cowork(Some("fast"), None).unwrap())
        .workspace_spec(json!({"sandbox": "folder", "workdir": workdir}))
        .build()
        .unwrap();
    let r2 = second.run_with("is my note there?", RunOptions::session(r.session_id.clone())).await.unwrap();
    assert!(r2.ok(), "{:?}", r2.error);
    let told = p2.requests()[0].messages.last().unwrap().text();
    assert!(told.starts_with("[Workspace notice: the sandbox this conversation was working in (fold_1) is no longer available"), "{told}");
    assert!(told.contains("Put back from the conversation's record: notes.md."));
    assert_eq!(second.workspace().unwrap().read("notes.md").await.unwrap(), "keep me");
    assert!(harness.audit.entries().iter().any(|e| e.action == "sandbox_replaced"));
    harness.close().await;
    let _ = std::fs::remove_dir_all(dir);
}
