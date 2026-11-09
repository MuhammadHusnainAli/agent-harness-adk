//! The bench: pre-defined sub-agents — proven, versioned, reused as they are.

use indexmap::IndexMap;

use super::spec::{SubAgentSpec, WorkspaceKind};

fn spec(name: &str, description: &str, instructions: &str, tags: &[&str], tier: &str, workspace: WorkspaceKind) -> SubAgentSpec {
    SubAgentSpec {
        tags: tags.iter().map(|t| t.to_string()).collect(),
        tier: Some(tier.into()),
        workspace,
        ..SubAgentSpec::new(name, description, instructions)
    }
}

/// The standard bench, as the Python harness ships it.
#[allow(non_snake_case)]
pub fn BENCH_SPECS() -> Vec<SubAgentSpec> {
    use WorkspaceKind::{None as N, Shared};
    vec![
        spec(
            "research",
            "Maps the source material and reports what it says, read-only.",
            "Find what is actually there and report it. Quote or cite the source for every claim. If the sources disagree, say so rather than picking one. Never guess at a fact you could not find — list it as an open question.",
            &["research", "discovery", "search", "sources", "investigate"],
            "balanced",
            N,
        ),
        spec(
            "planner",
            "Turns a goal into an ordered task graph with parallelism marked.",
            "Break the goal into tasks that one worker could finish alone. For each task give: a one-line statement, what it depends on, and how we will know it is done. Mark which tasks can run at the same time. Keep the plan as short as the goal allows.",
            &["plan", "planning", "breakdown", "scope", "schedule"],
            "deep",
            N,
        ),
        spec(
            "document_extractor",
            "Pulls structured fields out of documents, read-only.",
            "Extract exactly the fields you were asked for. Use null when a field is genuinely absent — never invent a plausible value. Note the location in the source for anything you extracted.",
            &["extract", "document", "pdf", "fields", "parse", "invoice"],
            "fast",
            N,
        ),
        spec(
            "data_analyst",
            "Aggregates data and explains the variance behind the numbers.",
            "State the number, then what drives it. Show the calculation you used. Call out the caveats in the data before someone acts on it.",
            &["data", "analysis", "numbers", "metrics", "variance", "statistics"],
            "balanced",
            Shared,
        ),
        spec(
            "validator",
            "Re-checks work against the rules and reports pass or fail.",
            "Check the work against the stated rules, one by one. For each: pass or fail, and the evidence. Do not fix anything — report. End with an overall verdict and the single most important problem.",
            &["validate", "check", "qa", "verify", "review", "test"],
            "balanced",
            N,
        ),
        spec(
            "compliance_checker",
            "Checks policy and data-handling rules, read-only.",
            "Check the work against the policies you were given. Quote the clause behind every finding. Flag anything that handles personal or regulated data. When a rule is ambiguous, say it is ambiguous — do not resolve it.",
            &["compliance", "policy", "legal", "regulation", "privacy", "risk"],
            "deep",
            N,
        ),
        spec(
            "drafting",
            "Writes correspondence and short-form copy to a brief.",
            "Write to the brief and the audience you were given. Plain sentences, no filler, no hedging. Match the register you were asked for; if none was given, write like a competent colleague.",
            &["draft", "write", "email", "letter", "copy", "correspondence"],
            "balanced",
            N,
        ),
        spec(
            "report_writer",
            "Consolidates findings into one structured report.",
            "Lead with the answer, then the evidence. Merge duplicate findings and attribute each one to where it came from. Keep every number traceable to its source. No filler sections.",
            &["report", "summary", "consolidate", "write-up", "document"],
            "balanced",
            Shared,
        ),
    ]
}

/// The pre-defined sub-agents: proven, versioned, evaluated, reused as they are.
#[derive(Debug, Clone, Default)]
pub struct Bench {
    specs: IndexMap<String, SubAgentSpec>,
}

impl Bench {
    pub fn new<I: IntoIterator<Item = SubAgentSpec>>(specs: I) -> Self {
        Self { specs: specs.into_iter().map(|s| (s.name.clone(), s)).collect() }
    }

    pub fn standard() -> Self {
        Self::new(BENCH_SPECS())
    }

    pub fn register(&mut self, spec: SubAgentSpec) {
        self.specs.insert(spec.name.clone(), spec);
    }

    pub fn get(&self, name: &str) -> Option<&SubAgentSpec> {
        self.specs.get(name)
    }

    pub fn names(&self) -> Vec<String> {
        let mut n: Vec<String> = self.specs.keys().cloned().collect();
        n.sort();
        n
    }

    pub fn iter(&self) -> impl Iterator<Item = &SubAgentSpec> {
        self.specs.values()
    }

    pub fn len(&self) -> usize {
        self.specs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }

    /// The staffing decision: is there already a sub-agent that covers this?
    pub fn find(&self, task: &str, threshold: f64) -> Option<&SubAgentSpec> {
        let mut best: (f64, Option<&SubAgentSpec>) = (0.0, None);
        for s in self.specs.values() {
            let score = s.matches(task);
            if score > best.0 {
                best = (score, Some(s));
            }
        }
        if best.0 >= threshold { best.1 } else { None }
    }

    pub fn catalogue(&self) -> String {
        self.specs.values().map(|s| format!("- {}: {}", s.name, s.description)).collect::<Vec<_>>().join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_right_specialist() {
        let b = Bench::standard();
        assert_eq!(b.len(), 8);
        assert_eq!(b.find("validate the invoice totals against the rules", 0.3).unwrap().name, "validator");
        assert_eq!(b.find("plan the migration schedule", 0.3).unwrap().name, "planner");
        assert!(b.find("hello there", 0.3).is_none());
        assert!(b.catalogue().starts_with("- research: Maps the source material"));
    }
}
