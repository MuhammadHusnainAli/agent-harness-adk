//! The `agent-harness` command. Built with `--features cli`.

fn main() {
    std::process::exit(agent_harness::cli::main());
}
