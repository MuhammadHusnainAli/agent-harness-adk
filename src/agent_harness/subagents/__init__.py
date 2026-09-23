"""Sub-agents: the bench you reuse, and the factory that writes a new one.

Two supplies, one decision. Before staffing a task the orchestrator asks: is
there a pre-defined sub-agent that already covers this? If yes it is reused as
it is. If no, the factory writes a brand-new specialist during the run —
blueprint, prompt, tool allowlist, model and effort, workspace isolation and an
input/output contract — and that specialist exists only for this job.
"""

from .bench import BENCH, BENCH_SPECS, Bench
from .builder import build_agent
from .factory import FACTORY_PROMPT, SubAgentFactory
from .spec import SubAgentSpec

__all__ = [
    "SubAgentSpec",
    "build_agent",
    "Bench",
    "BENCH",
    "BENCH_SPECS",
    "SubAgentFactory",
    "FACTORY_PROMPT",
]
