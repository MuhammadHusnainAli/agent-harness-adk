"""agent-harness — a fast, lightweight harness for production AI agents.

    from agent_harness import Agent, tool

    @tool
    def order_status(order_id: str) -> str:
        '''Look up an order.

        Args:
            order_id: the order number.
        '''
        return db.lookup(order_id)

    agent = Agent("support", "Answer billing questions.", tools=[order_status])
    print(agent.run_sync("Where is order 4182?").output)

Everything is optional except the agent itself: add sub-agents, skills, memory,
MCP servers and the runtime rails as the job needs them.
"""

from __future__ import annotations

from .agent import Agent
from .blueprint import AgentEntry, Blueprint
from .context import ContextAssembler, ContextCompactor, estimate_tokens
from .errors import (
    BudgetExceeded,
    ConfigurationError,
    GuardrailTripped,
    HarnessError,
    MaxStepsExceeded,
    MCPError,
    OutputContractError,
    PermissionDenied,
    ProviderError,
    RateLimitError,
    StopRequested,
    ToolError,
    ToolNotFound,
)
from .evaluation import (
    Comparison,
    EvalReport,
    Evaluator,
    Expect,
    GoldenTask,
    TaskOutcome,
    llm_judge,
)
from .guardrails import (
    POLICIES,
    AgentGuardrails,
    Check,
    CompletionContext,
    Custom,
    DetectorCheck,
    Finding,
    ForbidTools,
    Grounded,
    GroundednessDetector,
    InjectionDetector,
    LLMGuard,
    LLMVerdict,
    MaxCost,
    MaxSteps,
    MinLength,
    MustInclude,
    MustMatch,
    MustNotInclude,
    NoInjection,
    NoPII,
    NoPlaceholders,
    NoRepetition,
    NoSecrets,
    NotToxic,
    PIIDetector,
    RepetitionDetector,
    RequireCitation,
    RequireJSON,
    RequireStructured,
    RequireTools,
    SecretDetector,
    ToxicityDetector,
    Violation,
)
from .harness import Harness
from .memory import (
    Embedder,
    FileStore,
    HashEmbedder,
    InMemoryStore,
    MemoryManager,
    MemoryRecord,
    MemoryStore,
    OrchestratorMemory,
    ProviderEmbedder,
    SemanticMemory,
    SessionMemory,
    SubAgentMemory,
    Trace,
    UserMemory,
    VectorStore,
    available_backends,
    memory_provider,
)
from .orchestrator import Orchestrator, Plan, Review, Task
from .prompts import Prompt, PromptLibrary
from .providers import (
    AnthropicProvider,
    CompletionRequest,
    FakeProvider,
    GeminiProvider,
    ModelInfo,
    OpenAIProvider,
    Provider,
    ToolSchema,
    get_provider,
    register_model,
    register_provider,
    tool_call,
)
from .runtime import (
    AuditEntry,
    AuditTrail,
    Budget,
    BudgetGuard,
    Checkpoint,
    Checkpointer,
    ComponentHealth,
    ConcurrencyScheduler,
    DeliverableStore,
    FileSessionStore,
    Guardrails,
    HookContext,
    HookEngine,
    InMemorySessionStore,
    ModelRouter,
    PolicyGate,
    RateGuard,
    RateLimit,
    RecordingProvider,
    Replayer,
    ReplayProvider,
    ResultCache,
    RunJournal,
    ServiceHealth,
    Session,
    SessionStore,
    Span,
    StopController,
    StopState,
    Tracer,
    Workspace,
    WorkspaceBroker,
    console_exporter,
    jsonl_exporter,
)
from .runtime.permissions import Rule as PermissionRule
from .runtime.router import RouteRule
from .skills import Skill, SkillRegistry
from .spec import CompiledSpec, SpecCompiler
from .subagents import Bench, SubAgentFactory, SubAgentSpec
from .tools import Tool, ToolContext, ToolRegistry, tool
from .types import (
    Artifact,
    Message,
    ModelResponse,
    RunResult,
    StreamEvent,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from .versioning import AgentVersion

__version__ = "0.1.1"

# MCP pulls in the HTTP stack, and most agents never touch it — so it loads on
# first use. `from agent_harness import MCPServer` still works.
_LAZY = {"MCPServer": "mcp", "MCPClient": "mcp", "MCPManager": "mcp"}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(f".{_LAZY[name]}", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))

__all__ = [
    "__version__",
    # core
    "Agent",
    "Orchestrator",
    "Harness",
    "Plan",
    "Task",
    "Review",
    # sub-agents
    "SubAgentSpec",
    "AgentVersion",
    "Bench",
    "SubAgentFactory",
    # tools and skills
    "tool",
    "Tool",
    "ToolRegistry",
    "ToolContext",
    "Skill",
    "SkillRegistry",
    # prompts
    "Prompt",
    "PromptLibrary",
    # memory
    "MemoryManager",
    "MemoryStore",
    "MemoryRecord",
    "InMemoryStore",
    "FileStore",
    "UserMemory",
    "SessionMemory",
    "OrchestratorMemory",
    "SubAgentMemory",
    "SemanticMemory",
    "VectorStore",
    "Trace",
    "memory_provider",
    "available_backends",
    "Embedder",
    "HashEmbedder",
    "ProviderEmbedder",
    # providers
    "Provider",
    "AnthropicProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "FakeProvider",
    "CompletionRequest",
    "ToolSchema",
    "ModelInfo",
    "get_provider",
    "register_provider",
    "register_model",
    "tool_call",
    # mcp
    "MCPServer",
    "MCPClient",
    "MCPManager",
    # evaluation
    "Evaluator",
    "GoldenTask",
    "Expect",
    "EvalReport",
    "TaskOutcome",
    "Comparison",
    "llm_judge",
    # spec compilation
    "SpecCompiler",
    "CompiledSpec",
    "Blueprint",
    "AgentEntry",
    # guardrails
    "AgentGuardrails",
    "CompletionContext",
    "Violation",
    "Check",
    "RequireTools",
    "ForbidTools",
    "MustInclude",
    "MustNotInclude",
    "MustMatch",
    "MinLength",
    "MaxSteps",
    "MaxCost",
    "RequireCitation",
    "RequireJSON",
    "RequireStructured",
    "NoPlaceholders",
    "Custom",
    "NoPII",
    "NoSecrets",
    "NoInjection",
    "NotToxic",
    "NoRepetition",
    "Grounded",
    "DetectorCheck",
    "PIIDetector",
    "SecretDetector",
    "InjectionDetector",
    "ToxicityDetector",
    "GroundednessDetector",
    "RepetitionDetector",
    "Finding",
    "LLMGuard",
    "LLMVerdict",
    "POLICIES",
    # runtime rails
    "Budget",
    "BudgetGuard",
    "RateLimit",
    "RateGuard",
    "StopController",
    "StopState",
    "AuditTrail",
    "AuditEntry",
    "ServiceHealth",
    "ComponentHealth",
    "DeliverableStore",
    "Replayer",
    "RecordingProvider",
    "ReplayProvider",
    "PolicyGate",
    "PermissionRule",
    "HookEngine",
    "HookContext",
    "Guardrails",
    "Tracer",
    "Span",
    "console_exporter",
    "jsonl_exporter",
    "RunJournal",
    "ResultCache",
    "ConcurrencyScheduler",
    "ModelRouter",
    "RouteRule",
    "Session",
    "SessionStore",
    "InMemorySessionStore",
    "FileSessionStore",
    "Checkpoint",
    "Checkpointer",
    "Workspace",
    "WorkspaceBroker",
    # context
    "ContextAssembler",
    "ContextCompactor",
    "estimate_tokens",
    # types
    "Message",
    "ModelResponse",
    "RunResult",
    "StreamEvent",
    "Usage",
    "Artifact",
    "TextBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    # errors
    "HarnessError",
    "ConfigurationError",
    "ProviderError",
    "RateLimitError",
    "ToolError",
    "ToolNotFound",
    "PermissionDenied",
    "BudgetExceeded",
    "GuardrailTripped",
    "MaxStepsExceeded",
    "OutputContractError",
    "StopRequested",
    "MCPError",
]
