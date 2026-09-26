"""Environment-driven configuration (12-factor). Stdlib only.

Every knob the agent has lives here so a container can be reconfigured without a code change:
swap the fixture for the real API with UPSTREAM_BASE_URL, swap the model with OPENAI_BASE_URL/MODEL_ID.
"""
import os
from dataclasses import dataclass


def _env(name, default):
    return os.environ.get(name, default)


def _env_bool(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Config:
    upstream_base_url: str = "http://127.0.0.1:8080"
    model_provider: str = "rules"  # "rules" | "openai_compatible"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    model_id: str = "gpt-4o-mini"
    prompt_version: str = "v1"
    deadline_ms: int = 20000
    read_timeout_s: float = 2.0
    write_timeout_s: float = 3.0
    runs_dir: str = "runs"
    dry_run: bool = True
    kb_filter: bool = True  # KB_FILTER; False disables the applies_to filter (ablation only, never in production)
    max_llm_calls: int = 2
    max_tool_calls: int = 8
    max_output_tokens: int = 600
    # Retry policy (status-based; see upstream.RetryPolicy / llm.OpenAICompatibleLLM). Reads and model
    # calls retry up to MAX_RETRIES times on timeout / 429 / 5xx; writes never retry.
    max_retries: int = 3
    llm_max_retries: int = 3
    llm_timeout_s: float = 30.0
    llm_stream: bool = True  # stream the model call so time-to-first-token / inter-token gaps are measured
    trace_stdout: bool = False  # also emit every trace record as a JSON line on stdout (log forwarding on App Platform)
    service_name: str = "ic4-agent"


def load_config(**overrides) -> Config:
    """Build a Config from the environment, then apply explicit overrides (CLI flags win)."""
    api_key = _env("OPENAI_API_KEY", "")
    provider = _env("MODEL_PROVIDER", "")
    if not provider:
        provider = "openai_compatible" if api_key else "rules"
    values = dict(
        upstream_base_url=_env("UPSTREAM_BASE_URL", "http://127.0.0.1:8080").rstrip("/"),
        model_provider=provider,
        openai_base_url=_env("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        openai_api_key=api_key,
        model_id=_env("MODEL_ID", "gpt-4o-mini"),
        prompt_version=_env("PROMPT_VERSION", "v1"),
        deadline_ms=int(_env("DEADLINE_MS", "20000")),
        read_timeout_s=float(_env("READ_TIMEOUT_S", "2.0")),
        write_timeout_s=float(_env("WRITE_TIMEOUT_S", "3.0")),
        runs_dir=_env("RUNS_DIR", "runs"),
        dry_run=_env_bool("DRY_RUN", True),
        kb_filter=_env_bool("KB_FILTER", True),
        max_llm_calls=int(_env("MAX_LLM_CALLS", "2")),
        max_tool_calls=int(_env("MAX_TOOL_CALLS", "8")),
        max_output_tokens=int(_env("MAX_OUTPUT_TOKENS", "600")),
        max_retries=int(_env("MAX_RETRIES", "3")),
        llm_max_retries=int(_env("LLM_MAX_RETRIES", "3")),
        llm_timeout_s=float(_env("LLM_TIMEOUT_S", "30")),
        llm_stream=_env_bool("LLM_STREAM", True),
        trace_stdout=_env_bool("TRACE_STDOUT", False),
        service_name=_env("SERVICE_NAME", "ic4-agent"),
    )
    values.update({k: v for k, v in overrides.items() if v is not None})
    return Config(**values)
