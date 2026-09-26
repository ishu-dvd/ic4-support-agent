"""Status-based retry decisions (HTTP + model), streaming timing metrics, and cost attribution."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent.budget import Budget
from agent.config import Config
from agent.llm import Draft, LLMError, LLMUsage, OpenAICompatibleLLM, attribute_cost
from agent.upstream import HttpUpstream, RetryPolicy, UpstreamDegraded, UpstreamError, decide_retry

# ---- decide_retry: the decision table ---------------------------------------------------------------
P = RetryPolicy(max_retries=3)


@pytest.mark.parametrize(
    "attempt,kw,expected",
    [
        (0, dict(status=503), (True, "retry:status_503")),
        (0, dict(status=429), (True, "retry:status_429")),
        (0, dict(exc_kind="timeout"), (True, "retry:timeout")),
        (0, dict(exc_kind="unreachable"), (True, "retry:unreachable")),
        (2, dict(status=500), (True, "retry:status_500")),  # third failure -> fourth attempt allowed
        (3, dict(status=500), (False, "stop:max_retries")),  # default 3 retries = 4 attempts
        (0, dict(status=400), (False, "stop:status_400")),
        (0, dict(status=401), (False, "stop:status_401")),
        (0, dict(status=500, code="entitlement_service_error"), (False, "stop:code_entitlement_service_error")),
        (0, dict(status=503, remaining_ms=100, next_cost_ms=2500), (False, "stop:deadline")),
        (0, dict(status=503, remaining_ms=5000, next_cost_ms=2500), (True, "retry:status_503")),
    ],
)
def test_decide_retry_table(attempt, kw, expected):
    assert decide_retry(P, attempt, P.max_retries, **kw) == expected


def test_backoff_is_exponential_capped_and_jittered():
    p = RetryPolicy(backoff_base_s=0.1, backoff_max_s=0.35, jitter_s=0.05)
    b = [p.backoff_s(i) for i in range(4)]
    assert 0.1 <= b[0] <= 0.15 and 0.2 <= b[1] <= 0.25 and 0.35 <= b[2] <= 0.40 and 0.35 <= b[3] <= 0.40


# ---- HttpUpstream uses the policy ------------------------------------------------------------------
class _Tracer:
    def __init__(self):
        self.calls, self.events = [], []

    def call(self, name, **attrs):
        from contextlib import contextmanager

        @contextmanager
        def cm():
            span = dict(attrs)
            self.calls.append((name, span))
            yield span

        return cm()

    def step(self, name, **attrs):
        return self.call(name, **attrs)

    def event(self, name, **attrs):
        self.events.append((name, attrs))


FAST = RetryPolicy(max_retries=3, backoff_base_s=0.01, backoff_max_s=0.02, jitter_s=0.0)


def test_default_is_three_retries_with_events(fake_upstream):
    base, state = fake_upstream
    tr = _Tracer()
    up = HttpUpstream(base, 0.2, 0.2, tracer=tr, retry_policy=FAST)
    with pytest.raises(UpstreamDegraded):
        up.get("/slow")  # retries=None -> policy default 3
    assert state.calls["/slow"] == 4
    attempts = [s["attempt"] for n, s in tr.calls if n == "http GET /slow"]
    assert attempts == [0, 1, 2, 3]
    decisions = [s["retry_decision"] for n, s in tr.calls if n == "http GET /slow"]
    assert decisions == ["retry:timeout", "retry:timeout", "retry:timeout", "stop:max_retries"]
    names = [n for n, _ in tr.events]
    assert names == ["retry", "retry", "retry", "retries_exhausted"]
    assert tr.events[-1][1]["reason"] == "stop:max_retries" and tr.events[-1][1]["attempts"] == 4


def test_flaky_5xx_recovers_and_counts_one_retry(fake_upstream):
    base, state = fake_upstream
    tr = _Tracer()
    body = HttpUpstream(base, 0.5, 0.5, tracer=tr, retry_policy=FAST).get("/flaky")
    assert body["attempt"] == 2
    assert [n for n, _ in tr.events] == ["retry"]
    assert tr.events[0][1]["reason"] == "retry:status_500" and tr.events[0][1]["attempt"] == 1


def test_entitlement_error_still_never_retried_and_no_event(fake_upstream):
    base, state = fake_upstream
    tr = _Tracer()
    with pytest.raises(UpstreamDegraded) as ex:
        HttpUpstream(base, 0.5, 0.5, tracer=tr, retry_policy=FAST).get("/degraded")
    assert ex.value.code == "entitlement_service_error" and state.calls["/degraded"] == 1
    assert tr.events == []
    assert tr.calls[-1][1]["retry_decision"] == "stop:code_entitlement_service_error"


def test_retry_skipped_when_it_cannot_finish_before_deadline(fake_upstream):
    base, state = fake_upstream
    tr = _Tracer()
    up = HttpUpstream(base, 0.2, 0.2, tracer=tr, retry_policy=FAST, remaining_ms=lambda: 50)
    with pytest.raises(UpstreamDegraded):
        up.get("/slow")
    assert state.calls["/slow"] == 1  # a second 200 ms attempt cannot fit in 50 ms
    assert tr.calls[-1][1]["retry_decision"] == "stop:deadline"
    assert tr.events[-1][0] == "retries_exhausted" and tr.events[-1][1]["reason"] == "stop:deadline"


# ---- a tiny OpenAI-compatible server: streaming, 429s, bad JSON ---------------------------------------
def _sse(chunks, usage=None, model="openai-gpt-4o-mini"):
    out = []
    for c in chunks:
        out.append("data: " + json.dumps({"model": model, "choices": [{"delta": {"content": c}}]}) + "\n\n")
    if usage is not None:
        out.append("data: " + json.dumps({"model": model, "choices": [], "usage": usage}) + "\n\n")
    out.append("data: [DONE]\n\n")
    return "".join(out).encode()


class _State:
    def __init__(self):
        self.hits = {}
        self.lock = threading.Lock()
        self.behaviour = {}  # path -> list of scripted responses


def _handler(state: _State):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            with state.lock:
                k = state.hits[self.path] = state.hits.get(self.path, 0) + 1
                script = state.behaviour.get(self.path) or []
            step = script[min(k - 1, len(script) - 1)] if script else ("stream", None)
            kind, arg = step
            if kind == "status":
                payload = json.dumps({"error": {"message": "nope " + body.get("model", "")}}).encode()
                self.send_response(arg)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if kind == "reject_param":
                # DigitalOcean serverless: a 400 that names the offending *parameter* while we still send it
                # (kimi-k2.6: "temperature must be 1"; mistral: "chat_template is not supported ...").
                if arg in body:
                    msg = ("chat_template is not supported for Mistral tokenizers." if arg == "chat_template_kwargs"
                           else "%s must be 1 for this model" % arg)
                    payload = json.dumps({"error": {"message": msg, "type": "invalid_request_error"}}).encode()
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                kind = "stream"
            if kind == "sleep":
                time.sleep(arg)
            draft = {"category": "billing_proration", "diagnosis": "Prorated first invoice.",
                     "reply": "The first invoice after an upgrade is prorated. See kb-0001.", "kb_cited": ["kb-0001"],
                     "escalate_recommended": False}
            text = json.dumps(draft) if kind != "bad" else "not json at all"
            pieces = [text[i:i + 7] for i in range(0, len(text), 7)]
            if body.get("stream"):
                data = _sse(pieces, usage=None if kind == "nousage" else {"prompt_tokens": 120, "completion_tokens": 40})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                for piece in data.split(b"\n\n"):
                    if piece:
                        self.wfile.write(piece + b"\n\n")
                        self.wfile.flush()
                        time.sleep(0.004)
            else:
                data = json.dumps({"model": "openai-gpt-4o-mini", "choices": [{"message": {"content": text}}],
                                   "usage": {"prompt_tokens": 120, "completion_tokens": 40}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

    return H


@pytest.fixture(autouse=True)
def _forget_learned_params():
    """The per-model 'rejected parameter' memory is process-wide; tests must not leak it into each other."""
    from agent import llm as llm_mod

    llm_mod._UNSUPPORTED_PARAMS.clear()
    yield
    llm_mod._UNSUPPORTED_PARAMS.clear()


@pytest.fixture
def llm_server():
    state = _State()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _handler(state))
    srv.daemon_threads = True
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    t.start()
    try:
        yield "http://127.0.0.1:%d/v1" % srv.server_address[1], state
    finally:
        srv.shutdown()
        srv.server_close()


def _cfg(base, **kw):
    return Config(openai_base_url=base, openai_api_key="test-key", model_id="openai-gpt-4o-mini",
                  model_provider="openai_compatible", **kw)


PROMPT = {"system": "s" * 400, "user": "u" * 400, "category": "billing_proration"}


def test_streaming_measures_ttft_and_inter_token_gaps(llm_server):
    base, state = llm_server
    llm = OpenAICompatibleLLM(_cfg(base), retry_policy=FAST)
    draft, usage = llm.draft(PROMPT, Budget(20000, 2, 8))
    assert draft.source == "llm" and draft.kb_cited == ["kb-0001"]
    assert usage.model_id == "openai-gpt-4o-mini" and (usage.input_tokens, usage.output_tokens) == (120, 40)
    assert usage.cost_usd == pytest.approx(120 / 1000 * 0.00015 + 40 / 1000 * 0.0006, abs=1e-9)
    assert usage.ttft_ms is not None and usage.ttft_ms >= 0
    assert usage.stream_chunks > 5 and usage.tbt_ms_avg is not None and usage.tbt_ms_p95 >= usage.tbt_ms_p50
    assert usage.attempts == 1 and usage.retries == 0 and usage.usage_estimated is False
    assert usage.latency_ms >= usage.ttft_ms


def test_missing_usage_block_is_estimated_not_zero(llm_server):
    base, state = llm_server
    state.behaviour["/v1/chat/completions"] = [("nousage", None)]
    _, usage = OpenAICompatibleLLM(_cfg(base), retry_policy=FAST).draft(PROMPT, Budget(20000, 2, 8))
    assert usage.usage_estimated is True and usage.input_tokens > 0 and usage.output_tokens > 0
    assert usage.cost_usd is not None and usage.cost_usd > 0


def test_429_then_503_then_success_is_two_retries(llm_server):
    base, state = llm_server
    state.behaviour["/v1/chat/completions"] = [("status", 429), ("status", 503), ("stream", None)]
    tr = _Tracer()
    llm = OpenAICompatibleLLM(_cfg(base), retry_policy=FAST)
    llm.tracer = tr
    _, usage = llm.draft(PROMPT, Budget(20000, 2, 8))
    assert usage.attempts == 3 and usage.retries == 2
    assert [s["attempt"] for n, s in tr.calls] == [0, 1, 2]
    assert [s["retry_decision"] for n, s in tr.calls[:2]] == ["retry:status_429", "retry:status_503"]
    assert [(n, a["reason"]) for n, a in tr.events] == [("llm_retry", "retry:status_429"), ("llm_retry", "retry:status_503")]


def test_401_is_not_retried_and_message_stays_out_of_code(llm_server):
    base, state = llm_server
    state.behaviour["/v1/chat/completions"] = [("status", 401)]
    tr = _Tracer()
    llm = OpenAICompatibleLLM(_cfg(base), retry_policy=FAST)
    llm.tracer = tr
    with pytest.raises(LLMError) as ex:
        llm.draft(PROMPT, Budget(20000, 2, 8))
    assert ex.value.code == "http_401" and state.hits["/v1/chat/completions"] == 1
    assert tr.events == [("llm_retries_exhausted", {"attempts": 1, "reason": "stop:status_401", "code": "http_401"})]


def test_max_retries_exhausted_on_5xx(llm_server):
    base, state = llm_server
    state.behaviour["/v1/chat/completions"] = [("status", 500)]
    with pytest.raises(LLMError) as ex:
        OpenAICompatibleLLM(_cfg(base, llm_max_retries=2), retry_policy=RetryPolicy(2, backoff_base_s=0.01, jitter_s=0)).draft(
            PROMPT, Budget(20000, 2, 8))
    assert ex.value.code == "http_500" and state.hits["/v1/chat/completions"] == 3


def test_bad_output_retried_once_then_stops(llm_server):
    base, state = llm_server
    state.behaviour["/v1/chat/completions"] = [("bad", None)]
    tr = _Tracer()
    llm = OpenAICompatibleLLM(_cfg(base), retry_policy=FAST)
    llm.tracer = tr
    with pytest.raises(LLMError) as ex:
        llm.draft(PROMPT, Budget(20000, 2, 8))
    assert ex.value.code == "bad_output" and state.hits["/v1/chat/completions"] == 2
    assert tr.events == [("llm_retry", {"attempt": 1, "reason": "retry:bad_output", "code": "bad_output"})]


def test_non_streaming_path_still_works(llm_server):
    base, state = llm_server
    draft, usage = OpenAICompatibleLLM(_cfg(base, llm_stream=False), retry_policy=FAST).draft(PROMPT, Budget(20000, 2, 8))
    assert draft.source == "llm" and usage.stream_chunks == 0 and usage.ttft_ms is not None and usage.tbt_ms_avg is None


# ---- cost attribution ---------------------------------------------------------------------------------
def test_attribute_cost_splits_output_by_text_share():
    u = LLMUsage(1000, 1000, "openai-gpt-4o-mini", 0.00075)
    d = Draft("c", diagnosis="x" * 25, reply="y" * 75, kb_cited=[], escalate_recommended=False, source="llm")
    u = attribute_cost(u, d)
    assert u.cost_input_usd == pytest.approx(0.00015) and u.cost_output_usd == pytest.approx(0.0006)
    assert u.cost_diagnosis_usd == pytest.approx(0.00015) and u.cost_reply_usd == pytest.approx(0.00045)
    assert u.cost_diagnosis_usd + u.cost_reply_usd == pytest.approx(u.cost_output_usd)


def test_attribute_cost_unknown_model_is_none_everywhere():
    u = attribute_cost(LLMUsage(10, 10, "mystery", None), Draft("c", "d", "r"))
    assert u.cost_input_usd is None and u.cost_diagnosis_usd is None and u.cost_reply_usd is None


def test_rules_usage_is_zero_cost_everywhere():
    u = attribute_cost(LLMUsage(0, 0, "rules", 0.0), Draft("c", "d", "r"))
    assert (u.cost_input_usd, u.cost_output_usd, u.cost_diagnosis_usd, u.cost_reply_usd) == (0.0, 0.0, 0.0, 0.0)


def test_400_naming_a_parameter_drops_it_and_retries_once(llm_server):
    base, state = llm_server
    state.behaviour["/v1/chat/completions"] = [("reject_param", "temperature")]
    tr = _Tracer()
    llm = OpenAICompatibleLLM(_cfg(base), retry_policy=FAST)
    llm.tracer = tr
    draft, usage = llm.draft(PROMPT, Budget(20000, 2, 8))
    assert draft.source == "llm"
    assert usage.attempts == 2 and usage.retries == 1
    assert state.hits["/v1/chat/completions"] == 2
    assert tr.calls[0][1]["retry_decision"] == "retry:param_unsupported"
    assert tr.calls[0][1]["error"] == "param_unsupported:temperature"
    assert [e[0] for e in tr.events] == ["llm_param_dropped"] and tr.events[0][1]["param"] == "temperature"


def test_plain_400_is_still_never_retried(llm_server):
    base, state = llm_server
    state.behaviour["/v1/chat/completions"] = [("status", 400), ("stream", None)]
    tr = _Tracer()
    llm = OpenAICompatibleLLM(_cfg(base), retry_policy=FAST)
    llm.tracer = tr
    with pytest.raises(LLMError) as ex:
        llm.draft(PROMPT, Budget(20000, 2, 8))
    assert ex.value.code == "http_400" and state.hits["/v1/chat/completions"] == 1
    assert tr.calls[-1][1]["retry_decision"] == "stop:status_400"


def test_thinking_kwarg_is_sent_by_default_and_dropped_when_the_body_uses_an_alias(llm_server):
    base, state = llm_server
    state.behaviour["/v1/chat/completions"] = [("reject_param", "chat_template_kwargs")]
    tr = _Tracer()
    llm = OpenAICompatibleLLM(_cfg(base), retry_policy=FAST)
    llm.tracer = tr
    draft, usage = llm.draft(PROMPT, Budget(20000, 2, 8))
    assert draft.source == "llm" and usage.attempts == 2
    assert tr.events[0][1]["param"] == "chat_template_kwargs"


def test_thinking_kwarg_can_be_disabled(llm_server):
    base, state = llm_server
    state.behaviour["/v1/chat/completions"] = [("reject_param", "chat_template_kwargs")]
    llm = OpenAICompatibleLLM(_cfg(base, llm_disable_thinking=False), retry_policy=FAST)
    _, usage = llm.draft(PROMPT, Budget(20000, 2, 8))
    assert usage.attempts == 1  # parameter never sent, so the scripted rejection never fires


def test_rejected_parameter_is_remembered_for_the_next_call_to_that_model(llm_server):
    from agent import llm as llm_mod

    base, state = llm_server
    state.behaviour["/v1/chat/completions"] = [("reject_param", "temperature")]
    llm_mod._UNSUPPORTED_PARAMS.pop("openai-gpt-4o-mini", None)
    try:
        llm = OpenAICompatibleLLM(_cfg(base), retry_policy=FAST)
        _, first = llm.draft(PROMPT, Budget(20000, 2, 8))
        _, second = llm.draft(PROMPT, Budget(20000, 2, 8))
        assert first.attempts == 2 and second.attempts == 1
        assert llm_mod._UNSUPPORTED_PARAMS["openai-gpt-4o-mini"] == {"temperature"}
    finally:
        llm_mod._UNSUPPORTED_PARAMS.pop("openai-gpt-4o-mini", None)
