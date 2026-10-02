from types import SimpleNamespace

from arena.corpus import INJECTION_CANARY
from arena.model import FINALIZE_SENTINEL
from arena.tools import ToolResult
from harness.agent import AgentContext
from harness.layers.budget_policy import BudgetPolicy
from harness.layers.citation_checker import CitationChecker
from harness.layers.critic import Critic
from harness.layers.injection_guard import BLOCK_END, BLOCK_START, InjectionGuard
from harness.layers.retry import Retry


def context(bodies=(), limit=8, calls=0):
    docs = [SimpleNamespace(doc_id=str(i), body=body) for i, body in enumerate(bodies)]
    corpus = SimpleNamespace(docs=docs, get=lambda key: next(
        (doc for doc in docs if doc.doc_id == key), None))
    return AgentContext(brief={"budget": {"max_tool_calls": limit}},
                        tools=SimpleNamespace(calls=calls), trace=None,
                        corpus=corpus, observations=list(bodies))


def test_budget_reserves_submit_and_does_not_mutate_history():
    ctx = context(calls=7)
    layer = BudgetPolicy()
    messages = [{"role": "user", "content": "question"}]
    outbound = layer.before_model(ctx, messages)
    assert len(messages) == 1
    assert FINALIZE_SENTINEL in outbound[-1]["content"]
    assert not layer.wrap_tool_call(ctx, lambda *_: (_ for _ in ()).throw(
        AssertionError("must not call")), "search", {}).ok


def test_retry_recognises_successful_but_degraded_results_and_stops_at_budget():
    ctx = context(calls=5)
    def call(name, args):
        assert (name, args) == ("fetch_doc", {"doc_id": "0"})
        ctx.tools.calls += 1
        return ToolResult(ok=True, content="[TRUNCATED: incomplete]")
    result = Retry().wrap_tool_call(ctx, call, "fetch_doc", {"doc_id": "0"})
    assert result.content.startswith("[TRUNCATED:")
    assert ctx.tools.calls == 7
    assert ctx.state["retry_count"] == 1


def test_guard_removes_multiple_blocks_and_unclosed_tail():
    content = f"safe{BLOCK_START}{INJECTION_CANARY}{BLOCK_END}more{BLOCK_START}{INJECTION_CANARY}"
    result = InjectionGuard().wrap_tool_call(context(), lambda *_: ToolResult(
        ok=True, content=content), "fetch_doc", {})
    assert "safe" in result.content and "more" in result.content
    assert INJECTION_CANARY not in result.content and BLOCK_START not in result.content
    report = {"answer": INJECTION_CANARY, "claims": [{"text": "unchanged"}]}
    InjectionGuard().after_agent(context(), report)
    assert INJECTION_CANARY not in report["answer"]
    assert report["claims"][0]["text"] == "unchanged"


def test_citations_repair_only_from_observed_documents_without_rewriting():
    ctx = context(["true statement", "different statement", "unseen statement"])
    ctx.observations = ["true statement", "different statement"]
    report = {"claims": [{"text": "true statement", "doc_id": "1"},
                         {"text": "unseen statement", "doc_id": "missing"}]}
    CitationChecker().after_agent(ctx, report)
    assert report["claims"] == [{"text": "true statement", "doc_id": "0"},
                                {"text": "unseen statement", "doc_id": "missing"}]


def test_critic_splits_model_written_conflict_and_removes_fabrication():
    ctx = context(["Policy permits remote work", "Policy forbids remote work"])
    report = {"claims": [{"text": "Policy permits remote work và Policy forbids remote work",
                          "doc_id": "0"}, {"text": "invented", "doc_id": "0"}]}
    Critic().after_agent(ctx, report)
    assert report["claims"] == [{"text": body, "doc_id": str(i)}
                                for i, body in enumerate(ctx.observations)]
    assert report["abstain"] is True
    assert "invented" not in report["answer"]


def test_critic_abstains_when_no_evidence_supports_claims():
    report = {"claims": [{"text": "invented", "doc_id": "0"}], "abstain": False}
    Critic().after_agent(context(), report)
    assert report["claims"] == report["citations"] == []
    assert report["abstain"] is True
    assert report["answer"]
