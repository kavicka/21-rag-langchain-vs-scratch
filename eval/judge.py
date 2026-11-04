"""LLM-as-judge (RAGAS answer correctness) vs step 15's rule-based scorer, on the same stored answers.

Question: if there were no exact gold values, could a small local model grade the answers instead?
The judge is the same Qwen3-1.7B (greedy, thinking off, up to 512 new tokens for RAGAS' JSON), reached
through ragas' LangchainLLMWrapper around the LangChain chat model from lc/llm.py.

RAGAS AnswerCorrectness with weights [1, 0]: the judge splits answer and reference into statements and
classifies them as TP / FP / FN; the score is their F1. Similarity (weight 0) is switched off so the
score depends on the judge only. A score ≥ 0.5 counts as "correct". Compared with the rule scorer:
agreement, Cohen's kappa, and how often the judge returned no score at all (a JSON it couldn't produce).

Run:  uv run --group judge python eval/judge.py --pipeline lc_default --balanced 24
"""

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.run_eval import RUNS, hardware, load_questions, merge_results  # noqa: E402
from rag.config import get_settings  # noqa: E402

THRESHOLD = 0.5


def reference(q: dict) -> str:
    """The gold value as a sentence, the form RAGAS expects for a reference answer."""
    g = q["gold"]
    match q["kind"]:
        case "phase":
            return f"The trial is {' / '.join(g)}."
        case "enrollment":
            return f"{g} participants were enrolled."
        case "min_age":
            return f"The minimum age for participants is {g} years."
        case "start_year":
            return f"The trial started in {g}."
        case "sponsor":
            return f"The lead sponsor is {g}."
        case "intervention":
            return f"The trial tested {', '.join(g)}."
        case "masking":
            return "The trial used no masking (open label)." if g == "NONE" else f"The masking was {g.lower()}."
    raise ValueError(q["kind"])


def kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def _ragas_shim() -> None:
    """ragas 0.4.3 imports langchain_community.chat_models.vertexai, removed in langchain-community 0.4.
    ragas only uses those classes in isinstance() checks, so empty stand-ins are enough to import it."""
    import types

    for mod, cls in [
        ("langchain_community.chat_models.vertexai", "ChatVertexAI"),
        ("langchain_community.llms", "VertexAI"),
    ]:
        try:
            m = __import__(mod, fromlist=[cls])
            getattr(m, cls)
        except (ImportError, AttributeError):
            m = sys.modules.get(mod) or types.ModuleType(mod)
            setattr(m, cls, type(cls, (), {}))
            sys.modules[mod] = m


def sensitivity(scored: list[dict]) -> dict:
    """Agreement at other cut-offs. Only ≥ 0.5 was fixed before the run; the others were added after seeing
    that the scores pile up exactly on 0.5 and 0.67, to show how much the verdict depends on the cut-off."""
    rule = [x["rule"] for x in scored]
    out = {}
    for name, f in [(">=0.5", lambda v: v >= 0.5), (">0.5", lambda v: v > 0.5), (">=0.75", lambda v: v >= 0.75)]:
        judge = [f(x["score"]) for x in scored]
        out[name] = {
            "agreement": round(sum(j == r for j, r in zip(judge, rule, strict=True)) / len(scored), 3),
            "kappa": kappa(judge, rule),
            "rule_wrong_judge_correct": sum(j and not r for j, r in zip(judge, rule, strict=True)),
            "rule_correct_judge_wrong": sum(r and not j for j, r in zip(judge, rule, strict=True)),
        }
    return out


def sync_only(chat):
    """ragas calls the model with agenerate(); ChatHuggingFace over a local pipeline raises NotImplementedError
    there. BaseChatModel's default async path runs the sync one in a thread, so a thin delegate is enough."""
    from langchain_core.language_models.chat_models import BaseChatModel

    class SyncOnly(BaseChatModel):
        inner: BaseChatModel

        @property
        def _llm_type(self) -> str:
            return "sync-only"

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            return self.inner._generate(messages, stop=stop, **kw)

    return SyncOnly(inner=chat)


def main() -> None:
    import os

    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")  # ragas sends usage analytics unless told not to
    _ragas_shim()
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerCorrectness

    from lc.llm import build_chat_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default="lc_default")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--balanced", type=int, default=None, help="n rule-correct + n rule-wrong answers (seed 0)")
    args = ap.parse_args()

    s = get_settings().model_copy(update={"llm_max_tokens": 512})
    qs = {q["id"]: q for q in load_questions()}
    recs = [json.loads(line) for line in (RUNS / f"generation_{args.pipeline}.jsonl").open()]
    recs = [r for r in recs if r["kind"] != "unanswerable"]
    if args.limit:
        recs = recs[:: max(1, len(recs) // args.limit)]
    if args.balanced:
        # equal numbers of rule-correct and rule-wrong answers, so agreement on both classes is measurable
        import random

        rng = random.Random(0)
        right, wrong = [r for r in recs if r["correct"]], [r for r in recs if not r["correct"]]
        recs = rng.sample(right, args.balanced) + rng.sample(wrong, min(args.balanced, len(wrong)))

    samples = [
        SingleTurnSample(user_input=qs[r["id"]]["question"], response=r["answer"], reference=reference(qs[r["id"]]))
        for r in recs
    ]
    metric = AnswerCorrectness(weights=[1.0, 0.0])
    t0 = time.perf_counter()
    result = evaluate(
        EvaluationDataset(samples=samples),
        metrics=[metric],
        llm=LangchainLLMWrapper(sync_only(build_chat_model(s))),
        embeddings=LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=s.embed_model)),
        run_config=RunConfig(max_workers=1, timeout=600, max_retries=2),
        raise_exceptions=False,
        show_progress=True,
    )
    seconds = time.perf_counter() - t0
    scores = [row["answer_correctness"] for row in result.to_pandas().to_dict("records")]

    rows = []
    for r, sc in zip(recs, scores, strict=True):
        ok = sc is not None and not math.isnan(sc)
        rows.append({"id": r["id"], "kind": r["kind"], "rule": r["correct"], "score": sc if ok else None})
    scored = [x for x in rows if x["score"] is not None]
    judge = [x["score"] >= THRESHOLD for x in scored]
    rule = [x["rule"] for x in scored]
    disagree = [
        {"id": x["id"], "kind": x["kind"], "rule": x["rule"], "score": round(x["score"], 3)}
        for x, j in zip(scored, judge, strict=True)
        if j != x["rule"]
    ]
    out = {
        "pipeline": args.pipeline,
        "sample": f"{args.balanced} rule-correct + {args.balanced} rule-wrong, seed 0" if args.balanced else "all",
        "n": len(rows),
        "no_score": len(rows) - len(scored),
        "agreement": round(sum(j == r for j, r in zip(judge, rule, strict=True)) / max(1, len(scored)), 3),
        "kappa": kappa(judge, rule),
        "rule_correct_judge_wrong": sum(r and not j for j, r in zip(judge, rule, strict=True)),
        "rule_wrong_judge_correct": sum(j and not r for j, r in zip(judge, rule, strict=True)),
        "rule_accuracy": round(sum(rule) / max(1, len(rule)), 3),
        "judge_accuracy": round(sum(judge) / max(1, len(judge)), 3),
        "seconds_per_answer": round(seconds / len(rows), 1),
        "judge": f"{s.llm_provider}:{s.llm_model}",
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "hardware": hardware(),
    }
    out["threshold_sensitivity"] = sensitivity(scored)
    (RUNS / f"judge_{args.pipeline}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    print(json.dumps(out, indent=1))
    print("disagreements:", disagree)
    merge_results(["judge"], {**out, "disagreements": disagree})


if __name__ == "__main__":
    main()
