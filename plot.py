"""Turn results.json into the PNGs in assets/. Measurement and presentation stay separate.

Colour encodes the family (scratch / LangChain / LangGraph), the row label names the variant.
One measure per panel, so no panel ever needs two y-scales.

Run:  uv run python plot.py
"""

import json

import matplotlib.pyplot as plt

from rag.config import ROOT

R = json.loads((ROOT / "results.json").read_text())
ASSETS = ROOT / "assets"
COLORS = {"scratch": "#2a78d6", "langchain": "#eb6834", "langgraph": "#6250d6"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e5e4df"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
    }
)


def family(name: str) -> str:
    return (
        "scratch"
        if name.startswith("scratch")
        else "langchain"
        if name.startswith("ablation")
        else "langgraph"
        if name.startswith("lg_")
        else "langchain"
    )


def bars(ax, labels: list[str], values: list[float], fams: list[str], title: str, fmt: str, xmax: float | None = None):
    y = range(len(labels))
    shown = [v if v is not None else 0 for v in values]  # None = not measured comparably: no bar, label "n/a"
    ax.barh(y, shown, height=0.6, color=[COLORS[f] for f in fams], edgecolor="#fcfcfb", linewidth=2)
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    top = xmax or max(shown) * 1.18
    ax.set_xlim(0, top)
    for i, v in enumerate(values):
        if v is None:
            ax.text(top * 0.01, i, "n/a", va="center", fontsize=9, color=MUTED)
            continue
        label = (f"{v:.1f}" if v < 10 else f"{v:.0f}") if fmt == "auto" else fmt.format(v)
        ax.text(v + top * 0.01, i, label, va="center", fontsize=9, color=INK)


def legend(fig, fams: list[str]) -> None:
    from matplotlib.patches import Patch

    names = {"scratch": "scratch (step 15)", "langchain": "LangChain", "langgraph": "LangGraph"}
    fig.legend(
        handles=[Patch(color=COLORS[f], label=names[f]) for f in fams],
        loc="lower center",
        ncol=len(fams),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )


RETRIEVAL_ROWS = [
    ("scratch:bm25", "scratch · BM25"),
    ("lc_tuned:bm25", "LC tuned · BM25 (step 15 tokenizer)"),
    ("ablation:rank_bm25_on_scratch_chunks", "rank_bm25 on scratch chunks + tokens"),
    ("ablation:rank_bm25_k1_1.2_on_scratch_chunks", "… and k1 = 1.2 (only idf differs)"),
    ("lc_tuned:bm25_ws", "LC tuned · BM25 (default tokenizer)"),
    ("scratch:dense", "scratch · dense"),
    ("lc_tuned:dense", "LC tuned · dense"),
    ("lc_default:dense", "LC default · dense"),
    ("lc_default:dense_instruction", "LC default · dense + bge instruction"),
    ("scratch:hybrid", "scratch · hybrid (RRF)"),
    ("lc_tuned:hybrid", "LC tuned · EnsembleRetriever"),
    ("scratch:rerank", "scratch · rerank"),
    ("lc_tuned:rerank", "LC tuned · rerank top 20"),
    ("lc_tuned:rerank_all", "LC tuned · rerank all fused"),
]


NOT_TIMED = {"ablation:rank_bm25_k1_1.2_on_scratch_chunks"}  # ran while the CPU was busy with the judge


def plot_retrieval() -> None:
    rows = [(k, lab) for k, lab in RETRIEVAL_ROWS if k in R["retrieval"]]
    labels, fams = [lab for _, lab in rows], [family(k) for k, _ in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), sharey=True, gridspec_kw={"width_ratios": [1, 1, 1]})
    for ax, (metric, title, fmt, xmax) in zip(
        axes,
        [
            ("evidence@1", "Answer chunk at rank 1", "{:.3f}", 1.12),
            ("evidence@5", "Answer chunk in top 5", "{:.3f}", 1.12),
            ("latency_ms_p50", "Retrieval latency p50 (ms, CPU)", "auto", None),
        ],
        strict=True,
    ):
        vals = [None if metric.startswith("latency") and k in NOT_TIMED else R["retrieval"][k][metric] for k, _ in rows]
        bars(ax, labels, vals, fams, title, fmt, xmax)
    legend(fig, ["scratch", "langchain"])
    fig.suptitle("Retrieval: 141 questions, same corpus, same models", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(ASSETS / "retrieval.png", dpi=150, bbox_inches="tight")


GEN_ROWS = [
    ("scratch", "scratch (step 15)"),
    ("lc_tuned", "LangChain tuned"),
    ("lc_default", "LangChain default"),
    ("lg_tuned", "LangGraph on tuned"),
    ("lg_default", "LangGraph on default"),
]


def plot_generation() -> None:
    rows = [(k, lab) for k, lab in GEN_ROWS if k in R.get("generation", {})]
    labels, fams = [lab for _, lab in rows], [family(k) for k, _ in rows]
    g = R["generation"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    bars(axes[0], labels, [g[k]["accuracy"] for k, _ in rows], fams, "Accuracy (126 answerable)", "{:.3f}", 1.12)
    bars(axes[1], labels, [g[k]["grounded"] for k, _ in rows], fams, "Correct and cites the gold trial", "{:.3f}", 1.12)
    bars(axes[2], labels, [g[k]["latency_ms_p50"] / 1000 for k, _ in rows], fams, "Latency p50 (s, CPU)", "{:.1f}")
    legend(fig, sorted({*fams}, key=list(COLORS).index))
    fig.suptitle("Answers: Qwen3-1.7B, greedy, 156 questions", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(ASSETS / "generation.png", dpi=150, bbox_inches="tight")


def plot_engineering() -> None:
    e = R["engineering"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.0))
    jobs = [j for j in e["loc"]["langchain"] if j != "total"]
    sides = ["scratch", "langchain"]
    labels = [f"{'scratch' if s == 'scratch' else 'LangChain'} · {j}" for j in jobs for s in sides if j in e["loc"][s]]
    vals = [e["loc"][s][j] for j in jobs for s in sides if j in e["loc"][s]]
    fams = [s for j in jobs for s in sides if j in e["loc"][s]]
    fams = ["langgraph" if "LangGraph" in lab else f for lab, f in zip(labels, fams, strict=True)]
    bars(axes[0], labels, vals, fams, "Logical lines of code", "{:.0f}")
    imp = e["import_ms"]
    bars(
        axes[1],
        list(imp),
        list(imp.values()),
        ["scratch", "langchain", "langgraph"],
        "Import time (ms, median of 7)",
        "{:.0f}",
    )
    o = e["overhead"]
    ov = {
        "scratch": o["rag_scratch_us"],
        "LCEL chain": o["rag_lcel_stream_us"],
        "LangGraph": o["rag_langgraph_stream_us"],
    }
    bars(
        axes[2],
        list(ov),
        [v / 1000 for v in ov.values()],
        ["scratch", "langchain", "langgraph"],
        "Orchestration per question (ms, fake model)",
        "{:.3f}",
    )
    legend(fig, list(COLORS))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(ASSETS / "engineering.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    plot_retrieval()
    if R.get("generation"):
        plot_generation()
    if R.get("engineering"):
        plot_engineering()
    print("plots written to", ASSETS)
