import { useEffect, useRef, useState } from "react";
import { ask, getHealth, type Done, type Health, type Hit, type Pipeline, type Step } from "./api";
import { AnswerView } from "./components/AnswerView";
import { SourcesPanel } from "./components/SourcesPanel";
import { StudyDrawer } from "./components/StudyDrawer";

const PIPELINES: { id: Pipeline; label: string; hint: string }[] = [
  { id: "scratch", label: "Scratch (step 15)", hint: "own BM25 + Qdrant, RRF, cross-encoder rerank" },
  { id: "lc_default", label: "LangChain default", hint: "tutorial defaults: 1000-char chunks, dense, k=4" },
  { id: "lc_tuned", label: "LangChain tuned", hint: "LangChain parts configured like step 15" },
  { id: "lg_tuned", label: "LangGraph (tuned)", hint: "grade → rewrite → retrieve again, on the tuned retriever" },
  { id: "lg_default", label: "LangGraph (default)", hint: "the same loop on the default retriever" },
];

const EXAMPLES = [
  'What is the minimum age for participants in the trial "A Study to Evaluate the Nipocalimab and Certolizumab Combination Therapy in Participants With Active Rheumatoid Arthritis"?',
  'How many participants were enrolled in the trial "Diabetes Prevention Program Outcomes Study"?',
  'What was the total budget of the trial "Diabetes Prevention Program Outcomes Study"?',
];

interface LaneState {
  answer: string;
  hits: Hit[];
  retrievalMs: number | null;
  steps: Step[];
  done: Done | null;
  error: string | null;
  busy: boolean;
}

const EMPTY: LaneState = { answer: "", hits: [], retrievalMs: null, steps: [], done: null, error: null, busy: false };

/** One pipeline's answer, LangGraph decisions and sources. Re-runs whenever `run` changes. */
function Lane({
  pipeline,
  setPipeline,
  question,
  run,
  onOpen,
}: {
  pipeline: Pipeline;
  setPipeline: (p: Pipeline) => void;
  question: string;
  run: number;
  onOpen: (id: string) => void;
}) {
  const [st, setSt] = useState<LaneState>(EMPTY);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (run === 0) return;
    abortRef.current?.abort(); // a new question cancels the previous stream
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    const patch = (p: Partial<LaneState> | ((s: LaneState) => Partial<LaneState>)) =>
      setSt((s) => ({ ...s, ...(typeof p === "function" ? p(s) : p) }));
    setSt({ ...EMPTY, busy: true });
    ask(
      question,
      pipeline,
      {
        onSources: (hits, ms) => patch({ hits, retrievalMs: ms }),
        onToken: (t) => patch((s) => ({ answer: s.answer + t })),
        onStep: (step) => patch((s) => ({ steps: [...s.steps, step] })),
        onDone: (done) => patch({ done }),
        onError: (error) => patch({ error }),
      },
      ctrl.signal,
    )
      .catch((err: Error) => {
        if (err.name !== "AbortError") patch({ error: err.message });
      })
      .finally(() => {
        if (!ctrl.signal.aborted) patch({ busy: false });
      });
    return () => ctrl.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run]);

  const meta = PIPELINES.find((p) => p.id === pipeline)!;
  return (
    <div className="lane">
      <label className="lane-head">
        <select value={pipeline} onChange={(e) => setPipeline(e.target.value as Pipeline)} aria-label="Pipeline">
          {PIPELINES.map((p) => (
            <option key={p.id} value={p.id}>
              {p.label}
            </option>
          ))}
        </select>
        <span className="muted small">{meta.hint}</span>
      </label>
      {st.steps.length > 0 && (
        <ol className="steps" aria-label="LangGraph steps">
          {st.steps.map((s, i) => (
            <li key={i} className={s.node === "grade" ? (s.relevant ? "ok" : "warn") : ""}>
              {s.node === "grade" && <>grade: sources {s.relevant ? "contain" : "do not contain"} the answer</>}
              {s.node === "rewrite" && <>rewrite → “{s.query}”</>}
              {s.node === "generate" && <>generate · {s.llm_calls} LLM calls in total</>}
            </li>
          ))}
        </ol>
      )}
      <AnswerView text={st.answer} done={st.done} error={st.error} busy={st.busy} onCite={onOpen} />
      <SourcesPanel hits={st.hits} retrievalMs={st.retrievalMs} cited={st.done?.citations ?? []} onOpen={onOpen} />
    </div>
  );
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [question, setQuestion] = useState(EXAMPLES[0]);
  const [left, setLeft] = useState<Pipeline>("scratch");
  const [right, setRight] = useState<Pipeline>("lc_default");
  const [run, setRun] = useState(0);
  const [openStudy, setOpenStudy] = useState<string | null>(null);

  useEffect(() => {
    getHealth().then(setHealth, (e: Error) => setHealthError(e.message));
  }, []);

  function submit(e?: React.FormEvent) {
    e?.preventDefault();
    if (question.trim().length >= 3) setRun((n) => n + 1);
  }

  return (
    <div className="app">
      <header>
        <h1>Clinical Trials RAG: scratch vs LangChain</h1>
        <p className="muted">
          {health
            ? `${health.n_studies} trials · ${health.llm} · the same question goes to both pipelines`
            : healthError
              ? `Backend not reachable: ${healthError}`
              : "Connecting…"}
        </p>
      </header>

      <form className="ask" onSubmit={submit}>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) submit(e);
          }}
          rows={2}
          maxLength={500}
          aria-label="Question"
        />
        <div className="controls">
          <button type="submit" className="primary" disabled={question.trim().length < 3}>
            Ask both
          </button>
        </div>
        <div className="examples">
          {EXAMPLES.map((q) => (
            <button type="button" key={q} onClick={() => setQuestion(q)}>
              {q}
            </button>
          ))}
        </div>
      </form>

      <main className="lanes">
        <Lane pipeline={left} setPipeline={setLeft} question={question} run={run} onOpen={setOpenStudy} />
        <Lane pipeline={right} setPipeline={setRight} question={question} run={run} onOpen={setOpenStudy} />
      </main>

      {openStudy && <StudyDrawer nctId={openStudy} onClose={() => setOpenStudy(null)} />}
    </div>
  );
}
