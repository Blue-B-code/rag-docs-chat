import { useState } from "react";
import { ingest, ask } from "./api.js";

export default function App() {
  const [tab, setTab] = useState("upload");
  const [files, setFiles] = useState([]);
  const [ingestState, setIngestState] = useState({ loading: false, result: null, error: null });
  const [question, setQuestion] = useState("");
  const [queryState, setQueryState] = useState({ loading: false, result: null, error: null });

  async function handleIngest(event) {
    event.preventDefault();
    if (files.length === 0) return;
    setIngestState({ loading: true, result: null, error: null });
    try {
      const result = await ingest(files);
      setIngestState({ loading: false, result, error: null });
      setFiles([]);
    } catch (err) {
      setIngestState({ loading: false, result: null, error: String(err) });
    }
  }

  async function handleAsk(event) {
    event.preventDefault();
    if (!question.trim()) return;
    setQueryState({ loading: true, result: null, error: null });
    try {
      const result = await ask(question);
      setQueryState({ loading: false, result, error: null });
    } catch (err) {
      setQueryState({ loading: false, result: null, error: String(err) });
    }
  }

  return (
    <main className="shell">
      <header className="hero">
        <h1>RAG Docs Chat</h1>
        <p>
          Upload documents (PDF / TXT / Markdown), then ask questions. Retrieval over local
          embeddings (Qdrant + fastembed) and answers from an OpenAI-compatible LLM.
        </p>
      </header>

      <nav className="tabs">
        <button className={tab === "upload" ? "active" : ""} onClick={() => setTab("upload")}>
          1 · Upload documents
        </button>
        <button className={tab === "ask" ? "active" : ""} onClick={() => setTab("ask")}>
          2 · Ask a question
        </button>
      </nav>

      {tab === "upload" && (
        <form className="card" onSubmit={handleIngest}>
          <input
            type="file"
            accept=".pdf,.txt,.md,.markdown"
            multiple
            onChange={(e) => setFiles([...e.target.files])}
          />
          <button type="submit" disabled={ingestState.loading || files.length === 0}>
            {ingestState.loading ? "Indexing…" : "Index documents"}
          </button>
          {ingestState.result && (
            <p className="ok">
              ✓ {ingestState.result.indexed_chunks} chunk(s) indexed from{" "}
              {ingestState.result.files.join(", ")}
            </p>
          )}
          {ingestState.error && <p className="err">{ingestState.error}</p>}
        </form>
      )}

      {tab === "ask" && (
        <form className="card" onSubmit={handleAsk}>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g. What is the company's refund policy?"
            rows={3}
          />
          <button type="submit" disabled={queryState.loading || !question.trim()}>
            {queryState.loading ? "Thinking…" : "Ask"}
          </button>

          {queryState.result && (
            <div className="answer">
              <h2>Answer</h2>
              <p>{queryState.result.answer}</p>
              {queryState.result.sources.length > 0 && (
                <details>
                  <summary>Sources ({queryState.result.sources.length})</summary>
                  <ul>
                    {queryState.result.sources.map((s, i) => (
                      <li key={i}>
                        <strong>{s.source}</strong> (score {s.score})
                        <p>{s.text}</p>
                      </li>
                    ))}
                  </ul>
                </details>
              )}
            </div>
          )}
          {queryState.error && <p className="err">{queryState.error}</p>}
        </form>
      )}
    </main>
  );
}
