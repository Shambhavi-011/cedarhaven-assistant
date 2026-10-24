/// <reference types="vite/client" />
import React, { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";

export interface CitedPage {
  doc_id: string;
  page_number: number;
}
interface Segment {
  text: string;
  highlight: boolean;
}
interface Evidence {
  doc_id: string;
  document_name: string;
  page_number: number;
  section: string;
  highlights: string[];
  segments: Segment[];
}
type AnswerStatus = "supported" | "partial" | "insufficient";
interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  question?: string;
  cited_pages?: CitedPage[];
  evidence?: Evidence[];
  answer_status?: AnswerStatus;
  status_note?: string;
}
interface Ticket {
  id: string;
  title: string;
  description: string;
  category: string;
  priority: string;
  priority_reason?: string;
  location: string;
  date_first_noticed?: string | null;
  contact?: string | null;
  status: string;
  created_at: string;
  source_pages: CitedPage[];
}
interface Draft extends Omit<Ticket, "id" | "status" | "created_at"> {
  missing: string[];
  possible_duplicate: { id: string; title: string } | null;
  emergency_notice: string | null;
}

const LANGUAGES = ["English", "French", "German", "Spanish", "Mandarin", "Hindi"];
const CATEGORIES = [
  "Waste & Recycling", "Roads & Potholes", "Streetlights & Signs", "Drainage & Water",
  "Parks & Trees", "Sidewalks & Winter", "By-law Complaint", "Permits & Licensing", "General Inquiry",
];
const PRIORITIES = ["Low", "Medium", "High", "Urgent"];
const SUGGESTIONS = [
  "Do I need a building permit for a 10 m² shed with electrical wiring?",
  "What hours can a contractor use a power saw on Saturday?",
  "My recycling was missed. When can I report it?",
  "How tall can a front-yard fence be?",
];
const STATUS_LABEL: Record<AnswerStatus, string> = {
  supported: "Supported by sources",
  partial: "Partially supported",
  insufficient: "Insufficient source evidence",
};

const API_BASE_URL =
  (import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8000";

// ---------- chat history (browser localStorage; not shared across devices) ----------
interface Conversation {
  id: string;
  title: string;
  updatedAt: number;
  messages: ChatMessage[];
}
const HISTORY_KEY = "ch_conversations_v1";
function loadConvs(): Conversation[] {
  try {
    const raw = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    return Array.isArray(raw) ? raw : [];
  } catch {
    return [];
  }
}
function saveConvs(list: Conversation[]) {
  for (const keep of [30, 10, 3]) {  // shrink if the browser storage quota is hit
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(list.slice(0, keep)));
      return;
    } catch { /* try fewer */ }
  }
}

// ---------- identity for tickets (no login in this project) ----------
let memoryToken = "";
function clientToken(): string {
  try {
    let t = localStorage.getItem("ch_client_token");
    if (!t) {
      t = Array.from(crypto.getRandomValues(new Uint8Array(24)), (b) => b.toString(16).padStart(2, "0")).join("");
      localStorage.setItem("ch_client_token", t);
    }
    return t;
  } catch {
    if (!memoryToken)
      memoryToken = Array.from(crypto.getRandomValues(new Uint8Array(24)), (b) => b.toString(16).padStart(2, "0")).join("");
    return memoryToken;
  }
}

class ApiError extends Error {
  status: number;
  detail: any;
  constructor(status: number, detail: any) {
    super(typeof detail === "string" ? detail : detail?.message || "Request failed.");
    this.status = status;
    this.detail = detail;
  }
}
async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", "X-Client-Token": clientToken(), ...(init.headers || {}) },
    });
  } catch {
    throw new ApiError(0, `Cannot reach the server at ${API_BASE_URL}. Is the backend running?`);
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new ApiError(res.status, body.detail ?? "Server error occurred.");
  return body as T;
}

// ---------- small answer renderer (paragraphs, lists, bold, citation chips) ----------
const SOURCES_HEADING = /^(sources|quellen|fuentes|来源|स्रोत)\s*:?\s*$/i;

function inline(text: string, key: string): React.ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*|\[CH-[A-Z]{3}-\d{2}[^\]]*\])/g).map((p, i) => {
    if (p.startsWith("**") && p.endsWith("**")) return <strong key={`${key}${i}`}>{p.slice(2, -2)}</strong>;
    if (p.startsWith("[CH-")) return <span key={`${key}${i}`} className="cite-chip">{p.slice(1, -1)}</span>;
    return <React.Fragment key={`${key}${i}`}>{p}</React.Fragment>;
  });
}

function AnswerBody({ text, hideSources }: { text: string; hideSources: boolean }) {
  let lines = text.split("\n");
  if (hideSources) {
    const idx = lines.findIndex((l) => SOURCES_HEADING.test(l.trim()));
    if (idx > 0) lines = lines.slice(0, idx); // sources are shown in the Evidence Explorer instead
  }
  const out: React.ReactNode[] = [];
  let list: string[] = [];
  const flush = () => {
    if (list.length) {
      const items = list;
      out.push(<ul key={`u${out.length}`}>{items.map((l, i) => <li key={i}>{inline(l, `l${i}`)}</li>)}</ul>);
      list = [];
    }
  };
  lines.forEach((raw, i) => {
    const l = raw.trim();
    if (/^[-•]\s+/.test(l)) list.push(l.replace(/^[-•]\s+/, ""));
    else {
      flush();
      if (l) out.push(<p key={`p${i}`}>{inline(l, `p${i}`)}</p>);
    }
  });
  flush();
  return <div className="answer-body">{out}</div>;
}

// ---------- Evidence Explorer ----------
function EvidenceExplorer({ msg }: { msg: ChatMessage }) {
  const [open, setOpen] = useState<number | null>(null);
  const ev = msg.evidence ?? [];
  const status = msg.answer_status;
  return (
    <section className="explorer" aria-label="Evidence Explorer">
      {status && (
        <div className={`status-row status-${status}`}>
          <span className="status-chip">{STATUS_LABEL[status]}</span>
          {msg.status_note && <span className="status-note">{msg.status_note}</span>}
        </div>
      )}
      {ev.length === 0 ? (
        <p className="explorer-empty">No supporting passage from the Cedarhaven documents was found for this answer.</p>
      ) : (
        <>
          <div className="explorer-title">Sources — select one to inspect the page text</div>
          <ul className="source-list">
            {ev.map((e, i) => {
              const isOpen = open === i;
              return (
                <li key={`${e.doc_id}-${e.page_number}`} className={isOpen ? "source open" : "source"}>
                  <button
                    type="button"
                    className="source-head"
                    aria-expanded={isOpen}
                    onClick={() => setOpen(isOpen ? null : i)}
                  >
                    <span className="source-id">{e.doc_id}</span>
                    <span className="source-meta">Page {e.page_number} · {e.section}</span>
                    <span className="source-doc">{e.document_name}</span>
                    <span className="chev" aria-hidden="true">{isOpen ? "–" : "+"}</span>
                  </button>
                  {isOpen && (
                    <div className="source-body">
                      <p className="source-hint">
                        {e.highlights.length
                          ? "Highlighted sentences are the page text most related to your question (matched by keywords). This is not a sentence-by-sentence check of the answer."
                          : "No sentence on this page matched your question closely; the full page text is shown."}
                      </p>
                      <div className="page-text">
                        {e.segments.map((s, k) => (s.highlight ? <mark key={k}>{s.text}</mark> : <span key={k}>{s.text}</span>))}
                      </div>
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        </>
      )}
    </section>
  );
}

// ---------- Ticket dialog ----------
function TicketDialog({
  draft, onClose, language,
}: { draft: Draft | null; onClose: () => void; language: string }) {
  const [form, setForm] = useState<Draft | null>(draft);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  const [dup, setDup] = useState<{ id: string; title: string } | null>(draft?.possible_duplicate ?? null);
  const [created, setCreated] = useState<Ticket | null>(null);
  const firstField = useRef<HTMLInputElement>(null);

  useEffect(() => { setForm(draft); setDup(draft?.possible_duplicate ?? null); }, [draft]);
  useEffect(() => { firstField.current?.focus(); }, [form !== null]);
  useEffect(() => {
    const k = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onClose]);

  const set = (k: keyof Draft, v: any) => setForm((f) => (f ? { ...f, [k]: v } : f));

  const submit = async (confirmDuplicate: boolean) => {
    if (!form) return;
    setSaving(true); setFailure(null); setErrors({});
    try {
      const t = await api<Ticket>("/api/tickets", {
        method: "POST",
        body: JSON.stringify({ ...form, confirm_duplicate: confirmDuplicate }),
      });
      setCreated(t);
    } catch (e: any) {
      if (e.status === 422 && e.detail?.errors) setErrors(e.detail.errors);
      else if (e.status === 409 && e.detail?.duplicate) setDup(e.detail.duplicate);
      else setFailure(e.message);
    } finally { setSaving(false); }
  };

  const field = (label: string, key: keyof Draft, props: any = {}) => (
    <label className="field">
      <span>{label}</span>
      <input
        ref={key === "title" ? firstField : undefined}
        value={(form?.[key] as string) ?? ""}
        onChange={(e) => set(key, e.target.value)}
        aria-invalid={!!errors[key as string]}
        {...props}
      />
      {errors[key as string] && <em className="field-error">{errors[key as string]}</em>}
    </label>
  );

  return (
    <div className="overlay" role="presentation" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="dialog" role="dialog" aria-modal="true" aria-labelledby="ticket-h">
        <div className="dialog-head">
          <h2 id="ticket-h">{created ? "Ticket created" : "Create support ticket"}</h2>
          <button className="icon-btn" aria-label="Close" onClick={onClose}>✕</button>
        </div>

        {created ? (
          <div className="dialog-body">
            <p className="ticket-id">{created.id}</p>
            <p>Saved to this application's database. Status: <strong>{created.status}</strong>.</p>
            <dl className="ticket-dl">
              <dt>Title</dt><dd>{created.title}</dd>
              <dt>Category</dt><dd>{created.category}</dd>
              <dt>Priority</dt><dd>{created.priority}</dd>
              <dt>Location</dt><dd>{created.location}</dd>
            </dl>
            <p className="fine-print">
              This is a training application. The ticket is stored locally and is not sent to any real
              municipality, email address or phone line.
            </p>
            <div className="dialog-actions"><button className="btn primary" onClick={onClose}>Done</button></div>
          </div>
        ) : !form ? (
          <div className="dialog-body"><p className="muted">Reading the conversation…</p></div>
        ) : (
          <form className="dialog-body" onSubmit={(e) => { e.preventDefault(); submit(false); }}>
            {form.emergency_notice && <div className="notice danger" role="alert">{form.emergency_notice}</div>}
            {form.missing.length > 0 && (
              <div className="notice warn">
                Missing detail: {form.missing.join(", ")}. Please fill it in; nothing has been guessed.
              </div>
            )}
            {dup && (
              <div className="notice warn" role="alert">
                You already created a similar ticket in the last 72 hours: <strong>{dup.id}</strong> ({dup.title}).
                Submitting again will create a separate ticket.
              </div>
            )}
            {field("Title", "title", { maxLength: 120 })}
            <label className="field">
              <span>Description</span>
              <textarea rows={4} value={form.description} maxLength={2000}
                onChange={(e) => set("description", e.target.value)} aria-invalid={!!errors.description} />
              {errors.description && <em className="field-error">{errors.description}</em>}
            </label>
            <div className="row">
              <label className="field">
                <span>Category</span>
                <select value={form.category} onChange={(e) => set("category", e.target.value)}>
                  {CATEGORIES.map((c) => <option key={c}>{c}</option>)}
                </select>
              </label>
              <label className="field">
                <span>Priority</span>
                <select value={form.priority} onChange={(e) => set("priority", e.target.value)}>
                  {PRIORITIES.map((c) => <option key={c}>{c}</option>)}
                </select>
              </label>
            </div>
            <p className="fine-print">Suggested because: {form.priority_reason}</p>
            {field("Location or nearest intersection (required)", "location", { maxLength: 200 })}
            <div className="row">
              {field("Date first noticed (optional)", "date_first_noticed", { type: "date" })}
              {field("Contact for updates (optional)", "contact", { maxLength: 120 })}
            </div>
            {form.source_pages.length > 0 && (
              <p className="fine-print">
                Related sources attached: {form.source_pages.map((p) => `${p.doc_id} p.${p.page_number}`).join(", ")}
              </p>
            )}
            {failure && <div className="error-banner" role="alert">{failure}</div>}
            <div className="dialog-actions">
              <button type="button" className="btn" onClick={onClose}>Cancel</button>
              {dup ? (
                <button type="button" className="btn primary" disabled={saving} onClick={() => submit(true)}>
                  {saving ? "Saving…" : "Create anyway"}
                </button>
              ) : (
                <button type="submit" className="btn primary" disabled={saving}>{saving ? "Saving…" : "Confirm and create"}</button>
              )}
            </div>
          </form>
        )}
      </div>
    </div>
  );
}

function TicketsPanel({ onClose }: { onClose: () => void }) {
  const [tickets, setTickets] = useState<Ticket[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api<{ tickets: Ticket[] }>("/api/tickets").then((r) => setTickets(r.tickets)).catch((e) => setErr(e.message));
  }, []);
  useEffect(() => {
    const k = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onClose]);
  return (
    <aside className="side-panel" aria-label="My tickets">
      <div className="dialog-head">
        <h2>My tickets</h2>
        <button className="icon-btn" aria-label="Close" onClick={onClose}>✕</button>
      </div>
      <div className="dialog-body">
        <p className="fine-print">Tickets created from this browser. They are tied to a private token stored here, not to an account.</p>
        {err && <div className="error-banner">{err}</div>}
        {!tickets && !err && <p className="muted">Loading…</p>}
        {tickets && tickets.length === 0 && <p className="muted">You have not created any tickets yet.</p>}
        <ul className="ticket-list">
          {tickets?.map((t) => (
            <li key={t.id}>
              <div className="ticket-top"><strong>{t.id}</strong><span className="pill">{t.status}</span></div>
              <div>{t.title}</div>
              <div className="muted small">{t.category} · {t.priority} · {t.location}</div>
              <div className="muted small">{new Date(t.created_at).toLocaleString()}</div>
            </li>
          ))}
        </ul>
      </div>
    </aside>
  );
}

// ---------- App ----------
export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [language, setLanguage] = useState("English");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState<string | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [ticketOpen, setTicketOpen] = useState(false);
  const [ticketError, setTicketError] = useState<string | null>(null);
  const [panelOpen, setPanelOpen] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);

  // ----- chat history (stored in this browser only) -----
  const [convs, setConvs] = useState<Conversation[]>(loadConvs);
  const [activeId, setActiveId] = useState<string | null>(null);
  const activeRef = useRef<string | null>(null);
  const [historyOpen, setHistoryOpen] = useState(
    typeof window !== "undefined" ? window.innerWidth > 800 : true
  );

  // Save the current conversation whenever its messages change.
  useEffect(() => {
    if (messages.length === 0) return;
    if (!activeRef.current) {
      activeRef.current = `c${Date.now()}`;
      setActiveId(activeRef.current);
    }
    const id = activeRef.current;
    const firstUser = messages.find((m) => m.role === "user")?.content ?? "New chat";
    const title = firstUser.length > 48 ? firstUser.slice(0, 48) + "…" : firstUser;
    setConvs((prev) => {
      const next = [{ id, title, updatedAt: Date.now(), messages }, ...prev.filter((c) => c.id !== id)];
      saveConvs(next);
      return next;
    });
  }, [messages]);

  const newChat = () => {
    if (loading) return;
    activeRef.current = null;
    setActiveId(null);
    setMessages([]);          // empty thread => welcome screen with the 4 suggested questions
    setInput("");
    setError(null);
    if (window.innerWidth <= 800) setHistoryOpen(false);
    taRef.current?.focus();
  };

  const openConversation = (c: Conversation) => {
    if (loading) return;
    activeRef.current = c.id;
    setActiveId(c.id);
    setMessages(c.messages);
    setError(null);
    if (window.innerWidth <= 800) setHistoryOpen(false);
  };

  const deleteConversation = (id: string) => {
    if (loading) return;
    setConvs((prev) => {
      const next = prev.filter((c) => c.id !== id);
      saveConvs(next);
      return next;
    });
    if (activeRef.current === id) newChat();
  };

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [messages, loading]);
  useEffect(() => {
    const ta = taRef.current;
    if (ta) { ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight, 160) + "px"; }
  }, [input]);

  const run = useCallback(async (question: string, addUser: boolean) => {
    setError(null);
    if (addUser) setMessages((p) => [...p, { id: `u${Date.now()}`, role: "user", content: question }]);
    setLoading(true);
    try {
      const data = await api<any>("/api/chat", { method: "POST", body: JSON.stringify({ question, language }) });
      setMessages((p) => [...p, {
        id: `a${Date.now()}`, role: "assistant", content: data.answer, question,
        cited_pages: data.cited_pages || [], evidence: data.evidence || [],
        answer_status: data.answer_status, status_note: data.status_note,
      }]);
    } catch (e: any) {
      setError(e.message || "Unable to reach the backend API.");
    } finally { setLoading(false); }
  }, [language]);

  const send = (q: string) => {
    const t = q.trim();
    if (!t || loading) return;
    setInput("");
    run(t, true);
  };

  const regenerate = () => {
    const last = messages[messages.length - 1];
    if (loading || !last || last.role !== "assistant" || !last.question) return;
    setMessages((p) => p.slice(0, -1));
    run(last.question, false);
  };

  const copy = async (m: ChatMessage) => {
    try { await navigator.clipboard.writeText(m.content); setCopied(m.id); setTimeout(() => setCopied(null), 1500); }
    catch { setError("Copy is not available in this browser context."); }
  };

  const openTicket = async (index: number) => {
    setTicketError(null); setDraft(null); setTicketOpen(true);
    const upTo = messages.slice(0, index + 1);
    const m = messages[index];
    try {
      const d = await api<Draft>("/api/tickets/draft", {
        method: "POST",
        body: JSON.stringify({
          messages: upTo.map((x) => ({ role: x.role, content: x.content })),
          cited_pages: m.cited_pages || [], language,
        }),
      });
      setDraft(d);
    } catch (e: any) { setTicketError(e.message); }
  };

  const onKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); send(input); }
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <button className="icon-btn menu-btn" aria-label="Toggle chat history" aria-expanded={historyOpen}
            onClick={() => setHistoryOpen((v) => !v)}>☰</button>
          <span className="logo" aria-hidden="true">C</span>
          <div>
            <h1>Cedarhaven Assistant</h1>
            <p>Answers from CH-SRM-01, CH-BYL-02 and CH-PER-03 only</p>
          </div>
        </div>
        <div className="top-actions">
          <button className="btn ghost" onClick={() => setPanelOpen((v) => !v)} aria-pressed={panelOpen}>My tickets</button>
          <select className="select" value={language} onChange={(e) => setLanguage(e.target.value)}
            disabled={loading} aria-label="Answer language">
            {LANGUAGES.map((l) => <option key={l}>{l}</option>)}
          </select>
        </div>
      </header>

      <div className="body">
        {historyOpen && (
          <nav className="history" aria-label="Chat history">
            <button className="btn primary new-chat" onClick={newChat} disabled={loading}>＋ New chat</button>
            <div className="history-title">Previous chats</div>
            {convs.length === 0 ? (
              <p className="fine-print">Your conversations will appear here.</p>
            ) : (
              <ul className="history-list">
                {convs.map((c) => (
                  <li key={c.id} className={c.id === activeId ? "history-item active" : "history-item"}>
                    <button className="history-open" onClick={() => openConversation(c)} disabled={loading}
                      aria-current={c.id === activeId}>
                      <span className="history-name">{c.title}</span>
                      <span className="history-date">{new Date(c.updatedAt).toLocaleString([], { dateStyle: "medium", timeStyle: "short" })}</span>
                    </button>
                    <button className="icon-btn" aria-label={`Delete chat: ${c.title}`} disabled={loading}
                      onClick={() => deleteConversation(c.id)}>🗑</button>
                  </li>
                ))}
              </ul>
            )}
            <p className="fine-print">History is saved only in this browser.</p>
          </nav>
        )}
        <main className="chat" aria-live="polite">
          <div className="thread">
            {messages.length === 0 && !loading && (
              <div className="welcome">
                <h2>How can I help with Cedarhaven services?</h2>
                <p>
                  Ask about 311 service requests, by-laws, permits and licensing. Every answer shows the
                  document and page it came from, and says so when the documents don't cover something.
                </p>
                <div className="suggest">
                  {SUGGESTIONS.map((s) => <button key={s} className="suggest-card" onClick={() => send(s)}>{s}</button>)}
                </div>
              </div>
            )}

            {messages.map((m, i) => (
              <article key={m.id} className={`msg ${m.role}`}>
                {m.role === "user" ? (
                  <div className="bubble">{m.content}</div>
                ) : (
                  <div className="reply">
                    <div className="reply-label">Cedarhaven Assistant</div>
                    <AnswerBody text={m.content} hideSources={(m.evidence?.length ?? 0) > 0} />
                    <EvidenceExplorer msg={m} />
                    <div className="actions">
                      <button className="link-btn" onClick={() => copy(m)}>{copied === m.id ? "Copied" : "Copy"}</button>
                      {i === messages.length - 1 && m.question && (
                        <button className="link-btn" onClick={regenerate} disabled={loading}>Regenerate</button>
                      )}
                      <button className="link-btn" onClick={() => openTicket(i)}>Create support ticket</button>
                    </div>
                  </div>
                )}
              </article>
            ))}

            {loading && (
              <div className="reply typing" role="status">
                <span className="dot" /><span className="dot" /><span className="dot" />
                <span className="sr-only">Searching the Cedarhaven documents…</span>
                <span className="typing-text">Searching the documents…</span>
              </div>
            )}
            {error && <div className="error-banner" role="alert">{error}</div>}
            <div ref={endRef} />
          </div>

          <form className="composer" onSubmit={(e) => { e.preventDefault(); send(input); }}>
            <textarea ref={taRef} rows={1} value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={onKey}
              placeholder="Ask about permits, by-laws, fees, noise hours…" aria-label="Your question"
              maxLength={1500} disabled={loading} />
            <button type="submit" className="send" disabled={loading || !input.trim()} aria-label="Send question">➤</button>
            <p className="hint">Enter to send · Shift+Enter for a new line</p>
          </form>
        </main>

        {panelOpen && <TicketsPanel onClose={() => setPanelOpen(false)} />}
      </div>

      {ticketOpen && (
        ticketError ? (
          <div className="overlay"><div className="dialog" role="alertdialog">
            <div className="dialog-body">
              <div className="error-banner">{ticketError}</div>
              <div className="dialog-actions"><button className="btn primary" onClick={() => setTicketOpen(false)}>Close</button></div>
            </div></div></div>
        ) : (
          <TicketDialog draft={draft} language={language}
            onClose={() => { setTicketOpen(false); setDraft(null); }} />
        )
      )}
    </div>
  );
}