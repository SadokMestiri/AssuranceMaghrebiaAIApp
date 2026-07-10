import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Sparkles, User, Download } from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { YEAR_MAX, YEAR_MIN } from "../contexts/FilterContext";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── localStorage schema ────────────────────────────────────────────────────────
// maghrebia_conv_index_v1  → [{ id, title, createdAt, lastUpdated, messageCount }]
// maghrebia_conv_{id}      → [...messages]
// maghrebia_current_conv   → id (string)
const INDEX_KEY   = "maghrebia_conv_index_v1";
const CURRENT_KEY = "maghrebia_current_conv";
const MAX_CONVS   = 20;

function lsGet(key, fallback = null) {
  try { return JSON.parse(localStorage.getItem(key)) ?? fallback; }
  catch { return fallback; }
}
function lsSet(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch {}
}
function lsDel(key) {
  try { localStorage.removeItem(key); } catch {}
}

function loadIndex()          { return lsGet(INDEX_KEY, []); }
function saveIndex(list)      { lsSet(INDEX_KEY, list.slice(0, MAX_CONVS)); }
function loadMessages(id)     { return lsGet(`maghrebia_conv_${id}`, null); }
function saveMessages(id, msgs) {
  // Never persist streaming placeholders
  lsSet(`maghrebia_conv_${id}`, msgs.filter((m) => !m.isStreaming));
}

function makeId() {
  return `conv-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function formatDate(ts) {
  const diff = Date.now() - ts;
  const d = Math.floor(diff / 86_400_000);
  if (d === 0) return "Aujourd'hui";
  if (d === 1) return "Hier";
  if (d < 7)  return `Il y a ${d} jours`;
  return new Date(ts).toLocaleDateString("fr-FR", { day: "numeric", month: "short" });
}

// ── Sub-components (unchanged) ─────────────────────────────────────────────────

const WELCOME = {
  role: "assistant",
  content: "Bonjour. Je suis l'agent IA métier. Posez une question KPI, risque impayé, forecast, drift ou segmentation.",
  tools: [], charts: [], tables: [], toolResults: [],
  llmUsed: null, synthesisMode: null, policyApplied: null, specialistReports: [],
};

function formatCellValue(value) {
  if (value === null || value === undefined) return "-";
  if (typeof value === "number") {
    if (Number.isInteger(value)) return value.toLocaleString("fr-TN");
    return value.toLocaleString("fr-TN", { maximumFractionDigits: 3 });
  }
  return String(value);
}

function MessageContent({ content }) {
  return (
    <div className="agent-message-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p:  ({ children }) => <p  className="agent-message-line">{children}</p>,
          ul: ({ children }) => <ul className="agent-message-list">{children}</ul>,
          ol: ({ children }) => <ol className="agent-message-list">{children}</ol>,
          li: ({ children }) => <li>{children}</li>,
        }}
      >
        {String(content || "")}
      </ReactMarkdown>
    </div>
  );
}

// Excel opens UTF-8 CSV natively (with a BOM for accented characters), so
// this covers "export as Excel" without pulling in a new xlsx dependency.
function downloadCsv(filename, columns, rows) {
  const escapeCell = (value) => {
    const str = value === null || value === undefined ? "" : String(value);
    return /[",\n;]/.test(str) ? `"${str.replace(/"/g, '""')}"` : str;
  };
  const header = columns.map(escapeCell).join(",");
  const body = rows.map((row) => columns.map((c) => escapeCell(row?.[c])).join(",")).join("\n");
  const csvContent = "﻿" + [header, body].filter(Boolean).join("\n");
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename.toLowerCase().endsWith(".csv") ? filename : `${filename}.csv`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

function ExportButton({ filename, columns, rows }) {
  if (!rows?.length || !columns?.length) return null;
  return (
    <button
      type="button"
      className="agent-export-btn"
      title="Exporter en CSV (compatible Excel)"
      onClick={() => downloadCsv(filename, columns, rows)}
    >
      <Download size={12} /> Exporter
    </button>
  );
}

function ToolTable({ table, index }) {
  const rows    = Array.isArray(table?.rows)    ? table.rows    : [];
  const columns = Array.isArray(table?.columns) ? table.columns
    : rows.length > 0 ? Object.keys(rows[0]) : [];
  if (rows.length === 0 || columns.length === 0) return null;
  return (
    <article className="agent-structured-card" key={`table-${index}`}>
      <h5>
        {table?.title || `Table ${index + 1}`}
        <ExportButton filename={table?.title || `table-${index + 1}`} columns={columns} rows={rows} />
      </h5>
      <div className="agent-mini-table-wrapper">
        <table className="agent-mini-table">
          <thead><tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
          <tbody>
            {rows.slice(0, 8).map((row, ri) => (
              <tr key={`row-${ri}`}>
                {columns.map((c) => <td key={`${ri}-${c}`}>{formatCellValue(row?.[c])}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
        {rows.length > 8 && (
          <p className="agent-table-truncated-note">
            Affichage de 8 sur {rows.length} lignes — export CSV pour tout voir.
          </p>
        )}
      </div>
    </article>
  );
}

// Fixed categorical order (never cycled past this set) — reused app-wide
// so a given hue reads consistently across dashboard tabs and agent charts.
const CATEGORICAL_PALETTE = ["#004A8D", "#F38F1D", "#25C6FF", "#BE123C", "#2E7D32", "#6A1B9A", "#00838F", "#E91E8C"];

// Part-to-whole charts blur past ~7 segments — fold the tail into "Autres"
// rather than rendering a rainbow of indistinguishable slivers.
function capPieSegments(items, xKey, yKey, maxSegments = 7) {
  if (items.length <= maxSegments) return items;
  const sorted = [...items].sort((a, b) => Number(b?.[yKey] || 0) - Number(a?.[yKey] || 0));
  const head = sorted.slice(0, maxSegments - 1);
  const tail = sorted.slice(maxSegments - 1);
  const otherTotal = tail.reduce((sum, row) => sum + Number(row?.[yKey] || 0), 0);
  return [...head, { [xKey]: "Autres", [yKey]: otherTotal }];
}

// Turns a clicked chart category into a natural-language follow-up question,
// using the chart's own title as the only signal for which dimension was
// clicked (no per-chart-type wiring needed — new chart titles degrade to the
// generic case instead of doing nothing).
function composeDrillDownQuestion(chart, clickedLabel) {
  const label = String(clickedLabel ?? "").trim();
  if (!label) return null;
  const title = String(chart?.title || "").toLowerCase();
  if (title.includes("gouvernorat") || title.includes("ville") || title.includes("localite")) {
    return `Top clients a ${label} par impaye`;
  }
  if (title.includes("branche")) {
    return `Analyse detaillee de la branche ${label}`;
  }
  if (title.includes("agent")) {
    return `Details sur l'agent ${label}`;
  }
  if (title.includes("produit")) {
    return `Details sur le produit ${label}`;
  }
  if (title.includes("marque")) {
    return `Details sur les vehicules de marque ${label}`;
  }
  if (title.includes("sinistre")) {
    return `Details sur les sinistres ${label}`;
  }
  return `Plus de details sur ${label} (${chart?.title || "graphique"})`;
}

function ToolChart({ chart, index, onDrillDown }) {
  const items = Array.isArray(chart?.items) ? chart.items : [];
  if (items.length === 0) return null;
  const keys  = Object.keys(items[0] || {});
  const xKey  = chart?.x_key || keys[0];
  const yKey  = chart?.y_key || keys[1];
  if (!xKey || !yKey) return null;
  const type = String(chart?.type || "bar").toLowerCase();
  const isPie  = type === "pie" || type === "donut";
  const isArea = type === "area";
  const isLine = type === "line";
  const palette = CATEGORICAL_PALETTE;
  const declaredSeries = Array.isArray(chart?.series) ? chart.series.filter(Boolean) : [];
  const chartSeries = declaredSeries.length > 0 ? declaredSeries
    : [{ key: yKey, label: yKey, color: isLine || isArea ? "#004A8D" : "#F38F1D" }];
  const fmt = (v, n) => [formatCellValue(v), String(n || "Valeur")];
  const pieItems = isPie ? capPieSegments(items, xKey, yKey) : items;
  const drillDownEnabled = typeof onDrillDown === "function" && (isPie || (!isLine && !isArea));
  const handleSliceClick = (data) => {
    if (!drillDownEnabled) return;
    const label = data?.[xKey] ?? data?.name ?? data?.payload?.[xKey];
    const followUp = composeDrillDownQuestion(chart, label);
    if (followUp) onDrillDown(followUp);
  };
  return (
    <article className="agent-structured-card" key={`chart-${index}`}>
      <h5>
        <span>
          {chart?.title || `Chart ${index + 1}`}
          {drillDownEnabled && <span className="agent-chart-drilldown-hint">cliquez pour explorer</span>}
        </span>
        <ExportButton filename={chart?.title || `chart-${index + 1}`} columns={[xKey, yKey]} rows={items} />
      </h5>
      <div className="agent-chart-box">
        <ResponsiveContainer width="100%" height={220}>
          {isPie ? (
            <PieChart margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
              <Pie
                data={pieItems}
                dataKey={yKey}
                nameKey={xKey}
                cx="50%"
                cy="50%"
                outerRadius={78}
                innerRadius={type === "donut" ? 46 : 0}
                paddingAngle={2}
                label={({ percent }) => (percent >= 0.06 ? `${(percent * 100).toFixed(0)}%` : "")}
                labelLine={false}
                onClick={drillDownEnabled ? handleSliceClick : undefined}
                cursor={drillDownEnabled ? "pointer" : "default"}
              >
                {pieItems.map((_, i) => (
                  <Cell key={`slice-${i}`} fill={palette[i % palette.length]} />
                ))}
              </Pie>
              <Tooltip formatter={fmt} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
            </PieChart>
          ) : isArea ? (
            <AreaChart data={items} margin={{ top: 8, right: 12, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#dbe7f3" />
              <XAxis dataKey={xKey} tick={{ fontSize: 12 }} stroke="#64748b" />
              <YAxis tick={{ fontSize: 12 }} stroke="#64748b" />
              <Tooltip formatter={fmt} />
              {chartSeries.length > 1 && <Legend />}
              {chartSeries.map((s, i) => (
                <Area key={`area-${i}`} type="monotone" dataKey={s?.key || yKey}
                  name={s?.label || s?.key || yKey}
                  stroke={s?.color || palette[i % palette.length]}
                  fill={s?.color || palette[i % palette.length]} fillOpacity={0.16}
                  strokeWidth={2.2} />
              ))}
            </AreaChart>
          ) : isLine ? (
            <LineChart data={items} margin={{ top: 8, right: 12, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#dbe7f3" />
              <XAxis dataKey={xKey} tick={{ fontSize: 12 }} stroke="#64748b" />
              <YAxis tick={{ fontSize: 12 }} stroke="#64748b" />
              <Tooltip formatter={fmt} />
              {chart?.forecast_start_period && (
                <ReferenceLine x={chart.forecast_start_period} stroke="#dc2626" strokeDasharray="4 4"
                  label={{ value: "Début prévision", position: "insideTopRight", fill: "#dc2626", fontSize: 11 }} />
              )}
              {chartSeries.length > 1 && <Legend />}
              {chartSeries.map((s, i) => (
                <Line key={`line-${i}`} type="monotone" dataKey={s?.key || yKey}
                  name={s?.label || s?.key || yKey}
                  stroke={s?.color || palette[i % palette.length]}
                  strokeWidth={s?.strokeWidth || 2.2} dot={s?.dot ?? false}
                  connectNulls={false} strokeDasharray={s?.strokeDasharray} />
              ))}
            </LineChart>
          ) : (
            <BarChart data={items} margin={{ top: 8, right: 12, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#dbe7f3" />
              <XAxis dataKey={xKey} tick={{ fontSize: 12 }} stroke="#64748b" />
              <YAxis tick={{ fontSize: 12 }} stroke="#64748b" />
              <Tooltip formatter={fmt} />
              <Bar dataKey={chartSeries[0]?.key || yKey} fill={chartSeries[0]?.color || "#004A8D"} radius={[8, 8, 0, 0]}
                onClick={drillDownEnabled ? handleSliceClick : undefined}
                cursor={drillDownEnabled ? "pointer" : "default"} />
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>
    </article>
  );
}

function ToolKpi({ kpi, index }) {
  const unit   = String(kpi?.unit || "").toUpperCase();
  const suffix = unit === "TND" ? " TND" : unit === "%" ? " %" : "";
  return (
    <article className="agent-structured-card agent-kpi-card" key={`kpi-${index}`}>
      <h5>{kpi?.label || `KPI ${index + 1}`}</h5>
      <p className="agent-kpi-value">{`${formatCellValue(kpi?.value)}${suffix}`}</p>
    </article>
  );
}

function extractAlerts(toolResults) {
  const results = Array.isArray(toolResults) ? toolResults : [];
  return results.flatMap((r) => Array.isArray(r?.payload?.alerts) ? r.payload.alerts : []).slice(0, 10);
}

function extractKpis(toolResults) {
  const results = Array.isArray(toolResults) ? toolResults : [];
  const seen = new Set();
  return results
    .flatMap((r) => Array.isArray(r?.payload?.kpis) ? r.payload.kpis : [])
    .filter((kpi, i) => {
      if (typeof kpi?.value !== "number") return false;
      const key = String(kpi?.key || `kpi-${i}`);
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    })
    .map((kpi, i) => ({ key: String(kpi?.key || `kpi-${i}`), label: kpi?.label || kpi?.key, value: kpi.value, unit: kpi?.unit || "" }))
    .slice(0, 6);
}

// ── Main component ─────────────────────────────────────────────────────────────

export default function AgentChat({ filters, recommendedPrompts = [], enableAnalyticFilters = true }) {
  const [conversations, setConversations] = useState([]);  // index entries
  const [sessionId,     setSessionId]     = useState(null);
  const [messages,      setMessages]      = useState([WELCOME]);
  const [question,      setQuestion]      = useState("");
  const [isLoading,     setIsLoading]     = useState(false);
  const [ready,         setReady]         = useState(false);
  const bottomRef = useRef(null);
  const titleRequestedRef = useRef(new Set());

  // ── Init: load index + restore last active conversation ────────────────────
  useEffect(() => {
    const index   = loadIndex();
    const current = lsGet(CURRENT_KEY, null);

    setConversations(index);

    if (current && index.some((c) => c.id === current)) {
      const msgs = loadMessages(current);
      if (msgs && msgs.length > 0) setMessages(msgs);
      setSessionId(current);
    } else {
      const newId = makeId();
      setSessionId(newId);
      lsSet(CURRENT_KEY, newId);
    }
    setReady(true);
  }, []);

  // ── Auto-save: whenever messages settle (no streaming) ────────────────────
  useEffect(() => {
    if (!sessionId || !ready) return;
    if (messages.some((m) => m.isStreaming)) return;

    const userMsgs = messages.filter((m) => m.role === "user");
    if (userMsgs.length === 0) return; // nothing to save yet

    const fallbackTitle = (() => {
      const raw = userMsgs[0].content || "";
      return raw.length > 55 ? raw.slice(0, 52) + "…" : raw;
    })();

    setConversations((prev) => {
      const exists = prev.some((c) => c.id === sessionId);
      const updated = exists
        ? prev.map((c) => c.id === sessionId
            ? { ...c, lastUpdated: Date.now(), messageCount: userMsgs.length }
            : c)
        : [{ id: sessionId, title: fallbackTitle, createdAt: Date.now(), lastUpdated: Date.now(), messageCount: 1 }, ...prev]
            .slice(0, MAX_CONVS);
      saveIndex(updated);
      return updated;
    });

    saveMessages(sessionId, messages);

    // Upgrade the naive truncated title to a short generated one — once per
    // conversation, right after its first full exchange completes. The
    // fallback above already shows instantly; this just swaps it in later.
    const firstAssistant = messages.find((m) => m.role === "assistant" && !m.isStreaming);
    if (userMsgs.length === 1 && firstAssistant && !titleRequestedRef.current.has(sessionId)) {
      titleRequestedRef.current.add(sessionId);
      fetch(`${API_BASE}/agent/title`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userMsgs[0].content || "", answer: firstAssistant.content || "" }),
      })
        .then((res) => (res.ok ? res.json() : null))
        .then((data) => {
          if (!data?.title) return;
          setConversations((prev) => {
            const updated = prev.map((c) => (c.id === sessionId ? { ...c, title: data.title } : c));
            saveIndex(updated);
            return updated;
          });
        })
        .catch(() => {});
    }
  }, [messages, sessionId, ready]);

  // ── Auto-scroll ────────────────────────────────────────────────────────────
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ── Conversation actions ───────────────────────────────────────────────────
  const selectConversation = (id) => {
    if (id === sessionId || isLoading) return;
    const msgs = loadMessages(id);
    setSessionId(id);
    setMessages(msgs && msgs.length > 0 ? msgs : [WELCOME]);
    setQuestion("");
    lsSet(CURRENT_KEY, id);
  };

  const startNewConversation = () => {
    const newId = makeId();
    setSessionId(newId);
    setMessages([WELCOME]);
    setQuestion("");
    lsSet(CURRENT_KEY, newId);
  };

  const deleteConversation = (id, e) => {
    e.stopPropagation();
    lsDel(`maghrebia_conv_${id}`);
    setConversations((prev) => {
      const updated = prev.filter((c) => c.id !== id);
      saveIndex(updated);
      return updated;
    });
    if (id === sessionId) startNewConversation();
  };

  // ── Filter payload ─────────────────────────────────────────────────────────
  const payloadTemplate = useMemo(() => {
    const base = { top_k: 3, horizon_months: 3 };
    if (!enableAnalyticFilters || !filters) return base;
    const yf = Math.min(YEAR_MAX, Math.max(YEAR_MIN, Math.trunc(Number(filters.yearFrom) || YEAR_MIN)));
    const yt = Math.min(YEAR_MAX, Math.max(YEAR_MIN, Math.trunc(Number(filters.yearTo)   || YEAR_MAX)));
    return {
      ...base,
      branch:      filters.branch === "ALL" ? null : filters.branch,
      year_from:   Math.min(yf, yt),
      year_to:     Math.max(yf, yt),
      month:       filters.month,
      gouvernorat: filters.gouvernorat,
    };
  }, [enableAnalyticFilters, filters]);

  // ── Submit ─────────────────────────────────────────────────────────────────
  const submitQuestion = async (rawQuestion) => {
    const trimmed = rawQuestion.trim();
    if (!trimmed || isLoading || !ready || !sessionId) return;

    setMessages((prev) => [...prev, { role: "user", content: trimmed, tools: [], charts: [], tables: [], toolResults: [] }]);
    setQuestion("");
    setIsLoading(true);

    const msgId = `stream-${Date.now()}`;
    setMessages((prev) => [...prev, {
      id: msgId, role: "assistant", isStreaming: true,
      streamProgress: [], streamingTokens: "", content: "",
      tools: [], charts: [], tables: [], toolResults: [],
      llmUsed: null, synthesisMode: null, policyApplied: null, specialistReports: [],
    }]);

    const patch = (updater) =>
      setMessages((prev) => prev.map((m) => m.id === msgId ? { ...m, ...updater(m) } : m));

    try {
      const res = await fetch(`${API_BASE}/api/v1/agent/query/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed, session_id: sessionId, ...payloadTemplate }),
      });
      if (!res.ok) throw new Error(`Agent API error ${res.status}`);

      const reader = res.body.getReader();
      const dec    = new TextDecoder();
      let buf = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop();
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6).trim();
          if (raw === "[DONE]") break;
          let ev;
          try { ev = JSON.parse(raw); } catch { continue; }

          if (ev.type === "progress") {
            patch((m) => ({ streamProgress: [...(m.streamProgress || []), ev.label] }));
          } else if (ev.type === "llm_token") {
            patch((m) => ({ streamingTokens: (m.streamingTokens || "") + ev.token }));
          } else if (ev.type === "result") {
            const a = ev.data || {};
            patch(() => ({
              isStreaming: false, streamProgress: [], streamingTokens: "",
              content:          a.answer          || "Aucune réponse retournée.",
              tools:            a.invoked_tools   || [],
              charts:           a.charts          || [],
              tables:           a.tables          || [],
              toolResults:      a.tool_results    || [],
              status:           a.status          || "ok",
              llmUsed:          Boolean(a.llm_used),
              synthesisMode:    a.synthesis_mode  || (a.llm_used ? "llm" : "deterministic"),
              policyApplied:    a.policy_applied  || null,
              specialistReports: a.specialist_reports || [],
            }));
          } else if (ev.type === "error") {
            patch(() => ({ isStreaming: false, content: `Erreur agent : ${ev.detail || "Erreur inconnue"}`, status: "error" }));
          }
        }
      }
    } catch (err) {
      patch(() => ({ isStreaming: false, content: `Erreur agent : ${String(err.message || err)}`, status: "error" }));
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      void submitQuestion(question);
    }
  };

  // ── Conversation list (sorted newest first) ────────────────────────────────
  const sortedConvs = [...conversations].sort((a, b) => b.lastUpdated - a.lastUpdated);

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="agent-workspace">

      {/* ── History sidebar ─────────────────────────────────────── */}
      <aside className="agent-history-panel">
        <p className="agent-history-label">Conversations</p>

        <button className="agent-new-conv-btn" onClick={startNewConversation} disabled={isLoading}>
          <span aria-hidden>+</span> Nouvelle conversation
        </button>

        <div className="agent-history-list">
          {sortedConvs.length === 0 ? (
            <p className="agent-history-empty">Aucun historique</p>
          ) : (
            sortedConvs.map((conv) => (
              <button
                key={conv.id}
                className={`agent-history-item${conv.id === sessionId ? " agent-history-active" : ""}`}
                onClick={() => selectConversation(conv.id)}
                disabled={isLoading}
              >
                <span className="agent-history-title">{conv.title || "Sans titre"}</span>
                <span className="agent-history-meta">
                  {formatDate(conv.lastUpdated)} · {conv.messageCount} msg
                </span>
                <span
                  role="button"
                  className="agent-history-delete"
                  onClick={(e) => deleteConversation(conv.id, e)}
                  title="Supprimer cette conversation"
                >
                  ×
                </span>
              </button>
            ))
          )}
        </div>
      </aside>

      {/* ── Chat panel ──────────────────────────────────────────── */}
      <section className="panel agent-panel">
        <div className="agent-headline">
          <h3>Agent IA CEO</h3>
          <span className="agent-status">{isLoading ? "Analyse..." : "Prêt"}</span>
        </div>

        {recommendedPrompts.length > 0 && (
          <div className="agent-prompt-box">
            <p className="agent-prompt-title">Prompts recommandés</p>
            <div className="agent-prompt-list">
              {recommendedPrompts.map((p) => (
                <button key={p} type="button" className="agent-prompt-btn"
                  onClick={() => void submitQuestion(p)} disabled={isLoading}>
                  {p}
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="agent-messages">
          {messages.map((msg, idx) => {
            const alerts = extractAlerts(msg.toolResults);
            const kpis   = extractKpis(msg.toolResults);
            return (
              <article
                key={msg.id || `${msg.role}-${idx}`}
                className={`agent-message ${msg.role === "user" ? "agent-user" : "agent-assistant"}`}
              >
                <div className="agent-message-avatar-row">
                  <span className={`agent-message-avatar ${msg.role === "user" ? "avatar-user" : "avatar-ai"}`}>
                    {msg.role === "user" ? <User size={13} /> : <Sparkles size={13} />}
                  </span>
                  <span className="agent-message-sender">{msg.role === "user" ? "Vous" : "Assistant IA"}</span>
                </div>

                {msg.isStreaming ? (
                  <div className="agent-stream-zone">
                    {(msg.streamProgress || []).map((label, i) => (
                      <div key={i} className="agent-stream-step">{label}</div>
                    ))}
                    {msg.streamingTokens ? (
                      <div className="agent-stream-tokens">
                        {msg.streamingTokens}<span className="agent-stream-cursor" />
                      </div>
                    ) : (
                      <div className="agent-stream-dots"><span /><span /><span /></div>
                    )}
                  </div>
                ) : (
                  <MessageContent content={msg.content} />
                )}

                {msg.tools?.length > 0 && (
                  <div className="agent-tools-row">
                    {msg.tools.map((t) => (
                      <span key={`${idx}-${t}`} className="agent-tool-chip">{t}</span>
                    ))}
                  </div>
                )}

                {msg.role === "assistant" && (msg.synthesisMode || msg.policyApplied || msg.specialistReports?.length) ? (
                  <div className="agent-meta-row">
                    {msg.synthesisMode && (
                      <span className={`agent-mode-chip ${msg.synthesisMode === "llm" ? "mode-llm" : "mode-deterministic"}`}>
                        {msg.synthesisMode === "llm" ? "Synthèse LLM" : "Synthèse déterministe"}
                      </span>
                    )}
                    {msg.policyApplied && <span className="agent-policy-chip">Policy : {msg.policyApplied}</span>}
                    {msg.specialistReports?.length > 0 && (
                      <span className="agent-specialist-chip">Spécialistes : {msg.specialistReports.length}</span>
                    )}
                  </div>
                ) : null}

                {msg.role === "assistant" && (msg.charts?.length || msg.tables?.length || alerts.length || kpis.length) ? (
                  <div className="agent-structured-zone">
                    {alerts.length > 0 && (
                      <article className="agent-structured-card">
                        <h5>Alertes détectées</h5>
                        <ul className="agent-alert-list">
                          {alerts.map((al, ai) => {
                            const sev = String(al?.severity || "low").toLowerCase();
                            return (
                              <li key={`alert-${ai}`}>
                                <span className={`severity-badge severity-${sev}`}>{sev}</span>
                                <span>{al?.message || "Alerte"}</span>
                              </li>
                            );
                          })}
                        </ul>
                      </article>
                    )}
                    {kpis.length > 0 && (
                      <div className="agent-kpi-grid">
                        {kpis.map((kpi, ki) => <ToolKpi kpi={kpi} index={ki} key={`kpi-${kpi.key}`} />)}
                      </div>
                    )}
                    {(msg.charts || []).map((c, ci) => (
                      <ToolChart chart={c} index={ci} key={`chart-${ci}`} onDrillDown={submitQuestion} />
                    ))}
                    {(msg.tables || []).map((t, ti) => <ToolTable table={t} index={ti} key={`table-${ti}`} />)}
                  </div>
                ) : null}
              </article>
            );
          })}
          <div ref={bottomRef} />
        </div>

        <div className="agent-input-row">
          <textarea
            placeholder="Ex : Donne les KPI critiques et une prévision 3 mois sur AUTO"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={3}
            disabled={isLoading || !ready}
          />
          <button type="button" onClick={() => void submitQuestion(question)}
            disabled={isLoading || !ready || !question.trim()}>
            {!ready ? "Chargement…" : "Envoyer"}
          </button>
        </div>
      </section>
    </div>
  );
}
