import { useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { YEAR_MAX, YEAR_MIN } from "../contexts/FilterContext";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const INITIAL_MESSAGES = [
  {
    role: "assistant",
    content:
      "Bonjour. Je suis l agent IA metier. Pose une question KPI, risque impaye, forecast, drift ou segmentation.",
    tools: [],
    charts: [],
    tables: [],
    toolResults: [],
    llmUsed: null,
    synthesisMode: null,
    policyApplied: null,
    specialistReports: [],
  },
];

function formatCellValue(value) {
  if (value === null || value === undefined) {
    return "-";
  }
  if (typeof value === "number") {
    if (Number.isInteger(value)) {
      return value.toLocaleString("fr-TN");
    }
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
          p: ({ children }) => <p className="agent-message-line">{children}</p>,
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

function ToolTable({ table, index }) {
  const rows = Array.isArray(table?.rows) ? table.rows : [];
  const columns = Array.isArray(table?.columns)
    ? table.columns
    : rows.length > 0
      ? Object.keys(rows[0])
      : [];

  if (rows.length === 0 || columns.length === 0) {
    return null;
  }

  return (
    <article className="agent-structured-card" key={`table-${index}`}>
      <h5>{table?.title || `Table ${index + 1}`}</h5>
      <div className="agent-mini-table-wrapper">
        <table className="agent-mini-table">
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column}>{column}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 8).map((row, rowIndex) => (
              <tr key={`row-${rowIndex}`}>
                {columns.map((column) => (
                  <td key={`${rowIndex}-${column}`}>{formatCellValue(row?.[column])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </article>
  );
}

function ToolChart({ chart, index }) {
  const items = Array.isArray(chart?.items) ? chart.items : [];
  if (items.length === 0) {
    return null;
  }

  const sampleRow = items[0] || {};
  const keys = Object.keys(sampleRow);
  const xKey = chart?.x_key || keys[0];
  const yKey = chart?.y_key || keys[1];

  if (!xKey || !yKey) {
    return null;
  }

  const isLine = String(chart?.type || "").toLowerCase() === "line";
  const forecastStartPeriod = chart?.forecast_start_period || null;
  const defaultLineColor = "#004A8D";
  const palette = ["#004A8D", "#F38F1D", "#1B68B2", "#BE123C"];
  const declaredSeries = Array.isArray(chart?.series) ? chart.series.filter(Boolean) : [];
  const chartSeries =
    declaredSeries.length > 0
      ? declaredSeries
      : [
          {
            key: yKey,
            label: yKey,
            color: isLine ? defaultLineColor : "#F38F1D",
          },
        ];

  const tooltipFormatter = (value, name) => [formatCellValue(value), String(name || "Valeur")];

  return (
    <article className="agent-structured-card" key={`chart-${index}`}>
      <h5>{chart?.title || `Chart ${index + 1}`}</h5>
      <div className="agent-chart-box">
        <ResponsiveContainer width="100%" height={220}>
          {isLine ? (
            <LineChart data={items} margin={{ top: 8, right: 12, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#dbe7f3" />
              <XAxis dataKey={xKey} tick={{ fontSize: 12 }} stroke="#64748b" />
              <YAxis tick={{ fontSize: 12 }} stroke="#64748b" />
              <Tooltip formatter={tooltipFormatter} />
              {forecastStartPeriod ? (
                <ReferenceLine
                  x={forecastStartPeriod}
                  stroke="#dc2626"
                  strokeDasharray="4 4"
                  label={{ value: "Debut prevision", position: "insideTopRight", fill: "#dc2626", fontSize: 11 }}
                />
              ) : null}
              {chartSeries.length > 1 ? <Legend /> : null}
              {chartSeries.map((serie, seriesIndex) => (
                <Line
                  key={`line-${seriesIndex}-${serie?.key || yKey}`}
                  type="monotone"
                  dataKey={serie?.key || yKey}
                  name={serie?.label || serie?.key || yKey}
                  stroke={serie?.color || palette[seriesIndex % palette.length]}
                  strokeWidth={serie?.strokeWidth || 2.2}
                  dot={serie?.dot ?? false}
                  connectNulls={false}
                  strokeDasharray={serie?.strokeDasharray}
                />
              ))}
            </LineChart>
          ) : (
            <BarChart data={items} margin={{ top: 8, right: 12, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#dbe7f3" />
              <XAxis dataKey={xKey} tick={{ fontSize: 12 }} stroke="#64748b" />
              <YAxis tick={{ fontSize: 12 }} stroke="#64748b" />
              <Tooltip formatter={tooltipFormatter} />
              <Bar dataKey={chartSeries[0]?.key || yKey} fill={chartSeries[0]?.color || "#004A8D"} radius={[8, 8, 0, 0]} />
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>
    </article>
  );
}

function ToolKpi({ kpi, index }) {
  const unit = String(kpi?.unit || "").toUpperCase();
  let suffix = "";
  if (unit === "TND") {
    suffix = " TND";
  } else if (unit === "%") {
    suffix = " %";
  }

  return (
    <article className="agent-structured-card agent-kpi-card" key={`kpi-${index}`}>
      <h5>{kpi?.label || `KPI ${index + 1}`}</h5>
      <p className="agent-kpi-value">{`${formatCellValue(kpi?.value)}${suffix}`}</p>
    </article>
  );
}

function extractAlerts(toolResults) {
  const results = Array.isArray(toolResults) ? toolResults : [];
  const alerts = [];

  results.forEach((result) => {
    const payloadAlerts = result?.payload?.alerts;
    if (Array.isArray(payloadAlerts)) {
      payloadAlerts.forEach((alert) => {
        alerts.push(alert);
      });
    }
  });

  return alerts.slice(0, 10);
}

function extractKpis(toolResults) {
  const results = Array.isArray(toolResults) ? toolResults : [];
  const kpis = [];
  const seen = new Set();

  results.forEach((result) => {
    const payloadKpis = result?.payload?.kpis;
    if (!Array.isArray(payloadKpis)) {
      return;
    }

    payloadKpis.forEach((kpi, kpiIndex) => {
      const value = kpi?.value;
      if (typeof value !== "number") {
        return;
      }

      const key = String(kpi?.key || `kpi-${kpiIndex}`);
      if (seen.has(key)) {
        return;
      }
      seen.add(key);

      kpis.push({
        key,
        label: kpi?.label || key,
        value,
        unit: kpi?.unit || "",
      });
    });
  });

  return kpis.slice(0, 6);
}

export default function AgentChat({ filters, recommendedPrompts = [], enableAnalyticFilters = true }) {
  const [messages, setMessages] = useState(INITIAL_MESSAGES);
  const [question, setQuestion] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const payloadTemplate = useMemo(() => {
    const template = {
      top_k: 3,
      horizon_months: 3,
    };

    if (!enableAnalyticFilters || !filters) {
      return template;
    }

    const rawYearFrom = Number(filters.yearFrom);
    const rawYearTo = Number(filters.yearTo);
    const normalizedYearFrom = Number.isFinite(rawYearFrom)
      ? Math.min(YEAR_MAX, Math.max(YEAR_MIN, Math.trunc(rawYearFrom)))
      : YEAR_MIN;
    const normalizedYearTo = Number.isFinite(rawYearTo)
      ? Math.min(YEAR_MAX, Math.max(YEAR_MIN, Math.trunc(rawYearTo)))
      : YEAR_MAX;

    return {
      ...template,
      branch: filters.branch === "ALL" ? null : filters.branch,
      year_from: Math.min(normalizedYearFrom, normalizedYearTo),
      year_to: Math.max(normalizedYearFrom, normalizedYearTo),
      month: filters.month,
      gouvernorat: filters.gouvernorat,
    };
  }, [enableAnalyticFilters, filters]);

  const submitQuestion = async (rawQuestion) => {
    const trimmed = rawQuestion.trim();
    if (!trimmed || isLoading) {
      return;
    }

    const userMessage = { role: "user", content: trimmed, tools: [], charts: [], tables: [], toolResults: [] };
    setMessages((prev) => [...prev, userMessage]);
    setQuestion("");
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE}/api/v1/agent/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: trimmed,
          ...payloadTemplate,
        }),
      });

      if (!response.ok) {
        throw new Error(`Agent API error ${response.status}`);
      }

      const payload = await response.json();
      const agent = payload.agent || {};

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: agent.answer || "Aucune reponse retournee par l agent.",
          tools: agent.invoked_tools || [],
          charts: agent.charts || [],
          tables: agent.tables || [],
          toolResults: agent.tool_results || [],
          status: agent.status || "ok",
          llmUsed: Boolean(agent.llm_used),
          synthesisMode: agent.synthesis_mode || (agent.llm_used ? "llm" : "deterministic"),
          policyApplied: agent.policy_applied || null,
          specialistReports: agent.specialist_reports || [],
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Erreur agent: ${String(error.message || error)}`,
          tools: [],
          charts: [],
          tables: [],
          toolResults: [],
          status: "error",
          llmUsed: null,
          synthesisMode: null,
          policyApplied: null,
          specialistReports: [],
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const askAgent = async () => {
    await submitQuestion(question);
  };

  const handleQuestionKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      void submitQuestion(question);
    }
  };

  return (
    <section className="panel agent-panel">
      <div className="agent-headline">
        <h3>Agent IA CEO</h3>
        <span className="agent-status">{isLoading ? "Analyse..." : "Pret"}</span>
      </div>

      {recommendedPrompts.length > 0 ? (
        <div className="agent-prompt-box">
          <p className="agent-prompt-title">Prompts recommandes</p>
          <div className="agent-prompt-list">
            {recommendedPrompts.map((prompt) => (
              <button
                key={prompt}
                type="button"
                className="agent-prompt-btn"
                onClick={() => {
                  void submitQuestion(prompt);
                }}
                disabled={isLoading}
              >
                {prompt}
              </button>
            ))}
          </div>
        </div>
      ) : null}

      <div className="agent-messages">
        {messages.map((message, index) => {
          const alerts = extractAlerts(message.toolResults);
          const kpis = extractKpis(message.toolResults);
          return (
            <article
              key={`${message.role}-${index}`}
              className={`agent-message ${message.role === "user" ? "agent-user" : "agent-assistant"}`}
            >
              <MessageContent content={message.content} />
              {message.tools && message.tools.length > 0 ? (
                <div className="agent-tools-row">
                  {message.tools.map((tool) => (
                    <span key={`${index}-${tool}`} className="agent-tool-chip">
                      {tool}
                    </span>
                  ))}
                </div>
              ) : null}

              {message.role === "assistant" && (message.synthesisMode || message.policyApplied || message.specialistReports?.length) ? (
                <div className="agent-meta-row">
                  {message.synthesisMode ? (
                    <span className={`agent-mode-chip ${message.synthesisMode === "llm" ? "mode-llm" : "mode-deterministic"}`}>
                      {message.synthesisMode === "llm" ? "Synthese LLM" : "Synthese deterministe"}
                    </span>
                  ) : null}
                  {message.policyApplied ? <span className="agent-policy-chip">Policy: {message.policyApplied}</span> : null}
                  {Array.isArray(message.specialistReports) && message.specialistReports.length > 0 ? (
                    <span className="agent-specialist-chip">Specialistes: {message.specialistReports.length}</span>
                  ) : null}
                </div>
              ) : null}

              {message.role === "assistant" && (message.charts?.length || message.tables?.length || alerts.length || kpis.length) ? (
                <div className="agent-structured-zone">
                  {alerts.length > 0 ? (
                    <article className="agent-structured-card">
                      <h5>Alertes detectees</h5>
                      <ul className="agent-alert-list">
                        {alerts.map((alert, alertIndex) => {
                          const severity = String(alert?.severity || "low").toLowerCase();
                          return (
                            <li key={`alert-${alertIndex}`}>
                              <span className={`severity-badge severity-${severity}`}>{severity}</span>
                              <span>{alert?.message || "Alerte"}</span>
                            </li>
                          );
                        })}
                      </ul>
                    </article>
                  ) : null}

                  {kpis.length > 0 ? (
                    <div className="agent-kpi-grid">
                      {kpis.map((kpi, kpiIndex) => (
                        <ToolKpi kpi={kpi} index={kpiIndex} key={`kpi-wrap-${kpi.key}`} />
                      ))}
                    </div>
                  ) : null}

                  {(message.charts || []).map((chart, chartIndex) => (
                    <ToolChart chart={chart} index={chartIndex} key={`chart-wrap-${chartIndex}`} />
                  ))}

                  {(message.tables || []).map((table, tableIndex) => (
                    <ToolTable table={table} index={tableIndex} key={`table-wrap-${tableIndex}`} />
                  ))}
                </div>
              ) : null}
            </article>
          );
        })}
      </div>

      <div className="agent-input-row">
        <textarea
          placeholder="Exemple: Donne les KPI critiques et une prevision 3 mois sur AUTO"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={handleQuestionKeyDown}
          rows={3}
          disabled={isLoading}
        />
        <button type="button" onClick={askAgent} disabled={isLoading || !question.trim()}>
          Envoyer
        </button>
      </div>
    </section>
  );
}
