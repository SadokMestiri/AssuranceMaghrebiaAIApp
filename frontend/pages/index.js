import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";

import AgentChat from "../components/AgentChat";
import ChartsPanel from "../components/ChartsPanel";
import FiltersBar from "../components/FiltersBar";
import GeoInsights from "../components/GeoInsights";
import KpiCards from "../components/KpiCards";
import MLOpsContent from "../components/MLOpsContent";
import { useFilters, YEAR_MAX, YEAR_MIN } from "../contexts/FilterContext";

const CarteWidget = dynamic(() => import("../components/CarteWidget"), {
  ssr: false,
});

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const AGENT_RECOMMENDED_PROMPTS = [
  "Resume les KPI critiques sur la periode active",
  "Donne une prevision 3 mois de la prime nette",
  "Detecte les anomalies impaye et propose un plan d action",
  "Explique les facteurs de risque impaye pour la branche AUTO",
  "Segmente les agents pour prioriser production et recouvrement",
];

const AGENT_TOOL_SLOTS = [
  "kpi tool",
  "rag tool",
  "alerte tool",
  "forecast tool",
  "anomaly tool",
  "drift tool",
  "explain tool",
  "segmentation tool",
  "client tool",
  "sql tool",
];

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed (${response.status}) on ${url}`);
  }
  return response.json();
}

function buildCommonQuery(filters) {
  const params = new URLSearchParams();

  if (filters.branch !== "ALL") {
    params.set("branch", filters.branch);
  }

  const rawYearFrom = Number(filters.yearFrom);
  const rawYearTo = Number(filters.yearTo);
  const normalizedYearFrom = Number.isFinite(rawYearFrom)
    ? Math.min(YEAR_MAX, Math.max(YEAR_MIN, Math.trunc(rawYearFrom)))
    : YEAR_MIN;
  const normalizedYearTo = Number.isFinite(rawYearTo)
    ? Math.min(YEAR_MAX, Math.max(YEAR_MIN, Math.trunc(rawYearTo)))
    : YEAR_MAX;

  params.set("year_from", String(Math.min(normalizedYearFrom, normalizedYearTo)));
  params.set("year_to", String(Math.max(normalizedYearFrom, normalizedYearTo)));

  return params.toString();
}

export default function DashboardPage() {
  const { filters } = useFilters();
  const [activeSection, setActiveSection] = useState("dashboard");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [dashboard, setDashboard] = useState(null);
  const [heatmapPoints, setHeatmapPoints] = useState([]);
  const [sinistresByGov, setSinistresByGov] = useState([]);
  const [topZones, setTopZones] = useState([]);
  const [updatedAt, setUpdatedAt] = useState("");
  const [agentStatus, setAgentStatus] = useState(null);
  const [smokeEval, setSmokeEval] = useState(null);
  const [smokeLoading, setSmokeLoading] = useState(false);
  const [warmupReport, setWarmupReport] = useState(null);
  const [warmupLoading, setWarmupLoading] = useState(false);

  useEffect(() => {
    let active = true;

    const loadData = async () => {
      setLoading(true);
      setError("");

      try {
        const query = buildCommonQuery(filters);

        const [dashboardPayload, heatmapPayload, sinistresPayload, zonesPayload, ltvPayload, churnPayload] = await Promise.all([
          fetchJson(`${API_BASE}/api/v1/kpis/dashboard/ceo?${query}`),
          fetchJson(`${API_BASE}/api/v1/geo/heatmap-polices?${query}&limit=300`),
          fetchJson(`${API_BASE}/api/v1/geo/sinistres/by-gouvernorat?${query}`),
          fetchJson(`${API_BASE}/api/v1/geo/top-zones-risque?${query}&limit=10`),
          fetchJson(`${API_BASE}/api/v1/kpis/ml/client-ltv`),
          fetchJson(`${API_BASE}/api/v1/kpis/ml/churn-risk`),
        ]);

        if (!active) {
          return;
        }

        dashboardPayload.ml_ltv = ltvPayload;
        dashboardPayload.ml_churn = churnPayload;
        setDashboard(dashboardPayload);
        setHeatmapPoints(heatmapPayload.items || []);
        setSinistresByGov(sinistresPayload.items || []);
        setTopZones(zonesPayload.items || []);
        setUpdatedAt(new Date().toISOString());
      } catch (requestError) {
        if (!active) {
          return;
        }
        setError(String(requestError.message || requestError));
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    };

    loadData();

    return () => {
      active = false;
    };
  }, [filters.branch, filters.yearFrom, filters.yearTo]);

  useEffect(() => {
    let active = true;

    const loadAgentStatus = async () => {
      try {
        const payload = await fetchJson(`${API_BASE}/api/v1/agent/status`);
        if (active) {
          setAgentStatus(payload);
        }
      } catch {
        if (active) {
          setAgentStatus({ status: "degraded", dependencies: {} });
        }
      }
    };

    loadAgentStatus();
    const timer = setInterval(loadAgentStatus, 30000);

    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  const runSmokeEval = async () => {
    setSmokeLoading(true);
    try {
      const payload = await fetchJson(`${API_BASE}/api/v1/agent/eval/smoke`);
      setSmokeEval(payload);
    } catch (requestError) {
      setSmokeEval({
        status: "error",
        passed: 0,
        total: 0,
        results: [{ name: "smoke_eval", pass: false, error: String(requestError.message || requestError) }],
      });
    } finally {
      setSmokeLoading(false);
    }
  };

  const runWarmup = async (preindex) => {
    setWarmupLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/v1/agent/warmup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          preindex,
          strict: false,
          max_docs_per_collection: 250,
        }),
      });

      if (!response.ok) {
        throw new Error(`Warmup API error ${response.status}`);
      }

      const payload = await response.json();
      setWarmupReport(payload);
      const statusPayload = await fetchJson(`${API_BASE}/api/v1/agent/status`);
      setAgentStatus(statusPayload);
    } catch (requestError) {
      setWarmupReport({
        status: "error",
        warmup: {
          errors: [String(requestError.message || requestError)],
        },
      });
    } finally {
      setWarmupLoading(false);
    }
  };

  const governorates = useMemo(() => {
    const set = new Set();

    (sinistresByGov || []).forEach((item) => {
      if (item.gouvernorat) {
        set.add(item.gouvernorat);
      }
    });

    (heatmapPoints || []).forEach((item) => {
      if (item.gouvernorat) {
        set.add(item.gouvernorat);
      }
    });

    (topZones || []).forEach((item) => {
      if (item.gouvernorat) {
        set.add(item.gouvernorat);
      }
    });

    return Array.from(set).sort((a, b) => a.localeCompare(b));
  }, [heatmapPoints, sinistresByGov, topZones]);

  const filteredHeatmap = useMemo(() => {
    if (filters.gouvernorat === "ALL") {
      return heatmapPoints;
    }
    return (heatmapPoints || []).filter((item) => item.gouvernorat === filters.gouvernorat);
  }, [filters.gouvernorat, heatmapPoints]);

  const filteredSinistresByGov = useMemo(() => {
    if (filters.gouvernorat === "ALL") {
      return sinistresByGov;
    }
    return (sinistresByGov || []).filter((item) => item.gouvernorat === filters.gouvernorat);
  }, [filters.gouvernorat, sinistresByGov]);

  const filteredTopZones = useMemo(() => {
    if (filters.gouvernorat === "ALL") {
      return topZones;
    }
    return (topZones || []).filter((item) => item.gouvernorat === filters.gouvernorat);
  }, [filters.gouvernorat, topZones]);

  return (
    <main className="app-shell">
      <aside className="app-nav-sidebar">
        <div className="app-brand-block">
          <p className="app-brand-kicker">Maghrebia</p>
          <h2>Control Center</h2>
        </div>

        <nav className="app-nav-menu">
          <button
            type="button"
            className={`app-nav-item ${activeSection === "dashboard" ? "active" : ""}`}
            onClick={() => setActiveSection("dashboard")}
          >
            Dashboard
          </button>
          <button
            type="button"
            className={`app-nav-item ${activeSection === "agent" ? "active" : ""}`}
            onClick={() => setActiveSection("agent")}
          >
            AI Agent
          </button>
          <button
            type="button"
            className={`app-nav-item ${activeSection === "mlops" ? "active" : ""}`}
            onClick={() => setActiveSection("mlops")}
          >
            MLOps
          </button>
        </nav>

        <p className="app-nav-note">Dashboard executif et Agent IA.</p>
      </aside>

      <section className="page-shell">
        <header className="hero">
          <p className="hero-kicker">Maghrebia Assurance</p>
          <h1>
            {activeSection === "dashboard"
              ? "Dashboard KPI & Geo Risk"
              : activeSection === "mlops"
              ? "Machine Learning Operations"
              : "AI Agent Workspace"}
          </h1>
          <p>
            {activeSection === "dashboard"
              ? "Vue metier pour primes, retention, commissions et zones a risque avec une carte Tunisie orientee pilotage CEO et recouvrement."
              : activeSection === "mlops"
              ? "Surveillance et inférence des modèles de ML (Impayés, Prophet, Drift, Isolation Forest)."
              : "Section dediee au chat metier avec orchestration intent, outils specialises et synthese LLM (filtres analytiques desactives)."}
          </p>
          {updatedAt ? (
            <span className="hero-pill">
              Derniere mise a jour: {new Date(updatedAt).toLocaleString("fr-TN")}
            </span>
          ) : null}
        </header>

        {activeSection === "dashboard" ? (
          <FiltersBar governorates={governorates} loading={loading} />
        ) : null}

        {error ? (
          <section className="panel error-panel">
            <h3>Erreur API</h3>
            <p>{error}</p>
          </section>
        ) : null}

        {activeSection === "dashboard" ? (
          <section className="dashboard-section">
            <KpiCards dashboard={dashboard} />
            <ChartsPanel dashboard={dashboard} monthFilter={filters.month} />

            <section className="layout-geo">
              <article className="panel map-panel">
                <h3>Carte Leaflet Tunisie - heatmap polices</h3>
                <CarteWidget points={filteredHeatmap} />
              </article>

              <GeoInsights sinistresByGov={filteredSinistresByGov} topZones={filteredTopZones} />
            </section>
          </section>
        ) : activeSection === "mlops" ? (
          <section className="mlops-section-layout">
            <MLOpsContent />
          </section>
        ) : (
          <section className="agent-section-layout">
            <AgentChat
              filters={filters}
              enableAnalyticFilters={false}
              recommendedPrompts={AGENT_RECOMMENDED_PROMPTS}
            />

            <aside className="panel agent-tools-panel">
              <h3>Agent Runtime</h3>
              <p className="muted-line">Etat des dependances et catalogue des tools metier actifs.</p>

              <div className="agent-runtime-status">
                <span className={`runtime-pill ${agentStatus?.status === "ready" ? "ok" : "warn"}`}>
                  {agentStatus?.status === "ready" ? "Runtime READY" : "Runtime DEGRADED"}
                </span>
                <span className={`runtime-pill ${agentStatus?.dependencies?.ollama?.ok ? "ok" : "warn"}`}>
                  Ollama {agentStatus?.dependencies?.ollama?.ok ? "UP" : "DOWN"}
                </span>
                <span className={`runtime-pill ${agentStatus?.dependencies?.qdrant?.ok ? "ok" : "warn"}`}>
                  Qdrant {agentStatus?.dependencies?.qdrant?.ok ? "UP" : "DOWN"}
                </span>
              </div>

              <div className="sidebar-divider" />
              <ul className="agent-tools-planned-list">
                {AGENT_TOOL_SLOTS.map((toolName) => (
                  <li key={toolName}>{toolName}</li>
                ))}
              </ul>

              <div className="sidebar-divider" />
              <div className="agent-eval-panel">
                <h4>Validation rapide</h4>
                <div className="agent-eval-actions">
                  <button type="button" onClick={runSmokeEval} disabled={smokeLoading || warmupLoading}>
                    {smokeLoading ? "Smoke en cours..." : "Lancer Smoke Eval"}
                  </button>
                  <button
                    type="button"
                    className="secondary"
                    onClick={() => runWarmup(false)}
                    disabled={smokeLoading || warmupLoading}
                  >
                    {warmupLoading ? "Warmup..." : "Warmup runtime"}
                  </button>
                  <button
                    type="button"
                    className="secondary"
                    onClick={() => runWarmup(true)}
                    disabled={smokeLoading || warmupLoading}
                  >
                    {warmupLoading ? "Indexing..." : "Warmup + Preindex"}
                  </button>
                </div>

                {smokeEval ? (
                  <div className="agent-eval-results">
                    <p className={`eval-summary ${smokeEval.status === "ok" ? "ok" : "warn"}`}>
                      Smoke: {smokeEval.status} ({smokeEval.passed}/{smokeEval.total})
                    </p>
                    <ul>
                      {(smokeEval.results || []).map((item) => (
                        <li key={item.name}>
                          <span className={`eval-badge ${item.pass ? "ok" : "warn"}`}>
                            {item.pass ? "PASS" : "FAIL"}
                          </span>
                          <span>{item.name}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}

                {warmupReport ? (
                  <div className="agent-eval-results">
                    <p className={`eval-summary ${warmupReport.status === "ok" ? "ok" : "warn"}`}>
                      Warmup: {warmupReport.status}
                    </p>
                    {(warmupReport.warmup?.errors || []).length > 0 ? (
                      <ul>
                        {(warmupReport.warmup.errors || []).map((errorValue, index) => (
                          <li key={`warmup-error-${index}`}>
                            <span className="eval-badge warn">WARN</span>
                            <span>{errorValue}</span>
                          </li>
                        ))}
                      </ul>
                    ) : null}
                  </div>
                ) : null}
              </div>
            </aside>
          </section>
        )}

        {loading ? (
          <section className="panel loading-panel">
            <p>Chargement en cours...</p>
          </section>
        ) : null}
      </section>
    </main>
  );
}
