import dynamic from "next/dynamic";
import { useEffect, useMemo, useRef, useState } from "react";
import { LogOut, Moon, Sun } from "lucide-react";

import AgentChat from "../components/AgentChat";
import AdminPanel from "../components/AdminPanel";
import AlertsPanel from "../components/AlertsPanel";
import LandingPage from "../components/LandingPage";
import { KeyrusMark } from "../components/KeyrusLogo";
import { useAuth } from "../contexts/AuthContext";
import ResiliationChart from "../components/ResiliationChart";
import ChartsPanel from "../components/ChartsPanel";
import DimNav from "../components/dims/DimNav";
import FiltersBar from "../components/FiltersBar";
import GeoInsights from "../components/GeoInsights";
import KpiCards from "../components/KpiCards";
import MLOpsContent from "../components/MLOpsContent";
import AgentDim from "../components/dims/AgentDim";
import ClientDim from "../components/dims/ClientDim";
import PoliceDim from "../components/dims/PoliceDim";
import ProduitDim from "../components/dims/ProduitDim";
import SinistreDim from "../components/dims/SinistreDim";
import VehiculeDim from "../components/dims/VehiculeDim";
import { useFilters, YEAR_MAX, YEAR_MIN } from "../contexts/FilterContext";
import { getAllowedSections, getAllowedDims } from "../lib/roleAccess";

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

// Which filters each dim actually applies. Used by FiltersBar to annotate
// inapplicable controls and by dim components to show contextual notices.
const DIM_FILTER_SUPPORT = {
  overview:  { branch: true,       year: true,  month: "partial", gouvernorat: "partial" },
  clients:   { branch: true,       year: true,  month: true,      gouvernorat: false },
  agents:    { branch: true,       year: true,  month: true,      gouvernorat: false },
  produits:  { branch: true,       year: true,  month: true,      gouvernorat: false },
  polices:   { branch: true,       year: false, month: false,     gouvernorat: false },
  sinistres: { branch: true,       year: true,  month: true,      gouvernorat: false },
  vehicules: { branch: "auto",     year: false, month: false,     gouvernorat: false },
};

const DIM_LABELS = {
  overview:  "Vue Globale",
  clients:   "Clients",
  agents:    "Agents",
  produits:  "Produits",
  vehicules: "Véhicules",
  polices:   "Polices",
  sinistres: "Sinistres",
};

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed (${response.status}) on ${url}`);
  }
  return response.json();
}

function buildCommonQuery(filters, { includeMonth = false } = {}) {
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

  if (includeMonth && filters.month && filters.month !== "ALL") {
    params.set("month", filters.month);
  }

  return params.toString();
}

function DashboardPage() {
  const { filters } = useFilters();
  const { user, logout } = useAuth();
  const allowedSections = useMemo(() => getAllowedSections(user?.role), [user?.role]);
  const allowedDims     = useMemo(() => getAllowedDims(user?.role), [user?.role]);
  const [darkMode, setDarkMode]             = useState(false);
  const [activeSection, setActiveSection]   = useState("dashboard");
  const [activeDim, setActiveDim]           = useState("overview");
  const [alertCount, setAlertCount]         = useState(0);
  const [alertHasCritical, setAlertHasCritical] = useState(false);

  // Lightweight poll just for the nav badge — AlertsPanel does its own full
  // fetch when the tab is actually open. Skipped entirely for roles that
  // can't see the tab (ceo/admin only) so it's not wasted API traffic.
  useEffect(() => {
    if (!allowedSections.includes("alerts")) return;
    let cancelled = false;
    const fetchCount = () => {
      const params = new URLSearchParams({ months: "12" });
      if (filters.branch !== "ALL") params.set("branch", filters.branch);
      fetch(`${API_BASE}/api/v1/alerts?${params.toString()}`)
        .then((res) => (res.ok ? res.json() : null))
        .then((data) => {
          if (cancelled || !data) return;
          setAlertCount(data.count || 0);
          setAlertHasCritical((data.alerts || []).some((a) => a.severity === "high"));
        })
        .catch(() => {});
    };
    fetchCount();
    const interval = setInterval(fetchCount, 90_000);
    return () => { cancelled = true; clearInterval(interval); };
  }, [filters.branch]);

  // Some roles don't have "dashboard"/"overview" in their allowed set (e.g.
  // "agent" has no overview dim) — snap to the first thing that role can
  // actually see instead of rendering a blank/forbidden default view.
  useEffect(() => {
    if (!allowedSections.includes(activeSection)) {
      setActiveSection(allowedSections[0] || "dashboard");
    }
  }, [allowedSections, activeSection]);

  useEffect(() => {
    if (activeSection === "dashboard" && !allowedDims.includes(activeDim)) {
      setActiveDim(allowedDims[0] || "overview");
    }
  }, [allowedDims, activeSection, activeDim]);
  const [loading, setLoading]               = useState(true);
  const [error, setError]                   = useState("");
  const [dashboard, setDashboard]           = useState(null);
  const [dashboardPrev, setDashboardPrev]   = useState(null);
  const [heatmapPoints, setHeatmapPoints]   = useState([]);
  const [resiliation, setResiliation]       = useState([]);
  const [sinistresByGov, setSinistresByGov] = useState([]);
  const [topZones, setTopZones]             = useState([]);
  const [updatedAt, setUpdatedAt]           = useState("");
  const [agentStatus, setAgentStatus]       = useState(null);
  const [smokeEval, setSmokeEval]           = useState(null);
  const [smokeLoading, setSmokeLoading]     = useState(false);
  const [warmupReport, setWarmupReport]     = useState(null);
  const [warmupLoading, setWarmupLoading]   = useState(false);
  const [pdfExporting, setPdfExporting]     = useState(false);
  const [pdfError, setPdfError]             = useState("");
  const dashboardCaptureRef = useRef(null);

  // Dimension data state
  const [dimData, setDimData]         = useState(null);
  const [dimDataPrev, setDimDataPrev] = useState(null);
  const [dimLoading, setDimLoading]   = useState(false);
  const [dimError, setDimError]       = useState("");

  // ── Dark mode ────────────────────────────────────────────────
  useEffect(() => {
    const saved = localStorage.getItem("theme");
    if (saved === "dark") {
      setDarkMode(true);
      document.documentElement.classList.add("dark");
    }
  }, []);

  function toggleDarkMode() {
    const next = !darkMode;
    setDarkMode(next);
    if (next) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  }

  async function exportDashboardPdf() {
    if (!dashboardCaptureRef.current || pdfExporting) return;
    setPdfExporting(true);
    setPdfError("");
    try {
      const [{ default: html2canvas }, { jsPDF }] = await Promise.all([
        import("html2canvas"),
        import("jspdf"),
      ]);

      const node = dashboardCaptureRef.current;

      // Leaflet's tile layer loads raster tiles asynchronously and positions
      // them with CSS transforms; html2canvas clones the DOM into a detached
      // container to rasterize it, which desyncs Leaflet's zoom/pan math from
      // its real container size — the map renders blank, mis-zoomed, or with
      // the wrong extent depending on exactly when the capture lands. Rather
      // than chase that timing race, skip the map during capture and stamp a
      // labeled placeholder in its place so the PDF stays consistent.
      const mapPanel = node.querySelector(".pdf-map-panel");
      const nodeRectPre = node.getBoundingClientRect();
      const mapGapPx = mapPanel
        ? (() => {
            const r = mapPanel.getBoundingClientRect();
            return { x: r.left - nodeRectPre.left, y: r.top - nodeRectPre.top, width: r.width, height: r.height };
          })()
        : null;

      const surfaceColor = getComputedStyle(document.documentElement).getPropertyValue("--panel").trim() || "#ffffff";
      const canvas = await html2canvas(node, {
        scale: 2,
        useCORS: true,
        backgroundColor: surfaceColor,
        ignoreElements: (el) => el.classList?.contains("pdf-map-panel"),
      });

      const pdf = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      const margin = 10;
      const headerHeight = 16;

      const branchLabel = filters.branch === "ALL" ? "Toutes branches" : filters.branch;
      const periodLabel = `${filters.yearFrom}-${filters.yearTo}`;
      const dimLabel = DIM_LABELS[activeDim] || activeDim;

      pdf.setFontSize(13);
      pdf.setFont(undefined, "bold");
      pdf.text(`Maghrebia — Dashboard : ${dimLabel}`, margin, margin + 2);
      pdf.setFontSize(8.5);
      pdf.setFont(undefined, "normal");
      pdf.text(
        `${branchLabel} · période ${periodLabel} · exporté le ${new Date().toLocaleString("fr-TN")}`,
        margin,
        margin + 8
      );

      const imgData = canvas.toDataURL("image/png");
      const imgWidth = pageWidth - margin * 2;
      const imgHeight = (canvas.height * imgWidth) / canvas.width;
      const pageContentHeight = pageHeight - margin * 2;
      const mmPerPx = imgWidth / nodeRectPre.width;

      // Draws the "carte non incluse" placeholder wherever the excluded map's
      // gap intersects the page currently on top — the gap's position in the
      // full image is fixed, but which page (and where on it) it lands on
      // shifts with every addImage call below.
      function drawMapPlaceholder(imagePositionMm) {
        if (!mapGapPx) return;
        const gapTop = imagePositionMm + mapGapPx.y * mmPerPx;
        const gapBottom = gapTop + mapGapPx.height * mmPerPx;
        if (gapBottom <= 0 || gapTop >= pageHeight) return; // no overlap with this page
        const boxX = margin + mapGapPx.x * mmPerPx;
        const boxY = Math.max(gapTop, 0);
        const boxW = mapGapPx.width * mmPerPx;
        const boxH = Math.min(gapBottom, pageHeight) - boxY;
        if (boxH < 4) return;
        pdf.setDrawColor(200, 200, 200);
        pdf.setFillColor(246, 247, 250);
        pdf.rect(boxX, boxY, boxW, boxH, "FD");
        pdf.setFontSize(8.5);
        pdf.setTextColor(120, 120, 120);
        pdf.text("Carte interactive non incluse — voir le tableau de bord en direct", boxX + boxW / 2, boxY + boxH / 2, {
          align: "center",
        });
        pdf.setTextColor(0, 0, 0);
      }

      let heightLeft = imgHeight;
      let position = margin + headerHeight;

      pdf.addImage(imgData, "PNG", margin, position, imgWidth, imgHeight);
      drawMapPlaceholder(position);
      heightLeft -= pageContentHeight - headerHeight;

      while (heightLeft > 0) {
        position -= pageContentHeight;
        pdf.addPage();
        pdf.addImage(imgData, "PNG", margin, position, imgWidth, imgHeight);
        drawMapPlaceholder(position);
        heightLeft -= pageContentHeight;
      }

      pdf.save(`maghrebia-dashboard-${activeDim}-${Date.now()}.pdf`);
    } catch (exportError) {
      setPdfError(`Export PDF echoue: ${String(exportError.message || exportError)}`);
    } finally {
      setPdfExporting(false);
    }
  }

  function buildPrevQuery(filters) {
    const rawYearFrom = Number(filters.yearFrom);
    const rawYearTo   = Number(filters.yearTo);
    const yFrom = Number.isFinite(rawYearFrom) ? Math.min(YEAR_MAX, Math.max(YEAR_MIN, Math.trunc(rawYearFrom))) : YEAR_MIN;
    const yTo   = Number.isFinite(rawYearTo)   ? Math.min(YEAR_MAX, Math.max(YEAR_MIN, Math.trunc(rawYearTo)))   : YEAR_MAX;
    // YoY only makes sense for a single selected year — a multi-year span
    // has no well-defined "previous period" to compare against.
    if (yFrom !== yTo) return null;
    const prevYear = yFrom - 1;
    // Only fetch if that prior year is within available data
    if (prevYear < YEAR_MIN) return null;
    const params = new URLSearchParams();
    if (filters.branch !== "ALL") params.set("branch", filters.branch);
    params.set("year_from", String(prevYear));
    params.set("year_to",   String(prevYear));
    return params.toString();
  }

  // ── Load main dashboard data ─────────────────────────────────
  useEffect(() => {
    let active = true;

    const loadData = async () => {
      setLoading(true);
      setError("");

      try {
        const query     = buildCommonQuery(filters);
        const prevQuery = buildPrevQuery(filters);

        const [dashboardPayload, heatmapPayload, sinistresPayload, zonesPayload, ltvPayload, churnPayload, branchPayload, prevPayload] =
          await Promise.all([
            fetchJson(`${API_BASE}/api/v1/kpis/dashboard/ceo?${query}`),
            fetchJson(`${API_BASE}/api/v1/geo/heatmap-polices?${query}`),
            fetchJson(`${API_BASE}/api/v1/geo/sinistres/by-gouvernorat?${query}`),
            fetchJson(`${API_BASE}/api/v1/geo/top-zones-risque?${query}&limit=10`),
            fetchJson(`${API_BASE}/api/v1/kpis/ml/client-ltv`),
            fetchJson(`${API_BASE}/api/v1/kpis/ml/churn-risk`),
            fetchJson(`${API_BASE}/api/v1/kpis/annulation-monthly?${query}`),
            prevQuery ? fetchJson(`${API_BASE}/api/v1/kpis/dashboard/ceo?${prevQuery}`).catch(() => null) : Promise.resolve(null),
          ]);

        if (!active) return;

        dashboardPayload.ml_ltv   = ltvPayload;
        dashboardPayload.ml_churn = churnPayload;
        setDashboard(dashboardPayload);
        setDashboardPrev(prevPayload);
        setHeatmapPoints(heatmapPayload.items || []);
        setSinistresByGov(sinistresPayload.items || []);
        setTopZones(zonesPayload.items || []);
        setResiliation(branchPayload.items || []);
        setUpdatedAt(new Date().toISOString());
      } catch (requestError) {
        if (!active) return;
        setError(String(requestError.message || requestError));
      } finally {
        if (active) setLoading(false);
      }
    };

    loadData();
    return () => { active = false; };
  }, [filters.branch, filters.yearFrom, filters.yearTo]);

  // ── Load dimension data once ──────────────────────────────────
  useEffect(() => {
    let active = true;

    const loadDimData = async () => {
      setDimLoading(true);
      setDimError("");
      try {
        const query     = buildCommonQuery(filters, { includeMonth: true });
        const prevQuery = buildPrevQuery(filters);
        const [payload, prevPayload] = await Promise.all([
          fetchJson(`/api/dims?${query}`),
          prevQuery ? fetchJson(`/api/dims?${prevQuery}`).catch(() => null) : Promise.resolve(null),
        ]);
        if (active) {
          setDimData(payload);
          setDimDataPrev(prevPayload);
        }
      } catch (err) {
        if (active) setDimError(String(err.message || err));
      } finally {
        if (active) setDimLoading(false);
      }
    };

    loadDimData();
    return () => { active = false; };
  }, [filters.branch, filters.yearFrom, filters.yearTo, filters.month]);

  // ── Agent status polling — only runs when the agent tab is visible ─────────
  useEffect(() => {
    if (activeSection !== "agent") return;

    let active = true;

    const loadAgentStatus = async () => {
      try {
        const payload = await fetchJson(`${API_BASE}/api/v1/agent/status`);
        if (active) setAgentStatus(payload);
      } catch {
        if (active) setAgentStatus({ status: "degraded", dependencies: {} });
      }
    };

    loadAgentStatus();
    const timer = setInterval(loadAgentStatus, 30000);
    return () => { active = false; clearInterval(timer); };
  }, [activeSection]);

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
        body: JSON.stringify({ preindex, strict: false, max_docs_per_collection: 250 }),
      });
      if (!response.ok) throw new Error(`Warmup API error ${response.status}`);
      const payload = await response.json();
      setWarmupReport(payload);
      const statusPayload = await fetchJson(`${API_BASE}/api/v1/agent/status`);
      setAgentStatus(statusPayload);
    } catch (requestError) {
      setWarmupReport({ status: "error", warmup: { errors: [String(requestError.message || requestError)] } });
    } finally {
      setWarmupLoading(false);
    }
  };

  const governorates = useMemo(() => {
    const map = new Map(); 
    const add = (val) => {
      if (!val) return;
      map.set(val.toUpperCase().trim(), val.toUpperCase().trim()); 
    };
    (sinistresByGov || []).forEach((item) => add(item.gouvernorat));
    (heatmapPoints  || []).forEach((item) => add(item.gouvernorat));
    (topZones       || []).forEach((item) => add(item.gouvernorat));
    return Array.from(map.values()).sort((a, b) => a.localeCompare(b));
  }, [heatmapPoints, sinistresByGov, topZones]);

  const filteredHeatmap = useMemo(() => {
    if (filters.gouvernorat === "ALL") return heatmapPoints;
    return (heatmapPoints || []).filter((item) => item.gouvernorat === filters.gouvernorat);
  }, [filters.gouvernorat, heatmapPoints]);

  const filteredSinistresByGov = useMemo(() => {
    if (filters.gouvernorat === "ALL") return sinistresByGov;
    return (sinistresByGov || []).filter(
      (item) => item.gouvernorat?.toUpperCase().trim() === filters.gouvernorat?.toUpperCase().trim()  // ← fix
    );
  }, [filters.gouvernorat, sinistresByGov]);

  const filteredTopZones = useMemo(() => {
    if (filters.gouvernorat === "ALL") return topZones;
    return (topZones || []).filter(
      (item) => item.gouvernorat?.toUpperCase().trim() === filters.gouvernorat?.toUpperCase().trim()  // ← fix
    );
  }, [filters.gouvernorat, topZones]);

  // ── Dimension content renderer ────────────────────────────────
  function renderDimContent() {
    if (activeDim === "overview") {
      if (loading && !dashboard) {
        return (
          <section className="panel loading-panel">
            <p>Chargement du tableau de bord…</p>
          </section>
        );
      }
      if (error) {
        return (
          <section className="panel error-panel">
            <h3>Erreur de chargement</h3>
            <p>{error}</p>
          </section>
        );
      }
      return (
        <>
          <KpiCards dashboard={dashboard} dashboardPrev={dashboardPrev} />
          <ChartsPanel dashboard={dashboard} monthFilter={filters.month} />
          <section className="layout-geo">
            <div style={{ display: "flex", flexDirection: "column", gap: "12px", height: "100%" }}>
              <article className="panel map-panel pdf-map-panel">
                <h3>Carte Leaflet Tunisie — heatmap polices</h3>
                <CarteWidget points={filteredHeatmap} />
              </article>
              <ResiliationChart data={resiliation} />
            </div>
            <GeoInsights sinistresByGov={filteredSinistresByGov} topZones={filteredTopZones} />
          </section>
        </>
      );
    }

    if (dimLoading) {
      return (
        <section className="panel loading-panel">
          <p>Chargement des données dimensionnelles…</p>
        </section>
      );
    }

    if (dimError) {
      return (
        <section className="panel error-panel">
          <h3>Erreur dimensions</h3>
          <p>{dimError}</p>
        </section>
      );
    }

    switch (activeDim) {
      case "clients":   return <ClientDim   data={dimData?.clients}   dataPrev={dimDataPrev?.clients}   />;
      case "agents":    return <AgentDim    data={dimData?.agents}    dataPrev={dimDataPrev?.agents}    />;
      case "produits":  return <ProduitDim  data={dimData?.produits}  dataPrev={dimDataPrev?.produits}  />;
      case "vehicules": return <VehiculeDim data={dimData?.vehicules} dataPrev={dimDataPrev?.vehicules} branch={filters.branch} />;
      case "polices":   return <PoliceDim   data={dimData?.polices}   dataPrev={dimDataPrev?.polices}   />;
      case "sinistres": return <SinistreDim data={dimData?.sinistres} dataPrev={dimDataPrev?.sinistres} />;
      default:          return null;
    }
  }

  return (
    <main className="app-shell">
      <aside className="app-nav-sidebar">
        <div className="app-brand-block">
          <KeyrusMark className="app-brand-logo" />
          <h2>Control Center</h2>
        </div>

        <nav className="app-nav-menu">
          {allowedSections.includes("dashboard") && (
            <button
              type="button"
              className={`app-nav-item ${activeSection === "dashboard" ? "active" : ""}`}
              onClick={() => setActiveSection("dashboard")}
            >
              Dashboard
            </button>
          )}
          {allowedSections.includes("alerts") && (
            <button
              type="button"
              className={`app-nav-item ${activeSection === "alerts" ? "active" : ""}`}
              onClick={() => setActiveSection("alerts")}
            >
              Alertes
              {alertCount > 0 && (
                <span className={`app-nav-badge ${alertHasCritical ? "" : "app-nav-badge-info"}`}>
                  {alertCount}
                </span>
              )}
            </button>
          )}
          {allowedSections.includes("agent") && (
            <button
              type="button"
              className={`app-nav-item ${activeSection === "agent" ? "active" : ""}`}
              onClick={() => setActiveSection("agent")}
            >
              <span className="keyrus-ai-logo">
                <img src="/images/KeyrusAILightMode.png" alt="Keyrus Ai" className="kai-img kai-img-light" />
                <img src="/images/KeyrusAIDarkMode.png" alt="Keyrus Ai" className="kai-img kai-img-dark" />
              </span>
            </button>
          )}
          {allowedSections.includes("mlops") && (
            <button
              type="button"
              className={`app-nav-item ${activeSection === "mlops" ? "active" : ""}`}
              onClick={() => setActiveSection("mlops")}
            >
              Insurance Intelligence
            </button>
          )}
          {allowedSections.includes("admin") && (
            <button
              type="button"
              className={`app-nav-item ${activeSection === "admin" ? "active" : ""}`}
              onClick={() => setActiveSection("admin")}
            >
              Administration
            </button>
          )}
        </nav>

        <p className="app-nav-note">Dashboard executif et Agent IA.</p>

        {user && (
          <div className="app-user-footer">
            <div className="app-user-info">
              <span className="app-user-name">{user.prenom} {user.nom}</span>
              <span className="app-user-role">{user.role_label}</span>
            </div>
            <button type="button" className="app-logout-btn" onClick={logout} title="Se deconnecter">
              <LogOut size={15} />
            </button>
          </div>
        )}
      </aside>

      <button
        type="button"
        className="theme-toggle-float"
        onClick={toggleDarkMode}
        title={darkMode ? "Passer en mode clair" : "Passer en mode sombre"}
        aria-label={darkMode ? "Passer en mode clair" : "Passer en mode sombre"}
      >
        {darkMode ? <Sun size={18} /> : <Moon size={18} />}
      </button>

      <section className="page-shell">
        <header className="hero">
          <p className="hero-kicker">Keyrus - Make data matter.</p>
          <h1>
            {activeSection === "dashboard"
              ? "Dashboard KPI & Geo Risk"
              : activeSection === "mlops"
              ? "Insurance Intelligence"
              : "AI Agent Workspace"}
          </h1>
          <p>
            {activeSection === "dashboard"
              ? "Vue metier pour primes, retention, commissions et zones a risque avec une carte Tunisie orientee pilotage CEO et recouvrement."
              : activeSection === "mlops"
              ? "Surveillance et inférence des modèles de ML."
              : "Section dediee au chat metier avec orchestration intent, outils specialises et synthese LLM (filtres analytiques desactives)."}
          </p>
          {updatedAt ? (
            <span className="hero-pill">
              Derniere mise a jour: {new Date(updatedAt).toLocaleString("fr-TN")}
            </span>
          ) : null}
        </header>

        {activeSection === "dashboard" ? (
          <FiltersBar
            governorates={governorates}
            loading={loading}
            filterSupport={DIM_FILTER_SUPPORT[activeDim] ?? DIM_FILTER_SUPPORT.overview}
            onExportPdf={exportDashboardPdf}
            exportingPdf={pdfExporting}
          />
        ) : null}

        {pdfError ? (
          <section className="panel error-panel">
            <h3>Export PDF</h3>
            <p>{pdfError}</p>
          </section>
        ) : null}

        {error ? (
          <section className="panel error-panel">
            <h3>Erreur API</h3>
            <p>{error}</p>
          </section>
        ) : null}

        {activeSection === "dashboard" ? (
          <section className="dashboard-section" ref={dashboardCaptureRef}>
            {/* ── Power BI–style dimension nav ── */}
            <DimNav activeDim={activeDim} onDimChange={setActiveDim} allowedDims={allowedDims} />

            {/* ── Active dimension content ── */}
            {renderDimContent()}
          </section>
        ) : activeSection === "mlops" ? (
          <section className="mlops-section-layout">
            <MLOpsContent />
          </section>
        ) : activeSection === "admin" ? (
          <AdminPanel />
        ) : activeSection === "alerts" ? (
          <AlertsPanel branch={filters.branch} />
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

        {loading && dashboard && activeSection === "dashboard" && activeDim === "overview" ? (
          <section className="panel loading-panel">
            <p>Chargement en cours...</p>
          </section>
        ) : null}
      </section>
    </main>
  );
}

export default function IndexPage() {
  const { isAuthenticated, ready } = useAuth();

  if (!ready) return null;
  if (!isAuthenticated) return <LandingPage />;
  return <DashboardPage />;
}