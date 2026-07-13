import { useState } from "react";
import { Eye, EyeOff, LogIn, UserPlus, X, Sparkles, Radar, ShieldCheck, LineChart } from "lucide-react";
import { KeyrusMark } from "./KeyrusLogo";
import { useAuth } from "../contexts/AuthContext";

// "ceo"/"admin" are deliberately excluded from self-registration — see
// backend/auth.py SELF_REGISTER_ROLES, which the backend also enforces.
const ROLE_OPTIONS = [
  { key: "agent",     label: "Agent commercial" },
  { key: "sinistres", label: "Gestionnaire Sinistres" },
  { key: "analyst",   label: "Data Analyst / MLOps" },
];

const HIGHLIGHTS = [
  { icon: LineChart,   title: "Pilotage temps réel",   text: "Primes, résiliation, sinistralité, impayés — un seul écran." },
  { icon: Sparkles,    title: "Agent IA métier",        text: "Questions en langage naturel, comparaisons, exports." },
  { icon: Radar,       title: "Cartographie du risque", text: "Réseau de distribution et zones à risque en un coup d'œil." },
  { icon: ShieldCheck, title: "Gouvernance ML",         text: "Dérive de modèles, alertes, ré-entraînement suivi." },
];

const initialForm = { email: "", password: "", nom: "", prenom: "", role: "agent" };

export default function LandingPage() {
  const { login, register } = useAuth();
  const [mode, setMode]           = useState(null); // null | "login" | "register"
  const [form, setForm]           = useState(initialForm);
  const [error, setError]         = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const update = (key) => (e) => setForm((prev) => ({ ...prev, [key]: e.target.value }));

  const openMode = (next) => {
    setMode(next);
    setError("");
  };

  const closeModal = () => {
    setMode(null);
    setError("");
    setForm(initialForm);
  };

  const submit = async (e) => {
    e.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      if (mode === "login") {
        await login(form.email, form.password);
      } else {
        await register(form);
      }
    } catch (err) {
      setError(err.message || "Une erreur est survenue.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="landing-page">
      <header className="landing-topbar">
        <KeyrusMark className="landing-topbar-logo" />
        <nav className="landing-topbar-actions">
          <button type="button" className="landing-link-btn" onClick={() => openMode("login")}>
            <LogIn size={15} /> Se connecter
          </button>
          <button type="button" className="landing-btn landing-btn-primary landing-btn-compact" onClick={() => openMode("register")}>
            <UserPlus size={15} /> S&apos;inscrire
          </button>
        </nav>
      </header>

      <main className="landing-hero">
        <p className="landing-kicker">Keyrus Control Center</p>
        <h1>Placer l&apos;intelligence au cœur de votre portefeuille.</h1>
        <p className="landing-tagline">
          Pilotage exécutif, réseau commercial, sinistres et intelligence artificielle
          réunis sur une seule plateforme.
        </p>

        <div className="landing-highlights">
          {HIGHLIGHTS.map(({ icon: Icon, title, text }) => (
            <article key={title} className="landing-highlight-card">
              <Icon size={20} />
              <h3>{title}</h3>
              <p>{text}</p>
            </article>
          ))}
        </div>
      </main>

      {mode && (
        <div className="landing-modal-overlay" onClick={closeModal}>
          <div className="landing-modal" onClick={(e) => e.stopPropagation()}>
            <button type="button" className="landing-modal-close" onClick={closeModal} aria-label="Fermer">
              <X size={18} />
            </button>

            <form className="landing-form" onSubmit={submit}>
              <h2>{mode === "login" ? "Se connecter" : "Créer un compte"}</h2>

              {mode === "register" && (
                <>
                  <div className="landing-field-row">
                    <label>
                      <span>Prénom</span>
                      <input value={form.prenom} onChange={update("prenom")} required />
                    </label>
                    <label>
                      <span>Nom</span>
                      <input value={form.nom} onChange={update("nom")} required />
                    </label>
                  </div>
                  <label>
                    <span>Rôle</span>
                    <select value={form.role} onChange={update("role")}>
                      {ROLE_OPTIONS.map((r) => (
                        <option key={r.key} value={r.key}>{r.label}</option>
                      ))}
                    </select>
                  </label>
                </>
              )}

              <label>
                <span>Email</span>
                <input type="email" value={form.email} onChange={update("email")} required />
              </label>

              <label>
                <span>Mot de passe</span>
                <div className="landing-password-wrap">
                  <input
                    type={showPassword ? "text" : "password"}
                    value={form.password}
                    onChange={update("password")}
                    required
                    minLength={mode === "register" ? 8 : undefined}
                  />
                  <button
                    type="button"
                    className="landing-password-toggle"
                    onClick={() => setShowPassword((v) => !v)}
                    tabIndex={-1}
                  >
                    {showPassword ? <EyeOff size={15} /> : <Eye size={15} />}
                  </button>
                </div>
                {mode === "register" && <span className="landing-hint">8 caractères minimum</span>}
              </label>

              {error && <p className="landing-error">{error}</p>}

              <button type="submit" className="landing-btn landing-btn-primary" disabled={submitting}>
                {submitting ? "Patientez..." : mode === "login" ? "Se connecter" : "Créer mon compte"}
              </button>

              <p className="landing-switch">
                {mode === "login" ? (
                  <>Pas encore de compte ? <button type="button" onClick={() => openMode("register")}>S&apos;inscrire</button></>
                ) : (
                  <>Déjà un compte ? <button type="button" onClick={() => openMode("login")}>Se connecter</button></>
                )}
              </p>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
