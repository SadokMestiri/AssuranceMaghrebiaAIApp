import { useEffect, useState } from "react";
import { ShieldCheck, ShieldOff } from "lucide-react";
import { useAuth } from "../contexts/AuthContext";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Full role set (unlike the self-registration dropdown, an admin can grant
// any role, including ceo/admin — see backend/auth_router.py PATCH .../role).
const ALL_ROLES = [
  { key: "ceo",       label: "Direction / CEO" },
  { key: "agent",     label: "Agent commercial" },
  { key: "sinistres", label: "Gestionnaire Sinistres" },
  { key: "analyst",   label: "Data Analyst / MLOps" },
  { key: "admin",     label: "Administrateur" },
];

export default function AdminPanel() {
  const { token, user: currentUser } = useAuth();
  const [users, setUsers]     = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState("");
  const [busyId, setBusyId]   = useState(null);

  const authHeaders = { Authorization: `Bearer ${token}`, "Content-Type": "application/json" };

  const loadUsers = () => {
    setLoading(true);
    setError("");
    fetch(`${API_BASE}/api/v1/auth/users`, { headers: authHeaders })
      .then((res) => (res.ok ? res.json() : Promise.reject(res)))
      .then((data) => setUsers(data.users || []))
      .catch(() => setError("Impossible de charger les utilisateurs."))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    if (token) loadUsers();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  const changeRole = async (id, role) => {
    setBusyId(id);
    try {
      const res = await fetch(`${API_BASE}/api/v1/auth/users/${id}/role`, {
        method: "PATCH",
        headers: authHeaders,
        body: JSON.stringify({ role }),
      });
      if (!res.ok) throw new Error();
      const data = await res.json();
      setUsers((prev) => prev.map((u) => (u.id === id ? { ...u, ...data.user } : u)));
    } catch {
      setError("Echec du changement de role.");
    } finally {
      setBusyId(null);
    }
  };

  const toggleActive = async (id, isActive) => {
    setBusyId(id);
    try {
      const res = await fetch(`${API_BASE}/api/v1/auth/users/${id}/active`, {
        method: "PATCH",
        headers: authHeaders,
        body: JSON.stringify({ is_active: isActive }),
      });
      if (!res.ok) throw new Error((await res.json())?.detail || "");
      const data = await res.json();
      setUsers((prev) => prev.map((u) => (u.id === id ? { ...u, ...data.user } : u)));
    } catch (err) {
      setError(err.message || "Echec de la mise a jour du statut.");
    } finally {
      setBusyId(null);
    }
  };

  return (
    <section className="admin-panel-section">
      <div className="panel-headline">
        <h2>Administration — Utilisateurs</h2>
      </div>

      {error && <p className="admin-error">{error}</p>}

      {loading ? (
        <p className="dim-loading">Chargement des utilisateurs…</p>
      ) : (
        <div className="admin-table-wrapper">
          <table className="admin-table">
            <thead>
              <tr>
                <th>Utilisateur</th>
                <th>Email</th>
                <th>Role</th>
                <th>Statut</th>
                <th>Inscrit le</th>
              </tr>
            </thead>
            <tbody>
              {users.map((u) => (
                <tr key={u.id} className={!u.is_active ? "admin-row-inactive" : ""}>
                  <td>{u.prenom} {u.nom}</td>
                  <td>{u.email}</td>
                  <td>
                    <select
                      value={u.role}
                      disabled={busyId === u.id}
                      onChange={(e) => changeRole(u.id, e.target.value)}
                    >
                      {ALL_ROLES.map((r) => (
                        <option key={r.key} value={r.key}>{r.label}</option>
                      ))}
                    </select>
                  </td>
                  <td>
                    <button
                      type="button"
                      className={`admin-status-btn ${u.is_active ? "active" : "inactive"}`}
                      disabled={busyId === u.id || u.id === currentUser?.id}
                      title={u.id === currentUser?.id ? "Vous ne pouvez pas modifier votre propre statut" : ""}
                      onClick={() => toggleActive(u.id, !u.is_active)}
                    >
                      {u.is_active ? <ShieldCheck size={14} /> : <ShieldOff size={14} />}
                      {u.is_active ? "Actif" : "Desactive"}
                    </button>
                  </td>
                  <td>{new Date(u.created_at).toLocaleDateString("fr-TN")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
