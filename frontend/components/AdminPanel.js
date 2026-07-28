import { useEffect, useState } from "react";
import { ShieldCheck, ShieldOff, Trash2, AlertTriangle } from "lucide-react";
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
  const [confirmUser, setConfirmUser] = useState(null); // user pending delete confirmation

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
    setError("");
    // Optimistic update — flip the status in the UI right away so the click
    // feels instant, then reconcile with the server response below.
    setUsers((prev) => prev.map((u) => (u.id === id ? { ...u, is_active: isActive } : u)));
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
      // roll back the optimistic change if the request failed
      setUsers((prev) => prev.map((u) => (u.id === id ? { ...u, is_active: !isActive } : u)));
      setError(err.message || "Echec de la mise a jour du statut.");
    } finally {
      setBusyId(null);
    }
  };

  const deleteUser = async (id) => {
    setBusyId(id);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/api/v1/auth/users/${id}`, {
        method: "DELETE",
        headers: authHeaders,
      });
      if (!res.ok) throw new Error((await res.json())?.detail || "");
      setUsers((prev) => prev.filter((u) => u.id !== id));
      setConfirmUser(null);
    } catch (err) {
      setError(err.message || "Echec de la suppression du compte.");
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
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {users.map((u) => {
                const isSelf = u.id === currentUser?.id;
                return (
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
                      disabled={busyId === u.id || isSelf}
                      title={isSelf
                        ? "Vous ne pouvez pas modifier votre propre statut"
                        : (u.is_active ? "Cliquer pour suspendre ce compte" : "Cliquer pour réactiver ce compte")}
                      onClick={() => toggleActive(u.id, !u.is_active)}
                    >
                      {u.is_active ? <ShieldCheck size={14} /> : <ShieldOff size={14} />}
                      {u.is_active ? "Actif" : "Suspendu"}
                    </button>
                  </td>
                  <td>{new Date(u.created_at).toLocaleDateString("fr-TN")}</td>
                  <td>
                    <button
                      type="button"
                      className="admin-delete-btn"
                      disabled={busyId === u.id || isSelf}
                      title={isSelf
                        ? "Vous ne pouvez pas supprimer votre propre compte"
                        : "Supprimer définitivement ce compte"}
                      onClick={() => setConfirmUser(u)}
                    >
                      <Trash2 size={14} />
                      Supprimer
                    </button>
                  </td>
                </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {confirmUser && (
        <div className="admin-confirm-overlay" onClick={() => busyId ? null : setConfirmUser(null)}>
          <div className="admin-confirm-modal" onClick={(e) => e.stopPropagation()}>
            <div className="admin-confirm-icon"><AlertTriangle size={26} /></div>
            <h3>Supprimer le compte ?</h3>
            <p>
              Vous êtes sur le point de supprimer définitivement le compte de{" "}
              <b>{confirmUser.prenom} {confirmUser.nom}</b> ({confirmUser.email}).
              <br />Cette action est <b>irréversible</b>.
            </p>
            <div className="admin-confirm-actions">
              <button
                type="button"
                className="admin-confirm-cancel"
                disabled={busyId === confirmUser.id}
                onClick={() => setConfirmUser(null)}
              >
                Annuler
              </button>
              <button
                type="button"
                className="admin-confirm-delete"
                disabled={busyId === confirmUser.id}
                onClick={() => deleteUser(confirmUser.id)}
              >
                <Trash2 size={14} />
                {busyId === confirmUser.id ? "Suppression…" : "Supprimer définitivement"}
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
