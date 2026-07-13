// Per-role UI access matrix. Mirrors the roles defined in
// backend/auth.py ROLES. This gates which sections/dims render for the
// signed-in user — it is a UI convenience, not a security boundary: the
// backend APIs are not yet enforcing role checks, so this alone doesn't
// stop a determined user from calling the endpoints directly.
export const ROLE_ACCESS = {
  ceo: {
    sections: ["dashboard", "alerts", "mlops", "agent"],
    dims: ["overview", "clients", "agents", "produits", "vehicules", "polices", "sinistres"],
  },
  admin: {
    sections: ["dashboard", "alerts", "mlops", "agent", "admin"],
    dims: ["overview", "clients", "agents", "produits", "vehicules", "polices", "sinistres"],
  },
  analyst: {
    sections: ["dashboard", "mlops", "agent"],
    dims: ["overview", "clients", "agents", "produits", "vehicules", "polices", "sinistres"],
  },
  sinistres: {
    sections: ["dashboard", "agent"],
    dims: ["sinistres", "clients", "polices"],
  },
  agent: {
    sections: ["dashboard", "agent"],
    dims: ["clients", "produits", "polices", "vehicules"],
  },
};

const DEFAULT_ACCESS = { sections: ["dashboard"], dims: ["overview"] };

export function getAllowedSections(role) {
  return ROLE_ACCESS[role]?.sections || DEFAULT_ACCESS.sections;
}

export function getAllowedDims(role) {
  return ROLE_ACCESS[role]?.dims || DEFAULT_ACCESS.dims;
}
