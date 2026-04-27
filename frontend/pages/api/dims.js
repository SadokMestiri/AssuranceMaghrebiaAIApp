import fs from "fs";
import path from "path";
import { parse } from "csv-parse/sync";

const DATA_DIR = path.join(process.cwd(), "data", "raw");

function readCsv(filename) {
  const filePath = path.join(DATA_DIR, filename);
  if (!fs.existsSync(filePath)) return [];
  const content = fs.readFileSync(filePath, "utf-8").replace(/^\uFEFF/, "");
  return parse(content, { columns: true, skip_empty_lines: true, trim: true });
}

function countBy(arr, key, normalizer) {
  const map = {};
  for (const item of arr) {
    const raw = item[key] || "N/A";
    const k = normalizer ? normalizer(raw) : raw;
    map[k] = (map[k] || 0) + 1;
  }
  return Object.entries(map)
    .map(([label, count]) => ({ label, count }))
    .sort((a, b) => b.count - a.count);
}

function sumBy(arr, groupKey, sumKey) {
  const map = {};
  for (const item of arr) {
    const k = item[groupKey] || "N/A";
    map[k] = (map[k] || 0) + Number(item[sumKey] || 0);
  }
  return Object.entries(map)
    .map(([label, value]) => ({ label, value }))
    .sort((a, b) => b.value - a.value);
}

// ── CLIENT ───────────────────────────────────────────────────────
function buildClientData(clients, polices) {
  const normalizeType = (t) => {
    const s = String(t).trim().toUpperCase();
    return s === "P" || s === "M" ? s : "N/A";
  };
  const normalizeNatp = (n) => {
    const labels = { R: "Résident", C: "Citoyen", P: "Permanent", S: "Sans papiers" };
    return labels[n] || n || "N/A";
  };

  const sexeCounts = countBy(clients, "SEXE");
  const typeCounts = countBy(clients, "TYPE_PERSONNE", normalizeType).filter(
    (e) => e.label !== "N/A"
  );
  const natpCounts = countBy(clients, "NATP", normalizeNatp).filter((e) => e.label !== "N/A");

  const ageBuckets = { "<25": 0, "25-34": 0, "35-44": 0, "45-54": 0, "55-64": 0, "65+": 0 };
  let ageSum = 0, ageCount = 0;
  const currentYear = 2024;
  for (const c of clients) {
    const dob = c.DATE_NAISSANCE || "";
    if (dob.length >= 4) {
      try {
        const y = parseInt(dob.slice(0, 4), 10);
        const age = currentYear - y;
        if (age > 0 && age < 120) {
          ageSum += age;
          ageCount++;
          if (age < 25) ageBuckets["<25"]++;
          else if (age < 35) ageBuckets["25-34"]++;
          else if (age < 45) ageBuckets["35-44"]++;
          else if (age < 55) ageBuckets["45-54"]++;
          else if (age < 65) ageBuckets["55-64"]++;
          else ageBuckets["65+"]++;
        }
      } catch {}
    }
  }

  const cityCounts = countBy(clients, "VILLE");
  const topVilles = cityCounts.slice(0, 10);

  // Only count clients with SEXE F or M for percentage calculations
  const filteredClients = clients.filter((c) => c.SEXE === "F" || c.SEXE === "M");
  const nbF = filteredClients.filter((c) => c.SEXE === "F").length;
  const nbM = filteredClients.filter((c) => c.SEXE === "M").length;
  const total = filteredClients.length;

  // Polices résiliées par sexe
  const clientSexeMap = {};
  for (const c of clients) clientSexeMap[c.ID_CLIENT] = c.SEXE || "N/A";

  const churnBySexe = { F: { total: 0, resiliees: 0 }, M: { total: 0, resiliees: 0 } };
  for (const p of polices) {
    const sexe = clientSexeMap[p.ID_CLIENT];
    if (sexe !== "F" && sexe !== "M") continue;
    churnBySexe[sexe].total++;
    if (p.SITUATION === "R") churnBySexe[sexe].resiliees++;
  }

  const uniqueVilles = new Set(clients.map((c) => c.VILLE).filter(Boolean));

  return {
    kpis: {
      total,
      nb_f: nbF,
      nb_m: nbM,
      pct_f: total > 0 ? (nbF / total) * 100 : 0,
      pct_m: total > 0 ? (nbM / total) * 100 : 0,
      pct_moral: (clients.filter((c) => normalizeType(c.TYPE_PERSONNE) === "M").length / (clients.length || 1)) * 100,
      nb_villes: uniqueVilles.size,
      age_moyen: ageCount > 0 ? Math.round(ageSum / ageCount) : 0,
    },
    sexe: sexeCounts.filter((e) => e.label !== "N/A"),
    typePersonne: typeCounts,
    natp: natpCounts,
    ageTranches: Object.entries(ageBuckets).map(([label, count]) => ({ label, count })),
    topVilles,
    churnBySexe: ["F", "M"].map((s) => ({
      label: s === "F" ? "Femmes" : "Hommes",
      total: churnBySexe[s].total,
      resiliees: churnBySexe[s].resiliees,
    })),
  };
}

// ── AGENT ────────────────────────────────────────────────────────
function buildAgentData(agents, polices, emissions) {
  const groupeCounts = countBy(agents, "GROUPE_AGENT");
  const typeCounts   = countBy(agents, "TYPE_AGENT");
  const etatCounts   = countBy(agents, "ETAT_AGENT");
  const localiteCounts = countBy(agents, "LOCALITE_AGENT");

  const agentMap = {};
  for (const a of agents) agentMap[a.ID_AGENT] = a;

  const agentPnet = {};
  const agentQuitt = {};
  for (const e of emissions) {
    const aid = e.ID_AGENT;
    agentPnet[aid]  = (agentPnet[aid]  || 0) + Number(e.MT_PNET  || 0);
    agentQuitt[aid] = (agentQuitt[aid] || 0) + 1;
  }

  const agentPolices = {};
  for (const p of polices) {
    agentPolices[p.ID_AGENT] = (agentPolices[p.ID_AGENT] || 0) + 1;
  }

  const agentIds = Object.keys(agentMap);
  const actifs   = agents.filter((a) => a.ETAT_AGENT === "A").length;
  const inactifs  = agents.length - actifs;

  const totalPnet = Object.values(agentPnet).reduce((s, v) => s + v, 0);

  const topAgentsPnet = Object.entries(agentPnet)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([id, pnet]) => {
      const a = agentMap[id] || {};
      return { nom: `${a.NOM_AGENT || "?"} ${(a.PRENOM_AGENT || "").trim()}`.trim(), pnet };
    });

  const topAgentsPolices = Object.entries(agentPolices)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([id, nb_polices]) => {
      const a = agentMap[id] || {};
      return { nom: `${a.NOM_AGENT || "?"} ${(a.PRENOM_AGENT || "").trim()}`.trim(), nb_polices };
    });

  const uniqueLocalites = new Set(agents.map((a) => a.LOCALITE_AGENT).filter(Boolean));

  return {
    kpis: {
      total: agents.length,
      actifs,
      inactifs,
      pct_actifs: (actifs / agents.length) * 100,
      nb_groupes: new Set(agents.map((a) => a.GROUPE_AGENT)).size,
      nb_localites: uniqueLocalites.size,
      avg_pnet: actifs > 0 ? totalPnet / actifs : 0,
    },
    groupes:         groupeCounts,
    typeAgent:       typeCounts,
    etat:            etatCounts,
    topAgentsPnet,
    topAgentsPolices,
    localites:       localiteCounts,
  };
}

// ── PRODUIT ──────────────────────────────────────────────────────
function buildProduitData(produits, emissions) {
  const prodMap = {};
  for (const p of produits) prodMap[p.CODE_PRODUIT] = p;

  const prodPnet  = {};
  const prodCount = {};
  for (const e of emissions) {
    const code = e.CODE_PRODUIT;
    prodPnet[code]  = (prodPnet[code]  || 0) + Number(e.MT_PNET || 0);
    prodCount[code] = (prodCount[code] || 0) + 1;
  }

  const branchePnet  = {};
  const famillePnet  = {};
  for (const [code, pnet] of Object.entries(prodPnet)) {
    const p = prodMap[code] || {};
    const b = p.BRANCHE || "N/A";
    const f = p.FAMILLE_RISQUE || "N/A";
    branchePnet[b] = (branchePnet[b] || 0) + pnet;
    famillePnet[f] = (famillePnet[f] || 0) + pnet;
  }

  const topProduits = Object.entries(prodPnet)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([code, pnet]) => {
      const p = prodMap[code] || {};
      return { label: p.LIB_PRODUIT || code, pnet, branche: p.BRANCHE || "N/A" };
    });

  const topProduitsQuittances = Object.entries(prodCount)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([code, count]) => {
      const p = prodMap[code] || {};
      return { label: p.LIB_PRODUIT || code, count, branche: p.BRANCHE || "N/A" };
    });

  const totalPnet = Object.values(prodPnet).reduce((s, v) => s + v, 0);
  const totalQuitt = Object.values(prodCount).reduce((s, v) => s + v, 0);
  const topEntry = topProduits[0] || {};

  return {
    kpis: {
      nb_produits: produits.length,
      nb_familles: new Set(produits.map((p) => p.FAMILLE_RISQUE)).size,
      nb_branches: new Set(produits.map((p) => p.BRANCHE)).size,
      total_pnet: totalPnet,
      total_quitt: totalQuitt,
      top_produit: topEntry.label || "-",
      top_pnet: topEntry.pnet || 0,
    },
    byBranche: Object.entries(branchePnet).map(([label, pnet]) => ({ label, pnet })).sort((a, b) => b.pnet - a.pnet),
    byFamille: Object.entries(famillePnet).map(([label, pnet]) => ({ label, pnet })).sort((a, b) => b.pnet - a.pnet),
    topProduits,
    topProduitsQuittances,
  };
}

// ── VEHICULE ─────────────────────────────────────────────────────
function buildVehiculeData(vehicules) {
  const marque  = countBy(vehicules, "MARQUE");
  const genre   = countBy(vehicules, "GENRE_VEHICULE");
  const typeV   = countBy(vehicules, "TYPE_VEHICULE");

  const puissanceBuckets = { "≤6CV": 0, "7-9CV": 0, "10-12CV": 0, "13+CV": 0 };
  let puissSum = 0, puissCount = 0;
  let ageSum = 0, ageCount = 0;
  const curYear = 2024;

  for (const v of vehicules) {
    const p = parseFloat(v.PUISSANCE || 0);
    if (!isNaN(p) && p > 0) {
      if (p <= 6) puissanceBuckets["≤6CV"]++;
      else if (p <= 9) puissanceBuckets["7-9CV"]++;
      else if (p <= 12) puissanceBuckets["10-12CV"]++;
      else puissanceBuckets["13+CV"]++;
      puissSum += p;
      puissCount++;
    }
    const mec = v.DATE_MEC || "";
    if (mec.length >= 4) {
      try {
        const y = parseInt(mec.slice(0, 4), 10);
        if (y > 1980 && y <= curYear) { ageSum += curYear - y; ageCount++; }
      } catch {}
    }
  }

  const nbVp = vehicules.filter((v) => v.GENRE_VEHICULE === "VP").length;
  const nbVu = vehicules.filter((v) => ["VU", "PL", "TC"].includes(v.GENRE_VEHICULE)).length;

  return {
    kpis: {
      total: vehicules.length,
      nb_vp: nbVp,
      nb_vu: nbVu,
      pct_vp: (nbVp / vehicules.length) * 100,
      nb_marques: new Set(vehicules.map((v) => v.MARQUE).filter(Boolean)).size,
      avg_puissance: puissCount > 0 ? Math.round(puissSum / puissCount) : 0,
      avg_age: ageCount > 0 ? Math.round(ageSum / ageCount) : 0,
    },
    byGenre:    genre,
    byPuissance: Object.entries(puissanceBuckets).map(([label, count]) => ({ label, count })),
    topMarques:  marque.slice(0, 10),
    byType:      typeV.slice(0, 8),
  };
}

// ── POLICE ───────────────────────────────────────────────────────
function buildPoliceData(polices) {
  const bySituation  = countBy(polices, "SITUATION");
  const byType       = countBy(polices, "TYPE_POLICE");
  const byPeriodicite = countBy(polices, "PERIODICITE");
  const byDuree      = countBy(polices, "DUREE");

  const bmLabels = ["0-5", "6-8", "9-10", "11-12", "13+"];
  const bmBuckets = { "0-5": 0, "6-8": 0, "9-10": 0, "11-12": 0, "13+": 0 };
  let bmSum = 0, bmCount = 0;
  for (const p of polices) {
    const bm = parseFloat(p.BONUS_MALUS || 0);
    if (!isNaN(bm) && bm > 0) {
      bmSum += bm; bmCount++;
      if (bm <= 5) bmBuckets["0-5"]++;
      else if (bm <= 8) bmBuckets["6-8"]++;
      else if (bm <= 10) bmBuckets["9-10"]++;
      else if (bm <= 12) bmBuckets["11-12"]++;
      else bmBuckets["13+"]++;
    }
  }

  const total = polices.length;
  const enVigueur  = polices.filter((p) => p.SITUATION === "V").length;
  const resiliees  = polices.filter((p) => p.SITUATION === "R").length;
  const individuelles = polices.filter((p) => p.TYPE_POLICE === "individuel").length;
  const flottes    = polices.filter((p) => p.TYPE_POLICE === "flotte").length;

  const dureeLabels = { R: "Renouvelable", F: "Ferme", S: "STR" };

  return {
    kpis: {
      total,
      en_vigueur: enVigueur,
      resiliees,
      pct_vigueur: (enVigueur / total) * 100,
      pct_resiliees: (resiliees / total) * 100,
      individuelles,
      flottes,
      pct_indiv: (individuelles / total) * 100,
      pct_flotte: (flottes / total) * 100,
      avg_bm: bmCount > 0 ? bmSum / bmCount : 0,
    },
    bySituation,
    byType,
    byPeriodicite,
    byDuree: byDuree.map((e) => ({ ...e, label: dureeLabels[e.label] || e.label })),
    bonusMalus: bmLabels.map((label) => ({ label, count: bmBuckets[label] })),
  };
}

// ── SINISTRE ─────────────────────────────────────────────────────
function buildSinistreData(sinistres) {
  const byNature        = countBy(sinistres, "NATURE_SINISTRE").slice(0, 10);
  const byEtat          = countBy(sinistres, "ETAT_SINISTRE");
  const byResponsabilite = countBy(sinistres, "RESPONSABILITE");
  const byBranche       = countBy(sinistres, "BRANCHE");

  const monthly = {};
  for (const s of sinistres) {
    const y = s.ANNEE_SURVENANCE || "";
    const m = String(s.MOIS_SURVENANCE || "").padStart(2, "0");
    if (y && m) {
      const key = `${y}-${m}`;
      monthly[key] = (monthly[key] || 0) + 1;
    }
  }
  const monthlyData = Object.entries(monthly)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([label, count]) => ({ label, count }));

  const total     = sinistres.length;
  const ouverts   = sinistres.filter((s) => s.ETAT_SINISTRE === "Ouvert").length;
  const clos      = sinistres.filter((s) => s.ETAT_SINISTRE === "Clos").length;
  const totalEval = sinistres.reduce((s, x) => s + Number(x.MT_EVALUATION || 0), 0);
  const totalPaye = sinistres.reduce((s, x) => s + Number(x.MT_PAYE || 0), 0);
  const nbMateriel = sinistres.filter((s) => s.NATURE_SINISTRE === "Matériel").length;

  return {
    kpis: {
      total,
      ouverts,
      clos,
      pct_ouverts: (ouverts / total) * 100,
      pct_clos: (clos / total) * 100,
      total_eval: totalEval,
      total_paye: totalPaye,
      taux_paiement: totalEval > 0 ? (totalPaye / totalEval) * 100 : 0,
      nb_materiel: nbMateriel,
    },
    byNature,
    byEtat,
    byResponsabilite,
    byBranche,
    monthly: monthlyData,
  };
}

// ── HANDLER ──────────────────────────────────────────────────────
export default function handler(req, res) {
  try {
    const clients   = readCsv("DIM_CLIENT.csv");
    const agents    = readCsv("DIM_AGENT.csv");
    const polices   = readCsv("DIM_POLICE.csv");
    const produits  = readCsv("DIM_PRODUIT.csv");
    const vehicules = readCsv("DIM_VEHICULE.csv");
    const emissions = readCsv("DWH_FACT_EMISSION.csv");
    const sinistres = readCsv("DWH_FACT_SINISTRE.csv");

    res.status(200).json({
      clients:   buildClientData(clients, polices),
      agents:    buildAgentData(agents, polices, emissions),
      produits:  buildProduitData(produits, emissions),
      vehicules: buildVehiculeData(vehicules),
      polices:   buildPoliceData(polices),
      sinistres: buildSinistreData(sinistres),
    });
  } catch (err) {
    res.status(500).json({ error: String(err.message || err) });
  }
}
