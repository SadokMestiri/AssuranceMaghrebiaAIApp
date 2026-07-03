import React, { useState } from 'react';

// Options – adjust to match your data dictionaries
const BRANCHE_OPTIONS = [
  { value: 1, label: 'AUTO' },
  { value: 2, label: 'IRDS' },
  { value: 3, label: 'SANTE' },
];

const PERIODICITE_OPTIONS = [
  { value: 'A', label: 'Annuelle' },
  { value: 'S', label: 'Semestrielle' },
  { value: 'T', label: 'Trimestrielle' },
  { value: 'C', label: 'Mensuelle' },
];

const TYPE_POLICE_OPTIONS = [
  { value: 'E', label: 'E' },
  { value: 'B', label: 'B' },
  { value: 'C', label: 'C' },
];

const TYPE_PERSONNE_OPTIONS = [
  { value: 'P', label: 'Particulier' },
  { value: 'E', label: 'Entreprise' },
];

const SEXE_OPTIONS = [
  { value: 'M', label: 'Masculin' },
  { value: 'F', label: 'Féminin' },
];

const CODE_USAGE_OPTIONS = [
  { value: 'U1', label: 'Usage 1' },
  { value: 'U2', label: 'Usage 2' },
];

/**
 * ImpayeRiskSimulator – premium UI for the "Risque Impayé" tab.
 * It sends a POST request to `/ml/predict-notebook` and displays the response.
 */
export default function ImpayeRiskSimulator() {
  // ----- state for each feature ------------------------------------------------
  const [logMtPtt, setLogMtPtt] = useState('');
  const [periodicite, setPeriodicite] = useState('A');
  const [moisEmission, setMoisEmission] = useState('');
  const [trimestreEmission, setTrimestreEmission] = useState('');
  const [jourSemaine, setJourSemaine] = useState('');
  const [flagFinMois, setFlagFinMois] = useState(false);
  const [flagAvn, setFlagAvn] = useState(false);
  const [moisEcheance, setMoisEcheance] = useState('');
  const [bonusMalus, setBonusMalus] = useState('');
  const [idBranche, setIdBranche] = useState(1);
  const [brancheEnc, setBrancheEnc] = useState('');
  const [typePoliceEnc, setTypePoliceEnc] = useState('E');
  const [policeNbImpayesHist, setPoliceNbImpayesHist] = useState('');
  const [policeTauxImpayesHist, setPoliceTauxImpayesHist] = useState('');
  const [policeSeqQuit, setPoliceSeqQuit] = useState('');
  const [policeNbAnnulations, setPoliceNbAnnulations] = useState('');
  const [policeNbSinistres, setPoliceNbSinistres] = useState('');
  const [policeSpRatio, setPoliceSpRatio] = useState('');
  const [clientNbImpayes, setClientNbImpayes] = useState('');
  const [clientMtImpayeTot, setClientMtImpayeTot] = useState('');
  const [ageClient, setAgeClient] = useState('');
  const [typePersonneEnc, setTypePersonneEnc] = useState('P');
  const [sexeEnc, setSexeEnc] = useState('M');
  const [puissance, setPuissance] = useState('');
  const [logValeurVeh, setLogValeurVeh] = useState('');
  const [codeUsage, setCodeUsage] = useState('U1');
  const [mtCommission, setMtCommission] = useState('');
  const [mtFga, setMtFga] = useState('');
  const [mtTimbre, setMtTimbre] = useState('');

  // ----- result handling ------------------------------------------------------
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const buildPayload = () => ({
    LOG_MT_PTT: Number(logMtPtt),
    PERIODICITE_RISK: periodicite,
    MOIS_EMISSION: Number(moisEmission),
    TRIMESTRE_EMISSION: Number(trimestreEmission),
    JOUR_SEMAINE: Number(jourSemaine),
    FLAG_FIN_MOIS: flagFinMois ? 1 : 0,
    FLAG_AVN: flagAvn ? 1 : 0,
    MOIS_ECHEANCE: Number(moisEcheance),
    BONUS_MALUS: Number(bonusMalus),
    ID_BRANCHE: Number(idBranche),
    BRANCHE_ENC: brancheEnc,
    TYPE_POLICE_ENC: typePoliceEnc,
    POLICE_NB_IMPAYES_HIST: Number(policeNbImpayesHist),
    POLICE_TAUX_IMPAYE_HIST: Number(policeTauxImpayesHist),
    POLICE_SEQ_QUIT: Number(policeSeqQuit),
    POLICE_NB_ANNULATIONS: Number(policeNbAnnulations),
    POLICE_NB_SINISTRES: Number(policeNbSinistres),
    POLICE_SP_RATIO: Number(policeSpRatio),
    CLIENT_NB_IMPAYES: Number(clientNbImpayes),
    CLIENT_MT_IMPAYE_TOT: Number(clientMtImpayeTot),
    AGE_CLIENT: Number(ageClient),
    TYPE_PERSONNE_ENC: typePersonneEnc,
    SEXE_ENC: sexeEnc,
    PUISSANCE: Number(puissance),
    LOG_VALEUR_VEH: Number(logValeurVeh),
    CODE_USAGE: codeUsage,
    MT_COMMISSION: Number(mtCommission),
    MT_FGA: Number(mtFga),
    MT_TIMBRE: Number(mtTimbre),
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch(`${API}/api/v1/ml/predict-notebook`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(buildPayload()),
        });
      if (!resp.ok) throw new Error(`API error ${resp.status}`);
      const data = await resp.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Unexpected error');
    } finally {
      setLoading(false);
    }
  };

  // ----- UI -------------------------------------------------------------------
  return (
    <section className="min-h-screen bg-gradient-to-br from-gray-900 via-indigo-900 to-black p-8 flex flex-col items-center">
      <h1 className="text-4xl font-extrabold text-white mb-8 drop-shadow-lg">Simulateur Risque Impayé</h1>
      <form onSubmit={handleSubmit} className="w-full max-w-4xl bg-white/10 backdrop-blur-md rounded-xl p-8 shadow-xl space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Numeric inputs */}
          <div>
            <label className="block text-sm font-medium text-gray-200 mb-1">Log MT PTT</label>
            <input type="number" value={logMtPtt} onChange={e => setLogMtPtt(e.target.value)}
              className="w-full rounded-md bg-gray-800/60 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 p-2" required />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-200 mb-1">Mois d’émission</label>
            <input type="number" min="1" max="12" value={moisEmission} onChange={e => setMoisEmission(e.target.value)}
              className="w-full rounded-md bg-gray-800/60 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 p-2" required />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-200 mb-1">Trimestre d’émission</label>
            <input type="number" min="1" max="4" value={trimestreEmission} onChange={e => setTrimestreEmission(e.target.value)}
              className="w-full rounded-md bg-gray-800/60 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 p-2" required />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-200 mb-1">Jour de la semaine (0‑6)</label>
            <input type="number" min="0" max="6" value={jourSemaine} onChange={e => setJourSemaine(e.target.value)}
              className="w-full rounded-md bg-gray-800/60 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 p-2" required />
          </div>
          {/* Boolean flags */}
          <div className="flex items-center space-x-3">
            <input type="checkbox" checked={flagFinMois} onChange={e => setFlagFinMois(e.target.checked)}
              className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded" />
            <label className="text-sm font-medium text-gray-200">Fin du mois</label>
          </div>
          <div className="flex items-center space-x-3">
            <input type="checkbox" checked={flagAvn} onChange={e => setFlagAvn(e.target.checked)}
              className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded" />
            <label className="text-sm font-medium text-gray-200">Flag AVN</label>
          </div>
          {/* Categorical selectors */}
          <div>
            <label className="block text-sm font-medium text-gray-200 mb-1">Périodicité</label>
            <select value={periodicite} onChange={e => setPeriodicite(e.target.value)}
              className="w-full rounded-md bg-gray-800/60 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 p-2">
              {PERIODICITE_OPTIONS.map(o => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-200 mb-1">Branche</label>
            <select value={idBranche} onChange={e => setIdBranche(Number(e.target.value))}
              className="w-full rounded-md bg-gray-800/60 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 p-2">
              {BRANCHE_OPTIONS.map(o => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-200 mb-1">Type Police</label>
            <select value={typePoliceEnc} onChange={e => setTypePoliceEnc(e.target.value)}
              className="w-full rounded-md bg-gray-800/60 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 p-2">
              {TYPE_POLICE_OPTIONS.map(o => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>
          {/* Additional numeric fields – repeat pattern as needed */}
          <div>
            <label className="block text-sm font-medium text-gray-200 mb-1">Bonus‑Malus</label>
            <input type="number" step="0.01" value={bonusMalus} onChange={e => setBonusMalus(e.target.value)}
              className="w-full rounded-md bg-gray-800/60 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 p-2" />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-200 mb-1">Mois échéance</label>
            <input type="number" min="1" max="12" value={moisEcheance} onChange={e => setMoisEcheance(e.target.value)}
              className="w-full rounded-md bg-gray-800/60 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 p-2" />
          </div>
        </div>
        <button type="submit" disabled={loading}
          className="w-full py-3 mt-4 text-lg font-semibold text-white bg-indigo-600 hover:bg-indigo-700 rounded-md transition-colors disabled:opacity-50">
          {loading ? 'Calcul en cours…' : 'Calculer le risque'}
        </button>
        {error && <p className="text-red-400 mt-2">{error}</p>}
      </form>

      {result && (
        <div className="mt-8 w-full max-w-4xl bg-white/10 backdrop-blur-md rounded-xl p-6 shadow-lg">
          <h2 className="text-2xl font-bold text-white mb-4">Résultat</h2>
          <p className="text-white">Score de risque : <span className="font-mono text-lg">{result.risk_score ?? result.risk_score}</span></p>
          <p className="text-white">Label : <span className="font-medium">{result.risk_label}</span></p>
          <p className="text-white">Prime technique (€/an) : <span className="font-mono">{result.prime_technique?.toLocaleString()}</span></p>
          {result.top_features && (
            <div className="mt-4">
              <h3 className="text-white font-medium mb-2">Features les plus influentes</h3>
              <div className="space-y-2">
                {result.top_features.map((f) => (
                  <div key={f.name} className="flex items-center">
                    <span className="w-32 text-sm text-gray-200 truncate" title={f.name}>{f.name}</span>
                    <div className="flex-1 h-4 bg-gray-700 rounded">
                      <div className="h-4 bg-indigo-500 rounded" style={{ width: `${Math.min(Math.abs(f.importance) * 100, 100)}%` }}></div>
                    </div>
                    <span className="ml-2 text-xs text-gray-300">{f.importance.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
