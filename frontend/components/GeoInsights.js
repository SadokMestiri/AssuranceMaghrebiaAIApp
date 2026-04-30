import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

function ScoreBadge({ score }) {
  const pct = Math.min(100, Math.max(0, Number(score || 0)));
  const color =
    pct >= 70 ? "#e53e3e" : pct >= 40 ? "#dd6b20" : "#2b6cb0";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
      <div
        style={{
          width: "80px",
          height: "6px",
          borderRadius: "3px",
          background: "rgba(0,0,0,0.08)",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${pct}%`,
            height: "100%",
            background: color,
            borderRadius: "3px",
            transition: "width 0.4s ease",
          }}
        />
      </div>
      <span style={{ fontWeight: 600, color, fontSize: "13px", minWidth: "36px" }}>
        {pct.toLocaleString("fr-TN", { maximumFractionDigits: 1 })}
      </span>
    </div>
  );
}

function ScoreFormula() {
  return (
    <div
      style={{
        margin: "0 0 12px 0",
        padding: "8px 12px",
        background: "rgba(0, 74, 141, 0.06)",
        borderLeft: "3px solid #1B68B2",
        borderRadius: "0 6px 6px 0",
        fontSize: "12px",
        color: "#444",
        lineHeight: 1.6,
      }}
    >
      <strong style={{ color: "#1B68B2" }}>Score risque</strong> ={" "}
      <span style={{ fontFamily: "monospace" }}>
        0.70 × taux_sinistres/pnet (%)
      </span>{" "}
      +{" "}
      <span style={{ fontFamily: "monospace" }}>
        30 × nb_sinistres / max(nb_sinistres)
      </span>
      <br />
      <span style={{ color: "#888" }}>
        70% intensité financière · 30% volume relatif
      </span>
    </div>
  );
}

export default function GeoInsights({ sinistresByGov, topZones }) {
  return (
    <div className="geo-insights-grid">
      <article className="panel chart-panel">
        <h3>Sinistres par gouvernorat</h3>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={sinistresByGov}>
            <CartesianGrid strokeDasharray="4 4" stroke="rgba(0, 74, 141, 0.2)" />
            <XAxis
              dataKey="gouvernorat"
              tick={{ fontSize: 11 }}
              interval={0}
              angle={-25}
              textAnchor="end"
              height={70}
            />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip
              formatter={(value) => [value.toLocaleString("fr-TN"), "Sinistres"]}
            />
            <Bar dataKey="nb_sinistres" fill="#1B68B2" radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </article>

      <article className="panel table-panel">
        <h3>Top zones à risque</h3>
        <ScoreFormula />
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Rang</th>
                <th>Gouvernorat</th>
                <th>Nb sinistres</th>
                <th>Montant payé</th>
                <th>Taux S/P</th>
                <th>Score risque</th>
              </tr>
            </thead>
            <tbody>
              {(topZones || []).map((item) => (
                <tr key={`${item.rang}-${item.gouvernorat}`}>
                  <td style={{ fontWeight: 700, color: "#1B68B2" }}>
                    #{item.rang}
                  </td>
                  <td>{item.gouvernorat}</td>
                  <td>{Number(item.nb_sinistres || 0).toLocaleString("fr-TN")}</td>
                  <td>
                    {Number(item.total_mt_paye || 0).toLocaleString("fr-TN", {
                      maximumFractionDigits: 0,
                    })}{" "}
                    DT
                  </td>
                  <td>
                    {Number(item.taux_sinistres_sur_pnet_pct || 0).toLocaleString(
                      "fr-TN",
                      { maximumFractionDigits: 1 }
                    )}{" "}
                    %
                  </td>
                  <td>
                    <ScoreBadge score={item.score_risque} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </article>
    </div>
  );
}