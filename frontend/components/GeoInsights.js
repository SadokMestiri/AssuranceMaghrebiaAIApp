import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export default function GeoInsights({ sinistresByGov, topZones }) {
  return (
    <div className="geo-insights-grid">
      <article className="panel chart-panel">
        <h3>Sinistres proxy par gouvernorat</h3>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={sinistresByGov}>
            <CartesianGrid strokeDasharray="4 4" stroke="rgba(0, 74, 141, 0.2)" />
            <XAxis dataKey="gouvernorat" tick={{ fontSize: 11 }} interval={0} angle={-25} textAnchor="end" height={70} />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip />
            <Bar dataKey="total_sinistres_proxy" fill="#1B68B2" radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </article>

      <article className="panel table-panel">
        <h3>Top zones risque</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Rang</th>
                <th>Gouvernorat</th>
                <th>Score</th>
                <th>Taux proxy</th>
              </tr>
            </thead>
            <tbody>
              {(topZones || []).map((item) => (
                <tr key={`${item.rang}-${item.gouvernorat}`}>
                  <td>{item.rang}</td>
                  <td>{item.gouvernorat}</td>
                  <td>{Number(item.score_risque || 0).toLocaleString("fr-TN")}</td>
                  <td>{Number(item.taux_sinistres_proxy_sur_pnet_pct || 0).toLocaleString("fr-TN")} %</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </article>
    </div>
  );
}
