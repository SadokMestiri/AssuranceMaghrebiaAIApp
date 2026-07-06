import { CircleMarker, MapContainer, Popup, TileLayer } from "react-leaflet";

const fmt = new Intl.NumberFormat("fr-TN");
const fmtTND = new Intl.NumberFormat("fr-TN", { style: "currency", currency: "TND", maximumFractionDigits: 0 });

function buildColorFn(points) {
  const scores = (points || []).map((p) => Number(p.score_risque || 0)).sort((a, b) => a - b);
  const n = scores.length;
  if (n === 0) return () => "#004A8D";
  const p33 = scores[Math.floor(n * 0.33)] ?? 0;
  const p66 = scores[Math.floor(n * 0.66)] ?? 0;
  const p90 = scores[Math.floor(n * 0.90)] ?? 0;
  return (score) => {
    if (score >= p90) return "#b91c1c";
    if (score >= p66) return "#ea580c";
    if (score >= p33) return "#d97706";
    return "#16a34a";
  };
}

export default function CarteWidget({ points }) {
  const colorByRisk = buildColorFn(points);

  return (
    <div className="map-shell">
      <MapContainer center={[34.2, 9.5]} zoom={6} className="map-canvas" scrollWheelZoom={false}>
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {(points || []).map((point) => {
          const radius = 6 + Math.min(22, Math.sqrt(Number(point.nb_polices || 0) / 10));
          const riskRate = Number(point.score_risque || 0);

          return (
            <CircleMarker
              key={point.ville}
              center={[Number(point.latitude), Number(point.longitude)]}
              radius={radius}
              pathOptions={{
                color: colorByRisk(riskRate),
                fillColor: colorByRisk(riskRate),
                fillOpacity: 0.55,
                weight: 1.5,
              }}
            >
              <Popup>
                <strong>{point.ville}</strong>
                <br />
                Polices : {fmt.format(point.nb_polices || 0)}
                <br />
                Sinistres : {fmt.format(point.nb_sinistres || 0)}
                <br />
                Montant payé : {fmtTND.format(point.total_mt_paye || 0)}
                <br />
                Taux S/P : {Number(point.taux_sinistres_pct || 0).toFixed(1)} %
                <br />
                Score risque : {Number(point.score_risque || 0).toFixed(1)}
              </Popup>
            </CircleMarker>
          );
        })}
      </MapContainer>
    </div>
  );
}
