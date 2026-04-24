import { CircleMarker, MapContainer, Popup, TileLayer } from "react-leaflet";

function colorByRisk(rate) {
  if (rate >= 30) {
    return "#b91c1c";
  }
  if (rate >= 15) {
    return "#ea580c";
  }
  if (rate >= 8) {
    return "#d97706";
  }
  return "#004A8D";
}

export default function CarteWidget({ points }) {
  return (
    <div className="map-shell">
      <MapContainer center={[34.2, 9.5]} zoom={6} className="map-canvas" scrollWheelZoom={false}>
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {(points || []).map((point) => {
          const radius = 6 + Math.min(18, Math.sqrt(Number(point.nb_polices || 0)));
          const riskRate = Number(point.taux_sinistres_proxy_pct || 0);

          return (
            <CircleMarker
              key={`${point.id_agent}-${point.latitude}-${point.longitude}`}
              center={[Number(point.latitude), Number(point.longitude)]}
              radius={radius}
              pathOptions={{
                color: colorByRisk(riskRate),
                fillColor: colorByRisk(riskRate),
                fillOpacity: 0.5,
                weight: 1.5,
              }}
            >
              <Popup>
                <strong>{point.gouvernorat}</strong>
                <br />
                Polices: {Number(point.nb_polices || 0).toLocaleString("fr-TN")}
                <br />
                Sinistres proxy: {Number(point.nb_sinistres_proxy || 0).toLocaleString("fr-TN")}
                <br />
                Taux proxy: {Number(point.taux_sinistres_proxy_pct || 0).toLocaleString("fr-TN")} %
              </Popup>
            </CircleMarker>
          );
        })}
      </MapContainer>
    </div>
  );
}
