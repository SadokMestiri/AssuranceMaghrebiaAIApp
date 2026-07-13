import "leaflet/dist/leaflet.css";
import "../styles/globals.css";
import { AuthProvider } from "../contexts/AuthContext";
import { FilterProvider } from "../contexts/FilterContext";

export default function App({ Component, pageProps }) {
  return (
    <AuthProvider>
      <FilterProvider>
        <Component {...pageProps} />
      </FilterProvider>
    </AuthProvider>
  );
}
