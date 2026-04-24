import "leaflet/dist/leaflet.css";
import "../styles/globals.css";
import { FilterProvider } from "../contexts/FilterContext";

export default function App({ Component, pageProps }) {
  return (
    <FilterProvider>
      <Component {...pageProps} />
    </FilterProvider>
  );
}
