import { createRoot } from "react-dom/client";
// HashRouter (not BrowserRouter): the single-file artifact opens over file:// (origin "null"),
// where History pushState/replaceState throw SecurityError. Hash routing (#/browse) works on
// both file:// and http with no server-side route handling.
import { HashRouter } from "react-router-dom";
import App from "./App";

// No backend: the quantem.live Browse code opens a few EventSource streams
// (/api/gpu/stream, /api/browse/cache-stream) that would otherwise retry-loop
// forever against a server that isn't there. Replace EventSource with an inert
// stub so those live-status streams quietly do nothing in the standalone app.
class DeadEventSource {
  onmessage: ((e: MessageEvent) => void) | null = null;
  onerror: ((e: Event) => void) | null = null;
  addEventListener() {}
  removeEventListener() {}
  close() {}
}
(window as unknown as { EventSource: unknown }).EventSource = DeadEventSource;

createRoot(document.getElementById("root")!).render(
  <HashRouter>
    <App />
  </HashRouter>
);
