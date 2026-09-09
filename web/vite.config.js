import { defineConfig } from "vite";
export default defineConfig({
  server: {
    // Preserve the browser host so the local API can validate its Origin.
    proxy: { "/api": { target: "http://127.0.0.1:8787", changeOrigin: false } },
  },
  build: { chunkSizeWarningLimit: 800 },
});
