import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../proofline/ui/static",
    emptyOutDir: true,
  },
});

