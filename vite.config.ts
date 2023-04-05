// @ts-nocheck

import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import viteBasicSslPlugin from "@vitejs/plugin-basic-ssl"

// https://vitejs.dev/config/
export default defineConfig({
  server: {
    port: 3004,
    https: true
  },
  optimizeDeps: {
    include: ['lodash', '@dataloop-ai/components']
  },
  plugins: [
    viteBasicSslPlugin(),
    vue()
  ],
  test: {
      environment: 'jsdom',
      setupFiles: ['tests/setup.js'],
      deps: {
          inline: ['vitest-canvas-mock']
      },
      coverage: {
          reporter: ['lcovonly', 'text']
      }
  }
});
