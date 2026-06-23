import { defineConfig } from "vitest/config";

// jsdom gives detectTheme() a real document/window to read host markers from.
export default defineConfig({
  test: {
    environment: "jsdom",
    include: ["js/**/*.test.ts", "web/src/**/*.test.ts"],
  },
});
