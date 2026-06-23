// Theme auto-detection regression tests. Each environment (VS Code, Colab,
// Jupyter, Sphinx docs) advertises light/dark differently; detectTheme() must
// resolve every one. These cases are the contract - a widget that picks the
// wrong theme in any host is a regression.
import { describe, it, expect, beforeEach, vi } from "vitest";
import { detectTheme } from "./theme";

function reset() {
  document.documentElement.className = "";
  document.documentElement.removeAttribute("data-theme");
  document.documentElement.removeAttribute("data-mode");
  document.body.className = "";
  document.body.style.backgroundColor = "";
  delete document.body.dataset.jpThemeLight;
  document.getElementById("notebook")?.remove();
}

// jsdom has no matchMedia; stub it to "light" so the OS fallback is deterministic
// (no case below reaches it, but importing theme.ts must not crash).
beforeEach(() => {
  reset();
  vi.stubGlobal("matchMedia", (q: string) => ({ matches: false, media: q, addEventListener() {}, removeEventListener() {} }));
});

describe("detectTheme across hosts", () => {
  it("VS Code dark -> dark", () => {
    document.documentElement.className = "vscode-dark";
    expect(detectTheme()).toMatchObject({ environment: "vscode", theme: "dark" });
  });
  it("VS Code light -> light", () => {
    document.documentElement.className = "vscode-light";
    expect(detectTheme()).toMatchObject({ environment: "vscode", theme: "light" });
  });
  it("Colab dark -> dark", () => {
    document.body.className = "colaboratory";
    document.body.style.backgroundColor = "rgb(30, 30, 30)";
    expect(detectTheme()).toMatchObject({ environment: "colab", theme: "dark" });
  });
  it("Colab light -> light", () => {
    document.body.className = "colaboratory";
    document.body.style.backgroundColor = "rgb(255, 255, 255)";
    expect(detectTheme()).toMatchObject({ environment: "colab", theme: "light" });
  });
  it("Jupyter classic light -> light", () => {
    document.body.style.backgroundColor = "rgb(255, 255, 255)";
    const nb = document.createElement("div"); nb.id = "notebook"; document.body.appendChild(nb);
    expect(detectTheme()).toMatchObject({ environment: "jupyter-classic", theme: "light" });
  });
  it("JupyterLab dark -> dark", () => {
    document.body.dataset.jpThemeLight = "false";
    expect(detectTheme()).toMatchObject({ environment: "jupyterlab", theme: "dark" });
  });
  it("JupyterLab light -> light", () => {
    document.body.dataset.jpThemeLight = "true";
    expect(detectTheme()).toMatchObject({ environment: "jupyterlab", theme: "light" });
  });
  it("Sphinx docs light -> light", () => {
    document.documentElement.dataset.theme = "light";
    document.documentElement.dataset.mode = "light";
    expect(detectTheme()).toMatchObject({ environment: "docs", theme: "light" });
  });
  it("Sphinx docs dark -> dark", () => {
    document.documentElement.dataset.theme = "dark";
    document.documentElement.dataset.mode = "dark";
    expect(detectTheme()).toMatchObject({ environment: "docs", theme: "dark" });
  });
});
