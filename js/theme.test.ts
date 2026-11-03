// Theme auto-detection regression tests. Each environment (VS Code, Colab,
// Jupyter, Sphinx docs) advertises light/dark differently; detectTheme() must
// resolve every one. These cases are the contract - a widget that picks the
// wrong theme in any host is a regression.
import { describe, it, expect, beforeEach } from "vitest";
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

beforeEach(reset);

describe("detectTheme across hosts", () => {
  it("VS Code dark -> dark", () => {
    document.documentElement.className = "vscode-dark";
    expect(detectTheme()).toEqual({ theme: "dark" });
  });
  it("VS Code light -> light", () => {
    document.documentElement.className = "vscode-light";
    expect(detectTheme()).toEqual({ theme: "light" });
  });
  it("Colab dark -> dark", () => {
    document.body.className = "colaboratory";
    document.body.style.backgroundColor = "rgb(30, 30, 30)";
    expect(detectTheme()).toEqual({ theme: "dark" });
  });
  it("Colab light -> light", () => {
    document.body.className = "colaboratory";
    document.body.style.backgroundColor = "rgb(255, 255, 255)";
    expect(detectTheme()).toEqual({ theme: "light" });
  });
  it("Jupyter classic light -> light", () => {
    document.body.style.backgroundColor = "rgb(255, 255, 255)";
    const nb = document.createElement("div"); nb.id = "notebook"; document.body.appendChild(nb);
    expect(detectTheme()).toEqual({ theme: "light" });
  });
  it("JupyterLab dark -> dark", () => {
    document.body.dataset.jpThemeLight = "false";
    expect(detectTheme()).toEqual({ theme: "dark" });
  });
  it("JupyterLab light -> light", () => {
    document.body.dataset.jpThemeLight = "true";
    expect(detectTheme()).toEqual({ theme: "light" });
  });
  it("Sphinx docs light -> light", () => {
    document.documentElement.dataset.theme = "light";
    document.documentElement.dataset.mode = "light";
    expect(detectTheme()).toEqual({ theme: "light" });
  });
  it("Sphinx docs dark -> dark", () => {
    document.documentElement.dataset.theme = "dark";
    document.documentElement.dataset.mode = "dark";
    expect(detectTheme()).toEqual({ theme: "dark" });
  });
});

describe("detectTheme without host markers", () => {
  it("follows the OS color scheme", () => {
    const query = window.matchMedia;
    window.matchMedia = ((media: string) => ({ ...query(media), matches: true })) as typeof window.matchMedia;
    expect(detectTheme()).toEqual({ theme: "dark" });
    window.matchMedia = query;
    expect(detectTheme()).toEqual({ theme: "light" });
  });
});
