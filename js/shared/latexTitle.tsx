import * as React from "react";

/** Rich panel titles: a tiny LaTeX subset (sub/superscripts, \\mathrm, Greek and
 *  common symbols) rendered as spans so titles like $\\chi^2$ read like body text
 *  without pulling in a math typesetting library. */
export type RichTitleSpan = { text?: unknown; math?: unknown; color?: unknown };

export const LATEX_SYMBOLS: Record<string, string> = {
  alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ε", varepsilon: "ε",
  zeta: "ζ", eta: "η", theta: "θ", vartheta: "ϑ", iota: "ι", kappa: "κ",
  lambda: "λ", mu: "μ", nu: "ν", xi: "ξ", pi: "π", rho: "ρ", varrho: "ϱ",
  sigma: "σ", tau: "τ", upsilon: "υ", phi: "φ", varphi: "ϕ", chi: "χ",
  psi: "ψ", omega: "ω", Gamma: "Γ", Delta: "Δ", Theta: "Θ", Lambda: "Λ",
  Xi: "Ξ", Pi: "Π", Sigma: "Σ", Phi: "Φ", Psi: "Ψ", Omega: "Ω",
  pm: "±", times: "×", cdot: "·", degree: "°", angstrom: "Å", le: "≤",
  ge: "≥", neq: "≠", approx: "≈", infty: "∞",
};

export function readLatexGroup(expr: string, start: number): { text: string; next: number } {
  if (expr[start] !== "{") return { text: expr[start] || "", next: start + 1 };
  let depth = 0;
  for (let i = start; i < expr.length; i += 1) {
    if (expr[i] === "{") depth += 1;
    if (expr[i] === "}") depth -= 1;
    if (depth === 0) return { text: expr.slice(start + 1, i), next: i + 1 };
  }
  return { text: expr.slice(start + 1), next: expr.length };
}

function readLatexAtom(expr: string, start: number): { text: string; next: number } {
  if (expr[start] === "{") return readLatexGroup(expr, start);
  if (expr[start] === "\\") {
    const match = expr.slice(start + 1).match(/^[A-Za-z]+/);
    if (match) return { text: `\\${match[0]}`, next: start + 1 + match[0].length };
  }
  return { text: expr[start] || "", next: start + 1 };
}

function renderLatexMath(expr: string, keyPrefix: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];
  let i = 0;
  let key = 0;
  while (i < expr.length) {
    const ch = expr[i];
    if ((ch === "^" || ch === "_") && i + 1 < expr.length) {
      const atom = readLatexAtom(expr, i + 1);
      const Tag = ch === "^" ? "sup" : "sub";
      nodes.push(
        <Tag key={`${keyPrefix}-script-${key++}`} style={{ fontSize: "0.72em", lineHeight: 0 }}>
          {renderLatexMath(atom.text, `${keyPrefix}-script-${key}`)}
        </Tag>
      );
      i = atom.next;
      continue;
    }
    if (ch === "\\") {
      const match = expr.slice(i + 1).match(/^[A-Za-z]+/);
      if (match) {
        const command = match[0];
        i += command.length + 1;
        if ((command === "mathrm" || command === "text") && expr[i] === "{") {
          const group = readLatexGroup(expr, i);
          nodes.push(<span key={`${keyPrefix}-roman-${key++}`} style={{ fontStyle: "normal" }}>{group.text}</span>);
          i = group.next;
          continue;
        }
        nodes.push(LATEX_SYMBOLS[command] || command);
        continue;
      }
    }
    if (ch === "{" || ch === "}") {
      i += 1;
      continue;
    }
    nodes.push(ch);
    i += 1;
  }
  return nodes;
}

// Same UI font as panel titles / badges (not Cambria Math italic) so χ², λ, etc. match body text.
const UI_MATH_FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";

export function renderMathExpression(expr: string, keyPrefix: string): React.ReactNode {
  const normalized = expr.trim().replace(/\\+(?=[A-Za-z])/g, "\\");
  return (
    <span key={keyPrefix} data-quantem-math="true" style={{ fontFamily: UI_MATH_FONT, fontStyle: "normal" }}>
      {renderLatexMath(normalized, keyPrefix)}
    </span>
  );
}

function findUnescapedDollar(text: string, from = 0): number {
  for (let i = from; i < text.length; i += 1) {
    if (text[i] === "$" && text[i - 1] !== "\\") return i;
  }
  return -1;
}

function renderTextWithInlineMath(text: string, keyPrefix: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];
  let rest = text;
  let key = 0;
  while (rest.length) {
    const dollar = findUnescapedDollar(rest);
    const paren = rest.indexOf("\\(");
    const starts = [dollar, paren].filter((idx) => idx >= 0);
    if (!starts.length) {
      nodes.push(rest.replace(/\\\$/g, "$"));
      break;
    }
    const start = Math.min(...starts);
    if (start > 0) nodes.push(rest.slice(0, start).replace(/\\\$/g, "$"));
    const dollarMode = rest[start] === "$";
    const close = dollarMode ? findUnescapedDollar(rest, start + 1) - (start + 1) : rest.indexOf("\\)", start + 2) - (start + 2);
    if (close < 0) {
      nodes.push(rest.slice(start).replace(/\\\$/g, "$"));
      break;
    }
    const math = rest.slice(start + (dollarMode ? 1 : 2), start + (dollarMode ? 1 : 2) + close);
    nodes.push(renderMathExpression(math, `${keyPrefix}-math-${key++}`));
    rest = rest.slice(start + (dollarMode ? 2 : 4) + close);
  }
  return nodes;
}

export function richTitlePlainText(spans: RichTitleSpan[] | undefined, fallback: string): string {
  if (!Array.isArray(spans) || spans.length === 0) return fallback;
  const text = spans.map((span) => String(span?.text ?? span?.math ?? "")).join("");
  return text || fallback;
}

export function renderRichTitle(spans: RichTitleSpan[] | undefined, fallback: string): React.ReactNode {
  if (!Array.isArray(spans) || spans.length === 0) return renderTextWithInlineMath(fallback, "title-fallback");
  return spans.map((span, idx) => {
    const text = String(span?.text ?? "");
    const math = span?.math == null ? "" : String(span.math);
    const color = typeof span?.color === "string" && span.color.trim() ? span.color : undefined;
    return (
      <span key={`title-span-${idx}`} style={color ? { color } : undefined}>
        {math ? renderMathExpression(math, `title-span-${idx}`) : renderTextWithInlineMath(text, `title-span-${idx}`)}
      </span>
    );
  });
}
