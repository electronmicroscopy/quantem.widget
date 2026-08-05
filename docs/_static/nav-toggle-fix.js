// Keep the mobile hamburger alive across sphinx theme version drift.
//
// sphinx-book-theme and pydata-sphinx-theme both wire the sidebar drawer to
// document.querySelector('.primary-toggle') — the FIRST match. pydata >= 0.17
// renders its own hidden header button with that class ahead of the visible
// sphinx-book-theme hamburger, so every handler lands on the invisible button
// and the visible one goes dead (live docs, 2026-07-28). Forward clicks from
// the unwired buttons to the first (wired) one so any theme pair works.
document.addEventListener("DOMContentLoaded", () => {
  const toggles = Array.from(document.querySelectorAll(".primary-toggle"));
  if (toggles.length < 2) return; // single button: themes agree, nothing to fix
  const wired = toggles[0];
  for (const btn of toggles.slice(1)) {
    if (btn.dataset.navToggleFixed) continue; // idempotent if loaded twice
    btn.dataset.navToggleFixed = "1";
    btn.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      wired.click();
    });
  }
});
