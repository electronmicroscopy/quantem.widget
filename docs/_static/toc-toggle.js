// Contents sidebar toggle
(function () {
  const KEY = "quantem-docs-hide-toc";

  let hidden = false;
  try {
    hidden = localStorage.getItem(KEY) === "1";
  } catch (e) {}
  if (hidden) {
    document.documentElement.classList.add("toc-hidden");
  }

  document.addEventListener("DOMContentLoaded", function () {
    const sidebar = document.querySelector(".bd-sidebar-secondary");
    if (!sidebar) return;

    const button = document.createElement("button");
    button.className = "btn btn-sm toc-toggle-button";
    button.type = "button";
    button.title = "Show or hide the contents sidebar";
    button.setAttribute("aria-label", "Show or hide the contents sidebar");
    button.innerHTML = '<i class="fas fa-list"></i>';
    button.addEventListener("click", function () {
      const nowHidden = !document.documentElement.classList.contains("toc-hidden");
      document.documentElement.classList.toggle("toc-hidden", nowHidden);
      try {
        localStorage.setItem(KEY, nowHidden ? "1" : "0");
      } catch (e) {}
    });

    const header = document.querySelector(".article-header-buttons");
    if (header) {
      header.appendChild(button);
    } else {
      sidebar.prepend(button);
    }
  });
})();
