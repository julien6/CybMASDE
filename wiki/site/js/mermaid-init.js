document.addEventListener("DOMContentLoaded", function() {
  if (window.mermaid) {
    mermaid.initialize({
      startOnLoad: true,
      theme: document.documentElement.dataset.mdColorScheme === "slate" ? "dark" : "default",
      securityLevel: "loose"
    });
  }
});