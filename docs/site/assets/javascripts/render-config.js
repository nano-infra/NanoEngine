window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]]
  },
  options: {
    processHtmlClass: "arithmatex"
  }
};

document.addEventListener("DOMContentLoaded", function () {
  mermaid.initialize({
    startOnLoad: false,
    theme: document.body.getAttribute("data-md-color-scheme") === "slate" ? "dark" : "default"
  });
  mermaid.run({ querySelector: ".mermaid" });
});
