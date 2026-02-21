function initMermaid() {
  if (
    typeof mermaid === "undefined" ||
    typeof mermaid.run !== "function"
  ) {
    return
  }
  mermaid.run({ querySelector: ".mermaid" })
}

if (
  typeof mermaid !== "undefined" &&
  typeof mermaid.initialize === "function"
) {
  mermaid.initialize({ startOnLoad: false })
}

if (
  typeof document$ !== "undefined" &&
  document$ &&
  typeof document$.subscribe === "function"
) {
  document$.subscribe(initMermaid)
} else if (typeof document !== "undefined" && document) {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initMermaid)
  } else {
    initMermaid()
  }
}
