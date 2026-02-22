import panzoom from "https://cdn.jsdelivr.net/npm/panzoom@9/dist/panzoom.esm.js"

function applyPanzoom(svg) {
  if (svg.hasAttribute("data-panzoom-applied")) return
  svg.setAttribute("data-panzoom-applied", "true")

  svg.style.maxWidth = "none"
  svg.style.cursor = "grab"

  var instance = panzoom(svg, {
    maxZoom: 8,
    minZoom: 0.3,
    zoomSpeed: 0.065,
    bounds: true,
    boundsPadding: 0.1,
  })

  svg.addEventListener("dblclick", function () {
    instance.moveTo(0, 0)
    instance.zoomAbs(0, 0, 1)
  })
}

function scanAndApply() {
  document.querySelectorAll(".mermaid svg").forEach(applyPanzoom)
}

function watchForDiagrams() {
  var observer = new MutationObserver(function (mutations) {
    mutations.forEach(function (mutation) {
      mutation.addedNodes.forEach(function (node) {
        if (node.nodeType !== 1) return
        if (node.tagName === "svg" && node.closest(".mermaid")) {
          applyPanzoom(node)
        } else if (node.querySelectorAll) {
          node.querySelectorAll(".mermaid svg").forEach(applyPanzoom)
        }
      })
    })
  })
  observer.observe(document.body, { childList: true, subtree: true })
}

// Hook into MkDocs Material document$ observable (Instant Loading)
if (typeof document$ !== "undefined" && document$.subscribe) {
  document$.subscribe(function () {
    scanAndApply()
    watchForDiagrams()
  })
} else {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () {
      scanAndApply()
      watchForDiagrams()
    })
  } else {
    scanAndApply()
    watchForDiagrams()
  }
}
