// Apply panzoom to a single Mermaid SVG element
function applyPanzoom(svg) {
  if (svg.hasAttribute("data-panzoom-applied")) return
  svg.setAttribute("data-panzoom-applied", "true")

  // Make the SVG fill its container so panzoom has room
  svg.style.maxWidth = "none"
  svg.style.cursor = "grab"

  var instance = panzoom(svg, {
    maxZoom: 8,
    minZoom: 0.3,
    zoomSpeed: 0.065,
    bounds: true,
    boundsPadding: 0.1,
  })

  // Reset on double-click
  svg.addEventListener("dblclick", function () {
    instance.moveTo(0, 0)
    instance.zoomAbs(0, 0, 1)
  })
}

// Scan for newly rendered Mermaid SVGs and apply panzoom
function scanAndApply() {
  document.querySelectorAll(".mermaid svg").forEach(applyPanzoom)
}

// Watch for Mermaid rendering dynamically (it renders async)
function watchForDiagrams() {
  var observer = new MutationObserver(function (mutations) {
    mutations.forEach(function (mutation) {
      mutation.addedNodes.forEach(function (node) {
        if (node.nodeType !== 1) return
        if (node.tagName === "svg" && node.closest(".mermaid")) {
          applyPanzoom(node)
        } else {
          node.querySelectorAll &&
            node.querySelectorAll(".mermaid svg").forEach(applyPanzoom)
        }
      })
    })
  })
  observer.observe(document.body, { childList: true, subtree: true })
}

// Hook into MkDocs Material's document$ observable (fires on every navigation)
if (typeof document$ !== "undefined" && document$.subscribe) {
  document$.subscribe(function () {
    scanAndApply()
    watchForDiagrams()
  })
} else {
  // Fallback for non-Material or non-instant-loading builds
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
