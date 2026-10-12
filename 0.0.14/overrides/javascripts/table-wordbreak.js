// Table cells hold identifiers that must not split mid-token, but a narrow column still has to wrap
// somewhere. CSS has no camel-case rule, so mark the humps as the only break opportunities and let
// `overflow-wrap: normal` keep every other position intact: AxisMin|Offset, never AxisMinOffs|et.
// The split is lowercase-then-uppercase only, so acronyms such as S3DIS stay whole.
document$.subscribe(function () {
  var cells = document.querySelectorAll(".md-typeset__table table th, .md-typeset__table table td")
  cells.forEach(function (cell) {
    var walker = document.createTreeWalker(cell, NodeFilter.SHOW_TEXT)
    var nodes = []
    while (walker.nextNode()) nodes.push(walker.currentNode)

    nodes.forEach(function (node) {
      var pieces = node.nodeValue.replace(/([a-z])([A-Z])/g, "$1\u0000$2").split("\u0000")
      if (pieces.length < 2) return

      var fragment = document.createDocumentFragment()
      pieces.forEach(function (piece, index) {
        if (index) fragment.appendChild(document.createElement("wbr"))
        fragment.appendChild(document.createTextNode(piece))
      })
      node.parentNode.replaceChild(fragment, node)
    })
  })
})
