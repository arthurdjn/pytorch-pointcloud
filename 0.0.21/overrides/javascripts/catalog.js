// Behaviour of the model catalog drawn by `docs/plugins/macros.py`: the parity plots and the checkpoint tables.
;(function () {
  var SVG_NS = "http://www.w3.org/2000/svg"
  // Tick spacings, the smallest giving at most eight ticks wins; `TICK_STEPS` in `macros.py` holds the same list.
  var TICK_STEPS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]

  function svgElement(name, attributes) {
    var node = document.createElementNS(SVG_NS, name)
    Object.keys(attributes).forEach(function (key) { node.setAttribute(key, attributes[key]) })
    return node
  }

  function tickValues(lo, span) {
    var step = TICK_STEPS.filter(function (s) { return span / s <= 8 })[0] || 50
    var first = Math.ceil(lo / step - 1e-9) * step
    var values = []
    for (var i = 0; first + i * step <= lo + span + 1e-9; i++) values.push(+(first + i * step).toFixed(6))
    return values
  }

  document$.subscribe(function () {
    // A Markdown table wrapped in `<div class="tp-catalog" markdown>` takes the generated tables' class, and with it
    // their style in place of the theme's rules for plain tables.
    document.querySelectorAll(".tp-catalog table:not([class])").forEach(function (table) {
      table.classList.add("tp-catalog__table")
    })

    // The theme's link previews open on hover and cover the next rows of a checkpoint table. A capturing listener on
    // the link itself runs before the theme's, so stopping `mouseenter` there keeps the preview for keyboard focus only.
    document.querySelectorAll(".tp-catalog a").forEach(function (link) {
      link.addEventListener("mouseenter", function (event) { event.stopImmediatePropagation() }, true)
    })

    // Parity plots: the pointer picks the nearest visible point within reach rather than the one under it, so dense
    // clusters stay usable; a click opens that checkpoint's table row, the legend keys show or hide a task, and dragging
    // a box zooms into it (double-click or the reset button to go back).
    document.querySelectorAll(".tp-parity").forEach(function (figure) {
      var plot = figure.querySelector(".tp-parity__plot")
      var svg = plot.querySelector("svg")
      var tooltip = plot.querySelector(".tp-parity__tooltip")
      var resetButton = plot.querySelector(".tp-parity__reset")
      var ticks = svg.querySelector(".tp-parity__ticks")
      var band = svg.querySelector("polygon.tp-parity__band")
      var diagonal = svg.querySelector("line.tp-parity__diagonal")
      var bandLabel = svg.querySelector(".tp-parity__band-label")
      var points = Array.prototype.slice.call(plot.querySelectorAll(".tp-parity__point"))
      var left = +svg.dataset.left
      var top = +svg.dataset.top
      var size = +svg.dataset.size
      var home = { x: +svg.dataset.lo, y: +svg.dataset.lo, span: svg.dataset.hi - svg.dataset.lo }
      var view = home
      var reach = 24
      var active = null
      var origin = null
      var selection = null
      var dragged = false

      function px(value) { return left + (value - view.x) / view.span * size }
      function py(value) { return top + size - (value - view.y) / view.span * size }

      // Both axes keep the same span, so the diagonal stays at 45 degrees and the band keeps its width in points.
      function layout(next) {
        view = next
        var bottom = top + size
        var right = left + size
        var nodes = []
        tickValues(view.x, view.span).forEach(function (t) {
          nodes.push(svgElement("line", { class: "tp-parity__grid", x1: px(t), y1: top, x2: px(t), y2: bottom }))
          var label = svgElement("text", { class: "tp-parity__tick", x: px(t), y: bottom + 18, "text-anchor": "middle" })
          label.textContent = t
          nodes.push(label)
        })
        tickValues(view.y, view.span).forEach(function (t) {
          nodes.push(svgElement("line", { class: "tp-parity__grid", x1: left, y1: py(t), x2: right, y2: py(t) }))
          var label = svgElement("text", { class: "tp-parity__tick", x: left - 8, y: py(t) + 4, "text-anchor": "end" })
          label.textContent = t
          nodes.push(label)
        })
        ticks.replaceChildren.apply(ticks, nodes)

        var lo = Math.min(view.x, view.y) - 2
        var hi = Math.max(view.x, view.y) + view.span + 2
        band.setAttribute("points", [[lo, lo - 1], [hi, hi - 1], [hi, hi + 1], [lo, lo + 1]].map(function (p) {
          return px(p[0]) + "," + py(p[1])
        }).join(" "))
        diagonal.setAttribute("x1", px(lo))
        diagonal.setAttribute("y1", py(lo))
        diagonal.setAttribute("x2", px(hi))
        diagonal.setAttribute("y2", py(hi))
        var t = Math.max(view.x, view.y) + 0.05 * view.span
        bandLabel.setAttribute("transform", "translate(" + px(t) + " " + py(t + 1) + ") rotate(-45)")
        bandLabel.style.display = t + 0.25 * view.span < Math.min(view.x, view.y) + view.span ? "" : "none"

        points.forEach(function (point) {
          var x = +point.dataset.reference
          var y = +point.dataset.score
          var outside = x < view.x || x > view.x + view.span || y < view.y || y > view.y + view.span
          point.firstElementChild.setAttribute("transform", "translate(" + px(x) + " " + py(y) + ")")
          point.classList.toggle("is-outside", outside)
          point.setAttribute("tabindex", outside || point.classList.contains("is-hidden") ? "-1" : "0")
        })
        resetButton.hidden = view === home
        hide()
      }

      function row(term, value) {
        var dt = document.createElement("dt")
        var dd = document.createElement("dd")
        dt.textContent = term
        dd.textContent = value
        return [dt, dd]
      }

      function show(point) {
        if (active) active.classList.remove("is-active")
        active = point
        point.classList.add("is-active")

        var data = point.dataset
        var name = document.createElement("div")
        var meta = document.createElement("div")
        var values = document.createElement("dl")
        var marker = point.querySelector("path").cloneNode()
        var key = document.createElementNS(SVG_NS, "svg")
        name.className = "tp-parity__tooltip-name"
        meta.className = "tp-parity__tooltip-meta"
        name.textContent = data.name
        marker.removeAttribute("transform")
        key.setAttribute("viewBox", "-6 -6 12 12")
        key.dataset.task = data.task
        key.appendChild(marker)
        meta.appendChild(key)
        meta.appendChild(document.createTextNode(data.dataset + " \u00b7 " + data.metric))
        ;[
          row("Measured", data.score),
          row("Reference", data.reference),
          row("Difference", data.gap),
          row("Params", data.params ? data.params + " M" : "\u2013"),
          row("License", data.license),
        ].forEach(function (pair) {
          values.appendChild(pair[0])
          values.appendChild(pair[1])
        })
        tooltip.replaceChildren(name, meta, values)
        tooltip.hidden = false

        var bounds = plot.getBoundingClientRect()
        var box = point.getBoundingClientRect()
        var x = box.left + box.width / 2 - bounds.left
        var y = box.top + box.height / 2 - bounds.top
        var leftEdge = x + 14
        if (leftEdge + tooltip.offsetWidth > bounds.width) leftEdge = x - 14 - tooltip.offsetWidth
        var topEdge = Math.min(Math.max(y - tooltip.offsetHeight / 2, 0), bounds.height - tooltip.offsetHeight)
        tooltip.style.left = Math.max(leftEdge, 0) + "px"
        tooltip.style.top = topEdge + "px"
      }

      function hide() {
        if (active) active.classList.remove("is-active")
        active = null
        tooltip.hidden = true
      }

      function nearest(event) {
        var best = null
        var distance = reach * reach
        points.forEach(function (point) {
          if (point.classList.contains("is-hidden") || point.classList.contains("is-outside")) return
          var box = point.getBoundingClientRect()
          var dx = box.left + box.width / 2 - event.clientX
          var dy = box.top + box.height / 2 - event.clientY
          if (dx * dx + dy * dy < distance) {
            best = point
            distance = dx * dx + dy * dy
          }
        })
        return best
      }

      function open(point) {
        window.location.href = new URL(point.dataset.href, window.location.href).href
      }

      // Pointer position in SVG units, held inside the plot area.
      function plotPosition(event) {
        var p = new DOMPoint(event.clientX, event.clientY).matrixTransform(svg.getScreenCTM().inverse())
        return { x: Math.min(Math.max(p.x, left), left + size), y: Math.min(Math.max(p.y, top), top + size) }
      }

      svg.addEventListener("pointerdown", function (event) {
        dragged = false
        if (event.pointerType === "touch" || event.button !== 0) return
        var p = new DOMPoint(event.clientX, event.clientY).matrixTransform(svg.getScreenCTM().inverse())
        if (p.x < left || p.x > left + size || p.y < top || p.y > top + size) return
        event.preventDefault()
        origin = plotPosition(event)
        svg.setPointerCapture(event.pointerId)
      })
      svg.addEventListener("pointermove", function (event) {
        if (!origin) return
        var p = plotPosition(event)
        if (!dragged && Math.abs(p.x - origin.x) + Math.abs(p.y - origin.y) < 6) return
        if (!dragged) {
          dragged = true
          hide()
          selection = svgElement("rect", { class: "tp-parity__selection" })
          svg.appendChild(selection)
        }
        selection.setAttribute("x", Math.min(p.x, origin.x))
        selection.setAttribute("y", Math.min(p.y, origin.y))
        selection.setAttribute("width", Math.abs(p.x - origin.x))
        selection.setAttribute("height", Math.abs(p.y - origin.y))
      })
      svg.addEventListener("pointerup", function (event) {
        if (!origin) return
        if (dragged) {
          var p = plotPosition(event)
          var x0 = view.x + (Math.min(p.x, origin.x) - left) / size * view.span
          var x1 = view.x + (Math.max(p.x, origin.x) - left) / size * view.span
          var y0 = view.y + (top + size - Math.max(p.y, origin.y)) / size * view.span
          var y1 = view.y + (top + size - Math.min(p.y, origin.y)) / size * view.span
          var span = Math.max(x1 - x0, y1 - y0, 0.5)
          selection.remove()
          selection = null
          layout({ x: (x0 + x1 - span) / 2, y: (y0 + y1 - span) / 2, span: span })
        }
        origin = null
      })
      svg.addEventListener("dblclick", function () { layout(home) })
      resetButton.addEventListener("click", function () { layout(home) })

      plot.addEventListener("pointermove", function (event) {
        if (origin) return
        var point = nearest(event)
        if (point === active) return
        point ? show(point) : hide()
      })
      plot.addEventListener("pointerleave", hide)
      plot.addEventListener("click", function (event) {
        if (dragged || event.target === resetButton) return
        var point = event.target.closest(".tp-parity__point") || nearest(event)
        if (point) open(point)
      })
      points.forEach(function (point) {
        point.addEventListener("focus", function () { show(point) })
        point.addEventListener("blur", hide)
        point.addEventListener("keydown", function (event) {
          if (event.key === "Enter") open(point)
        })
      })

      figure.querySelectorAll("button.tp-parity__key").forEach(function (key) {
        key.addEventListener("click", function () {
          var visible = key.getAttribute("aria-pressed") !== "true"
          key.setAttribute("aria-pressed", String(visible))
          points.forEach(function (point) {
            if (point.dataset.task !== key.dataset.task) return
            point.classList.toggle("is-hidden", !visible)
            point.setAttribute("tabindex", visible && !point.classList.contains("is-outside") ? "0" : "-1")
          })
          hide()
        })
      })

      layout(home)
    })
  })
})()
