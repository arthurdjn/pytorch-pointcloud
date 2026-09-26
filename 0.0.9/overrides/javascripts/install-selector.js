function initInstallSelector() {
  var root = document.getElementById("install-selector");
  if (!root || root.dataset.iselInit) return;
  root.dataset.iselInit = "1";
  var TORCH = {
    "2.8": { v: "2.8.0", cuda: ["cpu", "cu126", "cu128", "cu129"] },
    "2.9": { v: "2.9.1", cuda: ["cpu", "cu126", "cu128", "cu130"] },
    "2.10": { v: "2.10.0", cuda: ["cpu", "cu126", "cu128", "cu130"] },
    "2.11": { v: "2.11.0", cuda: ["cpu", "cu126", "cu128", "cu130"] },
    "2.12": { v: "2.12.1", cuda: ["cpu", "cu126", "cu130", "cu132"] },
    "2.13": { v: "2.13.0", cuda: ["cpu", "cu126", "cu130", "cu132"] },
    "2.14": { v: "2.14.0", cuda: ["cpu", "cu126", "cu130", "cu132"] }
  };
  var CUDA_DOT = { cu126: "12.6", cu128: "12.8", cu129: "12.9", cu130: "13.0", cu132: "13.2" };
  var state = {
    pm: "uv", torch: "2.10", cuda: "cu128",
    extras: { pyg: true, flash: false, mamba: false, spconv: false, ocnn: false, torchsparse: false, sptr: false, lightning: false }
  };
  function disabledExtras() {
    var d = {};
    if (state.cuda === "cpu") { d.flash = "CUDA-only"; d.mamba = "CUDA-only"; d.torchsparse = "CUDA-only"; d.sptr = "CUDA-only"; }
    var noWheels = "no torch " + state.torch + " wheels on the Astral index";
    if (["2.13", "2.14"].indexOf(state.torch) !== -1) { d.flash = noWheels; }
    if (["2.12", "2.13", "2.14"].indexOf(state.torch) !== -1) { d.mamba = noWheels; }
    if (["2.13", "2.14"].indexOf(state.torch) !== -1) { d.sptr = "needs torch-scatter, which has no torch " + state.torch + " wheels"; }
    if (state.cuda === "cu130" || state.cuda === "cu132") { d.spconv = "no CUDA 13 build"; }
    return d;
  }
  function command() {
    var v = TORCH[state.torch].v;
    var tag = state.cuda;
    var pipish = state.pm === "uv" ? "uv pip install" : "pip install";
    var lines = [];
    if (state.pm === "conda") {
      lines.push("conda create -n pointcloud python=3.12 && conda activate pointcloud", "");
    }
    lines.push(pipish + " torch==" + v + " \\", "  --index-url https://download.pytorch.org/whl/" + tag);
    lines.push(pipish + " torch-pointcloud");
    if (state.extras.pyg) {
      lines.push("", pipish + " pyg-lib \\");
      lines.push("  -f https://data.pyg.org/whl/torch-" + v + "+" + tag + ".html");
    }
    var astral = [];
    if (state.extras.flash) astral.push("flash-attn==2.8.3.post1");
    if (state.extras.mamba) astral.push("mamba-ssm==2.3.2.post1", "causal-conv1d==1.6.2.post1");
    if (astral.length && tag !== "cpu") {
      var local = "+cu." + CUDA_DOT[tag] + ".torch." + state.torch;
      var flag = state.pm === "uv" ? "--index" : "--extra-index-url";
      lines.push("", pipish + " \\");
      astral.forEach(function (a) { lines.push('  "' + a + local + '" \\'); });
      lines.push("  " + flag + " https://wheels.astral.sh/simple/" + tag + "/");
    }
    if (state.extras.spconv && tag !== "cu130") {
      lines.push("");
      lines.push(pipish + (tag === "cpu" ? " spconv" : " spconv-cu126"));
    }
    if (state.extras.ocnn) {
      lines.push("", pipish + " ocnn");
      lines.push(pipish + " --no-build-isolation \\");
      lines.push('  "dwconv @ git+https://github.com/octree-nn/dwconv.git@ae53057eaf36dab01aa2727fcc93a749fd995af5"');
    }
    if (state.extras.torchsparse && tag !== "cpu") {
      lines.push("", pipish + " --no-deps --no-build-isolation \\");
      lines.push('  "torchsparse @ git+https://github.com/mit-han-lab/torchsparse.git@385f5ce8718fcae93540511b7f5832f4e71fd835"');
      lines.push(pipish + " --no-deps rootpath");
      lines.push(pipish + ' "backports.cached-property" wheel');
    }
    if (state.extras.sptr && tag !== "cpu") {
      lines.push("", pipish + " torch-scatter -f https://data.pyg.org/whl/torch-" + v + "+" + tag + ".html");
      lines.push(pipish + " --no-build-isolation \\");
      lines.push('  "sptr @ git+https://github.com/arthurdjn/SparseTransformer.git@fix/install-python-package"');
    }
    if (state.extras.lightning) {
      lines.push("", pipish + " lightning torchmetrics");
    }
    return lines.join("\n");
  }
  function render() {
    var dis = disabledExtras();
    var allowed = TORCH[state.torch].cuda;
    root.querySelectorAll("button[data-dim]").forEach(function (b) {
      var dim = b.dataset.dim, val = b.dataset.val;
      if (dim === "cuda") {
        var offc = allowed.indexOf(val) === -1;
        b.classList.toggle("isel-disabled", offc);
        if (offc) b.title = "no torch " + state.torch + " wheels for this CUDA version";
        else b.removeAttribute("title");
        b.classList.toggle("isel-active", state.cuda === val);
      } else if (dim === "extra") {
        var off = Object.prototype.hasOwnProperty.call(dis, val);
        if (off) state.extras[val] = false;
        b.classList.toggle("isel-disabled", off);
        if (off) b.title = dis[val];
        else b.removeAttribute("title");
        b.classList.toggle("isel-active", !off && !!state.extras[val]);
      } else {
        b.classList.toggle("isel-active", state[dim] === val);
      }
    });
    document.getElementById("isel-command").textContent = command();
  }
  var copyBtn = document.getElementById("isel-copy");
  function copied() {
    copyBtn.classList.add("isel-copied");
    setTimeout(function () { copyBtn.classList.remove("isel-copied"); }, 1200);
  }
  copyBtn.addEventListener("click", function () {
    var text = command();
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(copied, function () { fallbackCopy(text); });
    } else {
      fallbackCopy(text);
    }
  });
  function fallbackCopy(text) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    document.body.removeChild(ta);
    copied();
  }
  root.addEventListener("click", function (e) {
    var b = e.target.closest("button[data-dim]");
    if (!b || b.classList.contains("isel-disabled")) return;
    if (b.dataset.dim === "extra") state.extras[b.dataset.val] = !state.extras[b.dataset.val];
    else state[b.dataset.dim] = b.dataset.val;
    var allowed = TORCH[state.torch].cuda;
    if (allowed.indexOf(state.cuda) === -1) state.cuda = allowed[allowed.length - 1];
    render();
  });
  render();
}

initInstallSelector();
if (typeof document$ !== "undefined") {
  document$.subscribe(initInstallSelector);
}
