"""Macros callable from the documentation Markdown and from the source docstrings.

Zensical loads this module through its macros extension, configured in `zensical.toml`.
Every Markdown page and every docstring rendered by mkdocstrings passes through Jinja2
first, so a macro registered here is available in both:

```markdown
{{ paper("1706.02413") }}
```

Assets and metadata are produced ahead of the build by `docs/scripts/build_paper_cards.py`
(`make papers`), which scans the same call sites. A macro never reaches the network.
"""

import csv
import functools
import html
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

import jinja2

DOCS_DIR = Path(__file__).resolve().parents[1]
PAPERS_DIR = DOCS_DIR / "assets" / "papers"
METADATA_PATH = DOCS_DIR / "data" / "papers.json"
CATALOG_PATH = DOCS_DIR / "data" / "models.csv"
MODELS_DIR = DOCS_DIR.parent / "src" / "torch_pointcloud" / "models"

# Display name of each architecture, keyed by its module in `torch_pointcloud.models`.
MODEL_LABELS = {
    "centerpoint": "CenterPoint",
    "concerto": "Concerto",
    "dgcnn": "DGCNN",
    "kpconv": "KPConv",
    "lion": "LION",
    "octformer": "OctFormer",
    "point_bert": "Point-BERT",
    "point_m2ae": "Point-M2AE",
    "point_mae": "Point-MAE",
    "point_mamba": "PointMamba",
    "point_transformer": "Point Transformer V1",
    "point_transformer_v2": "Point Transformer V2",
    "point_transformer_v3": "Point Transformer V3",
    "pointcnn": "PointCNN",
    "pointconv": "PointConv",
    "pointgpt": "PointGPT",
    "pointmlp": "PointMLP",
    "pointnet": "PointNet",
    "pointnet2": "PointNet++",
    "pointnext": "PointNeXt",
    "pointpillars": "PointPillars",
    "pointrcnn": "PointRCNN",
    "pvcnn": "PVCNN",
    "pvcnn2": "PVCNN++",
    "randlanet": "RandLA-Net",
    "second": "SECOND",
    "sonata": "Sonata",
    "spformer_unet": "SPFormer-UNet",
    "sphereformer": "SphereFormer",
    "spunet": "SpUNet",
    "spvcnn": "SPVCNN",
    "threedetr": "3DETR",
    "utonia": "Utonia",
    "votenet": "VoteNet",
    "voxel_mamba": "Voxel-Mamba",
    "voxelnext": "VoxelNeXt",
}

# Label, docs page and plot marker (an SVG path around the origin) of each registry task.
TASKS: Dict[str, Tuple[str, str, str]] = {
    "classification": ("Classification", "models/classification.md", "M-4.5,0a4.5,4.5 0 1,0 9,0a4.5,4.5 0 1,0 -9,0Z"),
    "part-segmentation": ("Part segmentation", "models/part-segmentation.md", "M-4,-4h8v8h-8Z"),
    "semantic-segmentation": ("Semantic segmentation", "models/segmentation.md", "M0,-5.5L5.2,3.5H-5.2Z"),
    "detection": ("Detection", "models/detection.md", "M0,-5.5L5.5,0L0,5.5L-5.5,0Z"),
    "pretraining": ("Pretraining", "models/features.md", ""),
}

DATASET_LABELS = {
    "modelnet40": "ModelNet40",
    "scanobjectnn-hardest": "ScanObjectNN PB_T50_RS",
    "scanobjectnn-objbg": "ScanObjectNN OBJ_BG",
    "scanobjectnn-objonly": "ScanObjectNN OBJ_ONLY",
    "shapenetpart": "ShapeNetPart",
    "scannet20": "ScanNet20",
    "scannet200": "ScanNet200",
    "semantickitti": "SemanticKITTI",
    "kitti": "KITTI",
    "nuscenes": "nuScenes",
    "scannet": "ScanNet",
    "sunrgbd": "SUN RGB-D",
    "shapenet55": "ShapeNet55",
    "pretrain": "Multi-dataset",
}
LICENSE_LABELS = {"Apache-2.0": "Apache 2.0", "CC-BY-NC-4.0": "CC BY-NC 4.0"}
DASH = "\u2013"
MINUS = "\u2212"
DELTA = "\u0394"

# Parity plot geometry in SVG user units: a square plot area and the margins around it.
PLOT_SIZE = 380
PLOT_LEFT = 46
PLOT_TOP = 10
PLOT_BOTTOM = 42
PLOT_RIGHT = 10
# Tick spacings, the smallest giving at most eight ticks wins; `catalog.js` redraws zoomed axes with the same list.
TICK_STEPS = (0.1, 0.2, 0.5, 1, 2, 5, 10, 20)

CARD = """<div class="paper-card paper-card--page" markdown="1">
{preview}
<p class="paper-card__meta{modifier}" markdown="span">{cite}{date}</p>

</div>"""

PREVIEW = """
<img class="paper-card__page" src="{src}" alt="First page of {title}">
"""

ARXIV_CITE = ":arxiv-name: [`{key}`](https://arxiv.org/abs/{key})"
WORDMARK = " paper-card__meta--wordmark"


def alt_text(title: str) -> str:
    """Flatten a paper title into prose that survives an HTML attribute.

    Arithmatex scans the whole text stream, so a title carrying inline math (`PointCNN:
    Convolution On $\\mathcal{X}$-Transformed Points`) would have its `$...$` lifted out
    of the `alt` attribute, splitting the tag and spilling the rest onto the page.

    Args:
        title: Paper title as the arXiv API reports it.

    Returns:
        The title with TeX commands, math delimiters and quotes removed.

    Example:
        >>> alt_text(r"Convolution On $\\mathcal{X}$-Transformed Points")
        'Convolution On X-Transformed Points'
    """
    text = re.sub(r"\\[a-zA-Z]+", "", title)
    return re.sub(r'[${}"]', "", text)


def define_env(env: Any) -> None:
    """Register the documentation macros with the Jinja2 environment.

    Args:
        env: Macro environment supplied by the zensical macros extension.
    """
    env.macro(paper)
    env.macro(model_table)
    env.macro(architecture_table)
    env.macro(parity_plot)


@jinja2.pass_context
def paper(context: jinja2.runtime.Context, key: str, crop: Optional[float] = None) -> str:
    r"""Render a preview card for a paper, linking to where it was published.

    The card shows the top of page 1 and the paper's identifier and date. It degrades to
    a plain identifier and link when the preview has not been rendered yet.

    Args:
        key: Bare arXiv identifier, for example `1706.02413`, or the slug of a paper
            published elsewhere, for example `kitti-2012`.
        crop: Fraction of the first page to keep. Read by `build_paper_cards.py`, which
            renders the asset, and ignored here.

    Returns:
        Markdown, re-parsed by the surrounding page.

    Example:
        ```markdown
        {{ paper("1706.02413", crop=0.58) }}
        ```
    """
    del crop
    metadata: Dict[str, Dict[str, str]] = {}
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text())

    meta = metadata.get(key, {})
    asset = PAPERS_DIR / f"{key}-page.webp"
    preview = ""
    if meta and asset.exists():
        page = context.get("page")
        depth = len(PurePosixPath(page.path).parts) - 1 if page else 0
        src = "../" * depth + asset.relative_to(DOCS_DIR).as_posix()
        preview = PREVIEW.format(src=src, title=alt_text(meta["title"]))

    if "url" not in meta:
        cite, modifier = ARXIV_CITE.format(key=key), WORDMARK
    else:
        icon = f":{meta['icon']}: " if "icon" in meta else ""
        cite, modifier = f"{icon}[{meta['label']}]({meta['url']})", ""

    return CARD.format(
        preview=preview,
        cite=cite,
        modifier=modifier,
        date=f" &middot; {meta['date']}" if meta else "",
    )


def read_catalog() -> List[Dict[str, str]]:
    with CATALOG_PATH.open() as f:
        return list(csv.DictReader(f))


@functools.lru_cache(maxsize=None)
def paper_key(module: str) -> str:
    """The first `paper(...)` key cited by a model module's docstring, empty when it cites none."""
    match = re.search(r'paper\("([^"]+)"', (MODELS_DIR / f"{module}.py").read_text())
    return match.group(1) if match else ""


def relative_link(context: jinja2.runtime.Context, path: str) -> str:
    """Link from the page being rendered to the docs source `path`.

    Zensical resolves `.md` links in raw HTML attributes (`data-href` included) as it does in Markdown, so one form
    serves both.
    """
    page = context.get("page")
    depth = len(PurePosixPath(page.path).parts) - 1 if page else 0
    return "../" * depth + path


def dataset_label(dataset: str) -> str:
    area = re.fullmatch(r"s3dis-area(\d)", dataset)
    return f"S3DIS Area {area.group(1)}" if area else DATASET_LABELS.get(dataset, dataset)


def metric_label(task: str, dataset: str) -> str:
    if task == "detection":
        return {"kitti": "mAP (R11)", "nuscenes": "mAP"}.get(dataset, "mAP@0.25")
    return {"classification": "OA", "part-segmentation": "Instance mIoU", "semantic-segmentation": "mIoU"}.get(task, "")


def aligned(value: str, places: int = 2) -> str:
    """Pad a score to `places` decimals with figure spaces, so a column aligns on the point without implying digits."""
    whole, _, decimals = value.partition(".")
    return whole + ("." if decimals else "\u2008") + decimals + "\u2007" * (places - len(decimals))


def signed(value: float) -> str:
    return "0.00" if round(value, 2) == 0 else f"{value:+.2f}".replace("-", MINUS)


def millions(params: str) -> str:
    if not params:
        return ""
    value = float(params)
    return f"{value:.2f}" if value < 1 else f"{value:.1f}"


@jinja2.pass_context
def model_table(context: jinja2.runtime.Context, task: str) -> str:
    """Render the pretrained checkpoints of a task from the model catalog `docs/data/models.csv`.

    The catalog is synced from the registry by `docs/scripts/build_model_tables.py` (`make tables`). Checkpoints are
    grouped by evaluation dataset and ranked by the score this package measures, next to the score the reference
    publishes. Each row carries the checkpoint name as its `id`, which the points of `parity_plot` link to.

    Args:
        task: Registry task, for example `classification` or `semantic-segmentation`.

    Returns:
        An HTML table, one row per checkpoint.

    Example:
        ```markdown
        {{ model_table("classification") }}
        ```
    """
    rows = [row for row in read_catalog() if row["task"] == task and row["pretrained"] == "true"]
    scored = any(row["reference"] or row["score"] for row in rows)
    columns = ["Checkpoint", "Params (M)"] + (["Ref.", "Measured", DELTA] if scored else []) + ["License"]
    numeric = {"Params (M)", "Ref.", "Measured", DELTA}

    def rank(row: Dict[str, str]) -> Tuple[float, float, float, str]:
        """Best measured score first; unscored rows follow, smallest model first."""
        score, reference = -float(row["score"] or "-inf"), -float(row["reference"] or "-inf")
        return (score, reference, float(row["params"] or "inf"), row["checkpoint"])

    head = "".join(f'<th class="tp-num">{c}</th>' if c in numeric else f"<th>{c}</th>" for c in columns)
    groups = []
    for dataset in sorted({row["dataset"] for row in rows}):
        metric = metric_label(task, dataset)
        label = html.escape(dataset_label(dataset)) + (f" <span>{metric}</span>" if metric else "")
        lines = [f'<tr class="tp-catalog__group"><th colspan="{len(columns)}" scope="colgroup">{label}</th></tr>']
        for row in sorted((row for row in rows if row["dataset"] == dataset), key=rank):
            name = html.escape(row["checkpoint"])
            # A long name wraps before its dots only, never at a hyphen. The architecture leads; the dataset and
            # author tags, shared by most rows of a group, are muted.
            arch, *tags = name.split(".")
            text = f"<span>{arch}</span>" + "".join(f'<wbr><span class="tp-catalog__tag">.{tag}</span>' for tag in tags)
            api = relative_link(context, f"api/models/{row['architecture']}.md")
            cells = [
                f'<td><a class="tp-catalog__name" href="{api}">{text}</a></td>',
                f'<td class="tp-num">{millions(row["params"])}</td>',
            ]
            if scored:
                ref, ours = row["reference"], row["score"]
                gap = signed(float(ours) - float(ref)) if ref and ours else ""
                off = " tp-catalog__gap--off" if gap and abs(round(float(ours) - float(ref), 2)) > 1 else ""
                cells += [
                    f'<td class="tp-num">{aligned(ref) if ref else DASH}</td>',
                    f'<td class="tp-num tp-catalog__score">{aligned(ours) if ours else DASH}</td>',
                    f'<td class="tp-num tp-catalog__gap{off}">{gap}</td>',
                ]
            license = LICENSE_LABELS.get(row["license"], row["license"])
            restricted = " tp-catalog__license--nc" if "-NC" in row["license"] else ""
            cells.append(f'<td class="tp-catalog__license{restricted}">{html.escape(license)}</td>')
            lines.append(f'<tr id="{name}">' + "".join(cells) + "</tr>")
        groups.append("<tbody>" + "".join(lines) + "</tbody>")
    return f'<div class="tp-catalog"><table class="tp-catalog__table"><thead><tr>{head}</tr></thead>{"".join(groups)}</table></div>'


@jinja2.pass_context
def architecture_table(context: jinja2.runtime.Context, task: str) -> str:
    """Render the architectures registered for a task, oldest paper first.

    Rows come from the model catalog `docs/data/models.csv`, the paper from the first `paper(...)` call of the model
    module's docstring and its date from `docs/data/papers.json`. Only the display names in `MODEL_LABELS` are kept
    by hand.

    Args:
        task: Registry task, for example `classification` or `semantic-segmentation`.

    Returns:
        A Markdown table, one row per architecture.

    Example:
        ```markdown
        {{ architecture_table("classification") }}
        ```
    """
    metadata: Dict[str, Dict[str, str]] = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
    counts: Dict[str, int] = {}
    for row in read_catalog():
        if row["task"] == task:
            counts[row["architecture"]] = counts.get(row["architecture"], 0) + (row["pretrained"] == "true")

    entries = []
    for module, count in counts.items():
        key = paper_key(module)
        meta = metadata.get(key, {})
        year = meta.get("date", "").rsplit(" ", 1)[-1]
        if "url" in meta:
            icon = f":{meta['icon']}: " if "icon" in meta else ""
            cite = f"{icon}[{meta['title']}]({meta['url']})"
        else:
            cite = f":arxiv: [{meta.get('title', key)}](https://arxiv.org/abs/{key})" if key else ""
        label = MODEL_LABELS.get(module, module)
        checkpoints = f"[{count}]({relative_link(context, TASKS[task][1])}#checkpoints)" if count else DASH
        model = f"[{label}]({relative_link(context, f'api/models/{module}.md')})"
        if cite:
            model += f'<br><span class="tp-index__detail">{cite}</span>'
        entries.append((year, label, f"| {model} | {year} | {checkpoints} |"))

    lines = ["| Model | Year | Checkpoints |", "| --- | ---: | ---: |"]
    lines += [line for _, _, line in sorted(entries)]
    return '<div class="tp-index" markdown>\n\n' + "\n".join(lines) + "\n\n</div>"


@jinja2.pass_context
def parity_plot(context: jinja2.runtime.Context, task: Optional[str] = None) -> str:
    """Plot the score measured for each pretrained checkpoint against the score its reference publishes.

    Points are drawn at build time as inline SVG, one hue and one marker per task, and link to their row in the task's
    checkpoint table. Axes span the data, so a single task gets a zoomed view. `docs/overrides/javascripts/catalog.js`
    adds the hover details, the legend toggles and box zoom; it reads the plot area and the initial range from the
    `data-*` attributes of the `<svg>` and redraws the ticks, the band and the points for each zoomed range.

    Args:
        task: Registry task to plot, or `None` for every task with published scores.

    Returns:
        A raw HTML figure, or an empty string when no checkpoint of the task has both scores.

    Example:
        ```markdown
        {{ parity_plot("classification") }}
        ```
    """
    rows = [
        row
        for row in read_catalog()
        if row["pretrained"] == "true"
        and row["reference"]
        and row["score"]
        and TASKS.get(row["task"], ("", "", ""))[2]
        and task in (None, row["task"])
    ]
    if not rows:
        return ""
    rows.sort(key=lambda row: list(TASKS).index(row["task"]))

    values = [float(row[key]) for row in rows for key in ("reference", "score")]
    pad = max(1.0, 0.04 * (max(values) - min(values)))
    lo, hi = math.floor(min(values) - pad), min(math.ceil(max(values) + pad), 100)
    step = next(s for s in TICK_STEPS if (hi - lo) / s <= 8)
    first = math.ceil(lo / step - 1e-9) * step
    ticks = [round(first + i * step, 6) for i in range(int((hi - first) / step + 1e-9) + 1)]

    def x_of(value: float) -> float:
        return PLOT_LEFT + (value - lo) / (hi - lo) * PLOT_SIZE

    def y_of(value: float) -> float:
        return PLOT_TOP + PLOT_SIZE - (value - lo) / (hi - lo) * PLOT_SIZE

    width, height = PLOT_LEFT + PLOT_SIZE + PLOT_RIGHT, PLOT_TOP + PLOT_SIZE + PLOT_BOTTOM
    bottom, right = PLOT_TOP + PLOT_SIZE, PLOT_LEFT + PLOT_SIZE
    svg = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Measured against reference score" '
        f'data-left="{PLOT_LEFT}" data-top="{PLOT_TOP}" data-size="{PLOT_SIZE}" data-lo="{lo}" data-hi="{hi}">',
        f'<clipPath id="tp-parity-clip"><rect x="{PLOT_LEFT}" y="{PLOT_TOP}" width="{PLOT_SIZE}" '
        f'height="{PLOT_SIZE}"/></clipPath>',
        '<g class="tp-parity__ticks">',
    ]
    for tick in ticks:
        x, y = x_of(tick), y_of(tick)
        svg.append(f'<line class="tp-parity__grid" x1="{x:.1f}" y1="{PLOT_TOP}" x2="{x:.1f}" y2="{bottom}"/>')
        svg.append(f'<line class="tp-parity__grid" x1="{PLOT_LEFT}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}"/>')
        svg.append(f'<text class="tp-parity__tick" x="{x:.1f}" y="{bottom + 18}" text-anchor="middle">{tick:g}</text>')
        svg.append(
            f'<text class="tp-parity__tick" x="{PLOT_LEFT - 8}" y="{y + 4:.1f}" text-anchor="end">{tick:g}</text>'
        )
    band = [(lo, lo - 1), (hi, hi - 1), (hi, hi + 1), (lo, lo + 1)]
    svg += [
        "</g>",
        '<polygon class="tp-parity__band" clip-path="url(#tp-parity-clip)" points="'
        + " ".join(f"{x_of(x):.1f},{y_of(y):.1f}" for x, y in band)
        + '"/>',
        f'<line class="tp-parity__diagonal" clip-path="url(#tp-parity-clip)" x1="{x_of(lo):.1f}" '
        f'y1="{y_of(lo):.1f}" x2="{x_of(hi):.1f}" y2="{y_of(hi):.1f}"/>',
        # The band is labelled along its upper edge near the origin, where the data rarely is.
        f'<text class="tp-parity__tick tp-parity__band-label" transform="translate({x_of(lo + 0.05 * (hi - lo)):.1f} '
        f'{y_of(lo + 0.05 * (hi - lo) + 1):.1f}) rotate(-45)" y="-5">\u00b11 point</text>',
        f'<path class="tp-parity__axis" d="M{PLOT_LEFT},{PLOT_TOP}V{bottom}H{right}"/>',
        f'<text class="tp-parity__title" x="{PLOT_LEFT + PLOT_SIZE / 2}" y="{height - 4}" '
        'text-anchor="middle">Reference score</text>',
        f'<text class="tp-parity__title" transform="translate(12 {PLOT_TOP + PLOT_SIZE / 2}) rotate(-90)" '
        'text-anchor="middle">Measured score</text>',
        '<g class="tp-parity__points" clip-path="url(#tp-parity-clip)">',
    ]
    for row in rows:
        name, ref, ours = row["checkpoint"], float(row["reference"]), float(row["score"])
        data = {
            "task": row["task"],
            "name": name,
            "dataset": dataset_label(row["dataset"]),
            "metric": metric_label(row["task"], row["dataset"]),
            "reference": row["reference"],
            "score": row["score"],
            "gap": signed(ours - ref),
            "params": row["params"],
            "license": LICENSE_LABELS.get(row["license"], row["license"]),
        }
        attributes = " ".join(f'data-{key}="{html.escape(value)}"' for key, value in data.items())
        # Not an SVG `<a>`: instant navigation rewrites every `href` of an incoming page, and an SVG link's is
        # read-only, so the page swap would throw and stall.
        href = relative_link(context, TASKS[row["task"]][1]) + "#" + html.escape(name)
        label = html.escape(f"{name}: measured {row['score']}, reference {row['reference']}")
        svg.append(
            f'<g class="tp-parity__point" tabindex="0" role="link" data-href="{href}" aria-label="{label}" '
            f'{attributes}><path transform="translate({x_of(ref):.1f} {y_of(ours):.1f})" '
            f'd="{TASKS[row["task"]][2]}"/></g>'
        )
    svg.append("</g></svg>")
    # The reset button sits in the top-left corner of the plot area, which a parity plot leaves empty.
    reset = (
        f'<button type="button" class="tp-parity__reset" style="left: {100 * (PLOT_LEFT + 8) / width:.2f}%; '
        f'top: {100 * (PLOT_TOP + 8) / height:.2f}%" hidden>Reset zoom</button>'
    )

    # A single task needs no legend; with several, each key is a toggle that shows or hides its points.
    tasks = list(dict.fromkeys(row["task"] for row in rows))
    keys = "".join(
        f'<button type="button" class="tp-parity__key" data-task="{t}" aria-pressed="true">'
        f'<svg viewBox="-6 -6 12 12"><path d="{TASKS[t][2]}"/></svg>{TASKS[t][0]}</button>'
        for t in tasks
    )
    legend = f'<div class="tp-parity__legend">{keys}</div>' if len(tasks) > 1 else ""
    return (
        f'<figure class="tp-parity">{legend}'
        f'<div class="tp-parity__plot">{"".join(svg)}{reset}<div class="tp-parity__tooltip" hidden></div></div>'
        "</figure>"
    )
