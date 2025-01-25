from pathlib import Path

from mkdocs.config import Config

PROJECT_ROOT = Path(__file__).parent.parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"


def on_pre_build(config: Config) -> None:
    add_notebooks()


def add_notebooks() -> None:
    # Remove all notebooks that are no longer in the notebooks directory
    for notebook_path in DOCS_DIR.glob("notebooks/*.ipynb"):
        if not (PROJECT_ROOT / "notebooks" / notebook_path.name).is_file():
            notebook_path.unlink()

    # Copy all notebooks from the notebooks directory to the docs directory
    for notebook_path in Path(PROJECT_ROOT, "notebooks").glob("*.ipynb"):
        if notebook_path.name.startswith("[DEBUG]") or notebook_path.name.startswith("[WIP]"):
            continue

        out_path = DOCS_DIR / "notebooks" / notebook_path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not out_path.is_file() or out_path.read_text(encoding="utf-8") != notebook_path.read_text(encoding="utf-8"):
            out_path.write_text(notebook_path.read_text(encoding="utf-8"), encoding="utf-8")
