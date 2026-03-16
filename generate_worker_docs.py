#!/usr/bin/env python3
"""Generate documentation for NimbusImage workers.

Creates:
- {worker_dir}/{worker_name}.md  for each worker
- registry.md at the repo root

Usage:
    # Regenerate all docs
    python generate_worker_docs.py

    # Regenerate docs for specific workers only (registry always regenerated)
    python generate_worker_docs.py --workers cellposesam blob_intensity_worker

    # Regenerate registry only
    python generate_worker_docs.py --registry-only
"""

import argparse
import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

REPO_ROOT = Path(__file__).parent
WORKERS_ROOT = REPO_ROOT / "workers"

# Workers that serve as test/sample workers (not production annotation workers)
TEST_WORKER_NAMES = {
    "random_squares",
    "sample_interface",
    "test_multiple_annotation",
    "test_multiple_annotation_M1",
    "random_point",
    "random_point_annotation_M1",
    "annulus_generator_M1",
}

# Workers built by build_machine_learning_workers.sh (not in docker-compose.yml)
ML_WORKER_NAMES = {
    "cellpose",
    "cellpose_train",
    "cellposesam",
    "condensatenet",
    "deepcell",
    "piscis",
    "sam2_automatic_mask_generator",
    "sam2_fewshot_segmentation",
    "sam2_propagate",
    "sam2_refine",
    "sam2_video",
    "sam_automatic_mask_generator",
    "sam_fewshot_segmentation",
    "stardist",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class InterfaceParam:
    name: str
    type: str
    required: bool = False
    min: Optional[float] = None
    max: Optional[float] = None
    default: Optional[str] = None
    unit: Optional[str] = None
    items: Optional[list] = None
    tooltip: Optional[str] = None
    placeholder: Optional[str] = None
    dynamic: bool = False   # True when the value contains a runtime expression


@dataclass
class WorkerInfo:
    name: str
    dir_path: Path
    worker_type: str          # "annotation" or "property"
    category: Optional[str] = None  # "blobs"/"points"/"lines"/"connections"
    description: Optional[str] = None
    interface_name: Optional[str] = None
    interface_category: Optional[str] = None
    annotation_shape: Optional[str] = None
    is_property_worker: bool = False
    is_annotation_worker: bool = False
    params: list = field(default_factory=list)
    has_gpu: bool = False
    has_tests: bool = False
    is_test_worker: bool = False
    has_dynamic_params: bool = False   # True when >=1 param couldn't be statically parsed
    compose_service: Optional[str] = None  # Docker Compose service name (may differ from dir name)
    is_ml_worker: bool = False  # Built by build_machine_learning_workers.sh


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _strip_html(text: str) -> str:
    """Remove HTML tags and normalise whitespace."""
    return re.sub(r"<[^>]+>", "", text).strip()


def extract_docker_labels(dockerfile_path: Path) -> dict:
    """Extract LABEL key=value pairs from a Dockerfile."""
    labels: dict = {}
    if not dockerfile_path.exists():
        return labels

    content = dockerfile_path.read_text()

    # Detect GPU requirement
    if re.search(r"nvidia|cuda", content, re.IGNORECASE):
        labels["_has_gpu"] = True

    # Find all LABEL blocks (may span multiple lines via backslash continuation)
    label_block_re = re.compile(r"^LABEL\s+((?:[^\n\\]|\\\n)+)", re.MULTILINE)
    for block_match in label_block_re.finditer(content):
        block = block_match.group(1)
        # Collapse backslash-newline continuations
        block = re.sub(r"\\\n\s*", " ", block)
        # Parse key="value" or key=value pairs
        for m in re.finditer(r'(\w+)=(?:"([^"]*)"|([\S]*))', block):
            key = m.group(1)
            val = m.group(2) if m.group(2) is not None else (m.group(3) or "")
            labels[key] = val

    return labels


def extract_interface_info(entrypoint_path: Path):
    """
    Parse entrypoint.py to extract interface description and parameters.

    Handles workers where the interface dict contains non-literal expressions
    (e.g. a `models` variable built at runtime) by evaluating each key-value
    pair individually and marking dynamic ones.

    Returns:
        description (str | None)
        params (list[InterfaceParam])
        has_dynamic (bool)  -- True if any param couldn't be statically parsed
    """
    if not entrypoint_path.exists():
        return None, [], False

    source = entrypoint_path.read_text()
    description: Optional[str] = None
    params: list = []
    has_dynamic = False

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None, [], False

    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "interface"):
            continue
        for stmt in node.body:
            if not (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == "interface"
                and isinstance(stmt.value, ast.Dict)
            ):
                continue

            dict_node = stmt.value

            # Process each key-value pair individually so that partially-dynamic
            # dicts still yield as much information as possible.
            for key_node, val_node in zip(dict_node.keys, dict_node.values):
                try:
                    key = ast.literal_eval(key_node)
                except (ValueError, TypeError):
                    has_dynamic = True
                    continue

                # Try to evaluate the whole value dict
                try:
                    val = ast.literal_eval(val_node)
                    is_dynamic_val = False
                except (ValueError, TypeError):
                    # Has dynamic sub-expressions — recover what we can
                    val = {}
                    is_dynamic_val = True
                    has_dynamic = True
                    if isinstance(val_node, ast.Dict):
                        for sk_node, sv_node in zip(val_node.keys, val_node.values):
                            try:
                                sk = ast.literal_eval(sk_node)
                                sv = ast.literal_eval(sv_node)
                                val[sk] = sv
                            except (ValueError, TypeError):
                                pass

                if not isinstance(val, dict):
                    continue

                field_type = val.get("type", "")

                if field_type == "notes":
                    description = _strip_html(val.get("value", ""))
                    continue

                vue_attrs = val.get("vueAttrs", {})
                if not isinstance(vue_attrs, dict):
                    vue_attrs = {}

                tooltip = val.get("tooltip") or vue_attrs.get("label")
                placeholder = vue_attrs.get("placeholder")

                param = InterfaceParam(
                    name=key,
                    type=field_type if field_type else "unknown",
                    required=val.get("required", False),
                    tooltip=tooltip,
                    default=val.get("default"),
                    min=val.get("min"),
                    max=val.get("max"),
                    unit=val.get("unit"),
                    items=val.get("items"),
                    placeholder=placeholder,
                    dynamic=is_dynamic_val,
                )
                params.append(param)

    # Fallback regex for description when no interface dict was found / parsed
    if description is None:
        for pattern in [
            r"'type':\s*'notes'[^}]*?'value':\s*'([^']+)'",
            r"'type':\s*'notes'[^}]*?'value':\s*\"([^\"]+)\"",
        ]:
            m = re.search(pattern, source, re.DOTALL)
            if m:
                description = _strip_html(m.group(1))
                break

    return description, params, has_dynamic


# ---------------------------------------------------------------------------
# Docker Compose service name lookup
# ---------------------------------------------------------------------------

def _load_compose_service_map() -> dict:
    """
    Return a mapping of {worker_dir_name: compose_service_name} by parsing
    docker-compose.yml.  Falls back to an empty dict if yaml is unavailable
    or the file doesn't exist.
    """
    compose_path = REPO_ROOT / "docker-compose.yml"
    if not compose_path.exists() or not _YAML_AVAILABLE:
        return {}

    try:
        data = yaml.safe_load(compose_path.read_text())
    except Exception:
        return {}

    mapping: dict = {}
    for svc_name, cfg in (data.get("services") or {}).items():
        build = cfg.get("build", {})
        if not isinstance(build, dict):
            continue
        dockerfile = build.get("dockerfile", "")
        # Dockerfile path is like ./workers/annotations/cellposesam/Dockerfile
        # The parent directory is the worker directory.
        parts = Path(dockerfile).parts
        if len(parts) < 2:
            continue
        worker_dir = parts[-2]
        if worker_dir not in ("base_docker_images", "tests"):
            # Only record the first match (there shouldn't be duplicates).
            mapping.setdefault(worker_dir, svc_name)

    return mapping


# ---------------------------------------------------------------------------
# Worker discovery
# ---------------------------------------------------------------------------

def discover_workers() -> list:
    """Walk the workers/ tree and return a list of WorkerInfo objects."""
    infos: list = []
    svc_map = _load_compose_service_map()

    # --- Annotation workers ---
    ann_dir = WORKERS_ROOT / "annotations"
    if ann_dir.exists():
        for worker_dir in sorted(ann_dir.iterdir()):
            if not worker_dir.is_dir():
                continue
            entrypoint = worker_dir / "entrypoint.py"
            if not entrypoint.exists():
                continue

            labels = extract_docker_labels(worker_dir / "Dockerfile")
            desc, params, has_dynamic = extract_interface_info(entrypoint)

            infos.append(WorkerInfo(
                name=worker_dir.name,
                dir_path=worker_dir,
                worker_type="annotation",
                category=None,
                description=desc or labels.get("description"),
                interface_name=labels.get("interfaceName"),
                interface_category=labels.get("interfaceCategory"),
                annotation_shape=labels.get("annotationShape"),
                is_property_worker=False,
                is_annotation_worker=True,
                params=params,
                has_gpu="_has_gpu" in labels,
                has_tests=(worker_dir / "tests").exists(),
                is_test_worker=worker_dir.name in TEST_WORKER_NAMES,
                has_dynamic_params=has_dynamic,
                compose_service=svc_map.get(worker_dir.name),
                is_ml_worker=worker_dir.name in ML_WORKER_NAMES,
            ))

    # --- Property workers ---
    prop_dir = WORKERS_ROOT / "properties"
    if prop_dir.exists():
        for category_dir in sorted(prop_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            for worker_dir in sorted(category_dir.iterdir()):
                if not worker_dir.is_dir():
                    continue
                entrypoint = worker_dir / "entrypoint.py"
                if not entrypoint.exists():
                    continue

                labels = extract_docker_labels(worker_dir / "Dockerfile")
                desc, params, has_dynamic = extract_interface_info(entrypoint)

                infos.append(WorkerInfo(
                    name=worker_dir.name,
                    dir_path=worker_dir,
                    worker_type="property",
                    category=category_dir.name,
                    description=desc or labels.get("description"),
                    interface_name=labels.get("interfaceName"),
                    interface_category=labels.get("interfaceCategory"),
                    annotation_shape=labels.get("annotationShape"),
                    is_property_worker=True,
                    is_annotation_worker=False,
                    params=params,
                    has_gpu="_has_gpu" in labels,
                    has_tests=(worker_dir / "tests").exists(),
                    is_test_worker=False,
                    has_dynamic_params=has_dynamic,
                    compose_service=svc_map.get(worker_dir.name),
                ))

    return infos


# ---------------------------------------------------------------------------
# Document generators
# ---------------------------------------------------------------------------

def _param_table(params: list, has_dynamic: bool) -> str:
    """Render a list of InterfaceParam objects as a Markdown table."""
    if not params:
        if has_dynamic:
            return (
                "_Interface parameters are dynamically computed at runtime "
                "(e.g. model lists fetched from the server). "
                "See `entrypoint.py` -> `interface()` for details._\n"
            )
        return "_No configurable parameters._\n"

    rows = [
        "| Parameter | Type | Required | Default | Range / Options | Description |",
        "|-----------|------|:--------:|---------|-----------------|-------------|",
    ]

    for p in params:
        required = "Yes" if p.required else ""
        default_val = p.default
        if default_val is None:
            default = "--"
        elif default_val == -1:
            default = "none"
        else:
            default = str(default_val)

        # Range / options column
        if p.dynamic and p.items is None:
            range_opts = "_runtime_"
        elif p.items is not None:
            shown = [str(i) for i in p.items[:6]]
            suffix = ", ..." if len(p.items) > 6 else ""
            range_opts = ", ".join(shown) + suffix
        elif p.min is not None or p.max is not None:
            lo = str(p.min) if p.min is not None else "--"
            hi = str(p.max) if p.max is not None else "--"
            range_opts = f"{lo} - {hi}"
            if p.unit:
                range_opts += f" {p.unit}"
        else:
            range_opts = "--"

        # Description: prefer tooltip, fallback to placeholder
        tooltip = p.tooltip or p.placeholder or ""
        tooltip = tooltip.replace("\n", " ").strip()
        if len(tooltip) > 130:
            tooltip = tooltip[:127] + "..."

        dyn_note = " *(runtime)*" if p.dynamic else ""
        rows.append(
            f"| **{p.name}**{dyn_note} | `{p.type}` | {required} | {default} | {range_opts} | {tooltip} |"
        )

    result = "\n".join(rows) + "\n"
    if has_dynamic:
        result += "\n> **(runtime)** = option list is fetched at runtime (e.g. available trained models).\n"
    return result


def generate_worker_doc(info: WorkerInfo) -> str:
    """Produce the full Markdown content for a single worker's .md file."""
    lines: list = []

    display_name = info.interface_name or info.name.replace("_", " ").title()
    lines += [f"# {display_name}", ""]

    # --- Metadata ---
    type_label = "Annotation" if info.worker_type == "annotation" else "Property"
    lines.append(f"**Type:** {type_label} Worker  ")
    if info.category:
        lines.append(f"**Category:** {info.category.title()}  ")
    if info.annotation_shape:
        lines.append(f"**Annotation Shape:** `{info.annotation_shape}`  ")
    if info.interface_category:
        lines.append(f"**Interface Category:** {info.interface_category}  ")
    if info.has_gpu:
        lines.append("**GPU Support:** Yes (NVIDIA CUDA)  ")
    if info.is_test_worker:
        lines.append("**Role:** Test / Sample Worker  ")
    lines.append("")

    # --- Description ---
    if info.description:
        lines += ["## Description", "", info.description, ""]

    # --- Interface Parameters ---
    lines += ["## Interface Parameters", ""]
    lines.append(_param_table(info.params, info.has_dynamic_params))

    # --- Files ---
    lines += ["## Files", ""]
    file_rows = [
        "| File | Description |",
        "|------|-------------|",
        "| `entrypoint.py` | Main worker logic -- `interface()` defines the UI, `compute()` runs the analysis |",
    ]
    for fname, fdesc in [
        ("Dockerfile", "Docker build configuration (x86_64 / production)"),
        ("Dockerfile_M1", "ARM64 / Apple Silicon Docker configuration"),
        ("environment.yml", "Conda environment dependencies"),
    ]:
        if (info.dir_path / fname).exists():
            file_rows.append(f"| `{fname}` | {fdesc} |")
    if info.has_tests:
        file_rows.append("| `tests/` | pytest test suite with `Dockerfile_Test` |")
    lines += file_rows + [""]

    # --- Building ---
    # Use the Docker Compose service name when available.  ML workers are built
    # by build_machine_learning_workers.sh (no per-worker argument).  Workers
    # without any known build path omit the section entirely.
    if info.is_ml_worker:
        lines += ["## Building", ""]
        lines += [
            "This worker is built by the ML workers script (builds all ML workers together):",
            "",
            "```bash",
            "./build_machine_learning_workers.sh",
            "```",
            "",
        ]
    elif info.is_test_worker and info.compose_service:
        build_id = info.compose_service
        lines += ["## Building", ""]
        lines += [
            "```bash",
            f"./build_test_workers.sh {build_id}",
            "```",
            "",
        ]
    elif info.compose_service:
        build_id = info.compose_service
        lines += ["## Building", ""]
        lines += [
            "```bash",
            f"./build_workers.sh {build_id}",
            "```",
            "",
        ]
    # else: no known build command — omit the section

    # --- Testing ---
    if info.has_tests and info.compose_service:
        build_id = info.compose_service
        lines += [
            "## Testing",
            "",
            "```bash",
            f"./build_workers.sh --build-and-run-tests {build_id}",
            "```",
            "",
        ]

    # --- Footer ---
    lines += [
        "---",
        "_This file is auto-generated by `generate_worker_docs.py`. "
        "To update, edit `entrypoint.py` and re-run the generator, "
        "or open a PR -- the Claude Code hook regenerates docs automatically._",
        "",
    ]

    return "\n".join(lines)


def _doc_filename(worker_name: str) -> str:
    """Return the UPPERCASE documentation filename for a worker (e.g. 'deconwolf' -> 'DECONWOLF.md')."""
    return f"{worker_name.upper()}.md"


def _existing_doc_path(worker_dir: Path, worker_name: str) -> Optional[Path]:
    """Find the existing doc file for a worker.

    Checks for any .md file in the worker directory that looks like documentation
    (excluding README.md and EXPERIMENT_RESULTS.md). This handles cases where the
    doc filename doesn't exactly match the directory name (e.g. BLOB_INTENSITY.md
    in blob_intensity_worker/).

    Returns the Path if found, None otherwise.
    """
    # First check the exact expected name
    upper_path = worker_dir / _doc_filename(worker_name)
    if upper_path.exists():
        return upper_path

    # Check for any doc-like .md file in the directory
    skip_names = {"README.md", "EXPERIMENT_RESULTS.md"}
    for md_file in worker_dir.glob("*.md"):
        if md_file.name not in skip_names:
            return md_file

    return None


def generate_registry(workers: list) -> str:
    """Produce the full Markdown content for registry.md."""
    ann = [w for w in workers if w.worker_type == "annotation" and not w.is_test_worker]
    test = [w for w in workers if w.worker_type == "annotation" and w.is_test_worker]
    prop = [w for w in workers if w.worker_type == "property"]
    blobs = [w for w in prop if w.category == "blobs"]
    points = [w for w in prop if w.category == "points"]
    lines_ = [w for w in prop if w.category == "lines"]
    conns = [w for w in prop if w.category == "connections"]

    lines: list = []
    lines += [
        "# Worker Registry",
        "",
        "Complete index of all workers in the NimbusImage ImageAnalysisProject.",
        "Auto-generated by `generate_worker_docs.py` -- do not edit manually.",
        "",
        "## Summary",
        "",
        "| Category | Count |",
        "|----------|------:|",
        f"| Annotation Workers | {len(ann)} |",
        f"| Property Workers -- Blobs | {len(blobs)} |",
        f"| Property Workers -- Points | {len(points)} |",
        f"| Property Workers -- Lines | {len(lines_)} |",
        f"| Property Workers -- Connections | {len(conns)} |",
        f"| Test / Sample Workers | {len(test)} |",
        f"| **Total** | **{len(workers)}** |",
        "",
    ]

    def _row_ann(w: WorkerInfo) -> str:
        display = w.interface_name or w.name.replace("_", " ").title()
        desc = _strip_html(w.description or "")
        if len(desc) > 90:
            desc = desc[:87] + "..."
        gpu = "Yes" if w.has_gpu else ""
        tests = "Yes" if w.has_tests else ""
        rel = w.dir_path.relative_to(REPO_ROOT)
        link = f"[docs]({rel}/{_doc_filename(w.name)})"
        return f"| {display} | {desc} | {gpu} | {tests} | {link} |"

    def _row_prop(w: WorkerInfo) -> str:
        display = w.interface_name or w.name.replace("_", " ").title()
        desc = _strip_html(w.description or "")
        if len(desc) > 90:
            desc = desc[:87] + "..."
        tests = "Yes" if w.has_tests else ""
        rel = w.dir_path.relative_to(REPO_ROOT)
        link = f"[docs]({rel}/{_doc_filename(w.name)})"
        return f"| {display} | {desc} | {tests} | {link} |"

    def _ann_section(title: str, intro: str, worker_list: list) -> list:
        out: list = [f"## {title}", "", intro, ""]
        out += [
            "| Worker | Description | GPU | Tests | Docs |",
            "|--------|-------------|-----|-------|------|",
        ]
        for w in worker_list:
            out.append(_row_ann(w))
        out.append("")
        return out

    def _prop_section(title: str, intro: str, worker_list: list) -> list:
        out: list = [f"## {title}", "", intro, ""]
        out += [
            "| Worker | Description | Tests | Docs |",
            "|--------|-------------|-------|------|",
        ]
        for w in worker_list:
            out.append(_row_prop(w))
        out.append("")
        return out

    lines += _ann_section(
        "Annotation Workers",
        "Create new annotations by segmenting images or connecting existing annotations.",
        ann,
    )
    lines += _prop_section(
        "Property Workers -- Blobs",
        "Compute properties on polygon / blob annotations.",
        blobs,
    )
    lines += _prop_section(
        "Property Workers -- Points",
        "Compute properties on point annotations.",
        points,
    )
    lines += _prop_section(
        "Property Workers -- Lines",
        "Compute properties on line annotations.",
        lines_,
    )
    lines += _prop_section(
        "Property Workers -- Connections",
        "Compute properties based on relationships between annotations.",
        conns,
    )
    lines += _ann_section(
        "Test / Sample Workers",
        "Workers used for testing and demonstration -- not intended for production use.",
        test,
    )

    lines += [
        "---",
        "_Auto-generated by `generate_worker_docs.py`. "
        "Run `python generate_worker_docs.py` from the repo root to refresh._",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate NimbusImage worker documentation")
    parser.add_argument(
        "--workers", nargs="*", metavar="WORKER_NAME",
        help="Only regenerate docs for these worker names (registry always regenerated)",
    )
    parser.add_argument(
        "--registry-only", action="store_true",
        help="Only regenerate registry.md, skip per-worker docs",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing documentation files (by default, only creates stubs for workers without docs)",
    )
    args = parser.parse_args()

    print("Discovering workers...")
    all_workers = discover_workers()
    print(f"Found {len(all_workers)} workers")

    if not args.registry_only:
        if args.workers:
            target = [w for w in all_workers if w.name in args.workers]
            missing = set(args.workers) - {w.name for w in target}
            if missing:
                print(f"WARNING: workers not found: {', '.join(sorted(missing))}", file=sys.stderr)
        else:
            target = all_workers

        created = 0
        skipped = 0
        for info in target:
            doc_path = info.dir_path / _doc_filename(info.name)
            existing = _existing_doc_path(info.dir_path, info.name)

            if existing and not args.force:
                skipped += 1
                continue

            content = generate_worker_doc(info)
            doc_path.write_text(content)
            print(f"  Wrote {doc_path.relative_to(REPO_ROOT)}")
            created += 1

        if skipped:
            print(f"  Skipped {skipped} workers with existing docs (use --force to overwrite)")
        if created:
            print(f"  Created {created} new doc stubs")

    print("Generating registry.md...")
    (REPO_ROOT / "REGISTRY.md").write_text(generate_registry(all_workers))
    print("  Wrote REGISTRY.md")
    print("Done.")


if __name__ == "__main__":
    main()
