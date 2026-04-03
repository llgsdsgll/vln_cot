#!/usr/bin/env python3
"""Compare scene overlap between a split file and a scene directory."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


DEFAULT_SPLIT_CANDIDATES = [
    Path("/home/gs/my_test/vln_dataset/HM3D/annotations/splits/train_split.json"),
    Path("/home/gs/my_test/vln_dataset/HM3D/annotations/splits/train_split.txt"),
]
DEFAULT_SCENE_DIR = Path(
    "/home/gs/my_test/vln_dataset/data/scene_datasets/mp3d/mp3d_habitat/mp3d"
)
HM3D_SCENE_RE = re.compile(r"^\d{5}-([A-Za-z0-9]+)(?:_sub\d+)?$")
SCENE_TOKEN_RE = re.compile(r"[A-Za-z0-9]{11}")
SCENE_KEYS = {
    "scene",
    "scene_id",
    "scene_ids",
    "scene_name",
    "scene_names",
    "content_scenes",
    "episodes",
}
KNOWN_SUFFIXES = (
    ".basis.glb",
    ".semantic.glb",
    ".semantic.txt",
    ".glb",
    ".navmesh",
    ".json.gz",
    ".json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a split file against a scene directory and count overlapping scenes."
        )
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=None,
        help=(
            "Path to the split file. Supports .txt and .json. "
            "If omitted, the script tries the built-in default candidates."
        ),
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=DEFAULT_SCENE_DIR,
        help="Directory that contains one subdirectory per scene.",
    )
    parser.add_argument(
        "--show-overlap",
        action="store_true",
        help="Print the overlapping scene IDs.",
    )
    return parser.parse_args()


def resolve_split_path(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        if explicit_path.exists():
            return explicit_path
        raise FileNotFoundError(f"Split file not found: {explicit_path}")

    for candidate in DEFAULT_SPLIT_CANDIDATES:
        if candidate.exists():
            return candidate

    checked = "\n".join(f"  - {path}" for path in DEFAULT_SPLIT_CANDIDATES)
    raise FileNotFoundError(
        "Could not find a default split file. Checked:\n" f"{checked}"
    )


def strip_known_suffixes(text: str) -> str:
    result = text.strip()
    changed = True
    while changed and result:
        changed = False
        for suffix in KNOWN_SUFFIXES:
            if result.endswith(suffix):
                result = result[: -len(suffix)]
                changed = True
                break
    return result


def normalize_scene_id(raw_value: str) -> str | None:
    text = str(raw_value).strip()
    if not text:
        return None

    base = strip_known_suffixes(Path(text).name)
    hm3d_match = HM3D_SCENE_RE.match(base)
    if hm3d_match:
        return hm3d_match.group(1)

    if base.startswith("mp3d__"):
        deduped_tokens = list(dict.fromkeys(SCENE_TOKEN_RE.findall(base)))
        if len(deduped_tokens) == 1:
            return deduped_tokens[0]

    if SCENE_TOKEN_RE.fullmatch(base):
        return base

    deduped_tokens = list(dict.fromkeys(SCENE_TOKEN_RE.findall(base)))
    if len(deduped_tokens) == 1:
        return deduped_tokens[0]

    return None


def iter_json_strings(obj: object) -> list[str]:
    results: list[str] = []

    if isinstance(obj, str):
        return [obj]

    if isinstance(obj, list):
        for item in obj:
            results.extend(iter_json_strings(item))
        return results

    if isinstance(obj, dict):
        scene_key_found = False
        for key, value in obj.items():
            if key in SCENE_KEYS:
                scene_key_found = True
                results.extend(iter_json_strings(value))
        if scene_key_found:
            return results
        for value in obj.values():
            results.extend(iter_json_strings(value))
        return results

    return results


def load_split_scene_ids(split_path: Path) -> set[str]:
    if split_path.suffix.lower() == ".json":
        data = json.loads(split_path.read_text(encoding="utf-8"))
        candidates = iter_json_strings(data)
    else:
        candidates = split_path.read_text(encoding="utf-8").splitlines()

    scene_ids = set()
    for candidate in candidates:
        normalized = normalize_scene_id(candidate)
        if normalized is not None:
            scene_ids.add(normalized)
    return scene_ids


def load_directory_scene_ids(scene_dir: Path) -> set[str]:
    if not scene_dir.is_dir():
        raise NotADirectoryError(f"Scene directory not found: {scene_dir}")

    scene_ids = set()
    for child in scene_dir.iterdir():
        if child.is_dir():
            normalized = normalize_scene_id(child.name)
            if normalized is not None:
                scene_ids.add(normalized)
    return scene_ids


def main() -> int:
    args = parse_args()

    try:
        split_path = resolve_split_path(args.split_path)
        split_scene_ids = load_split_scene_ids(split_path)
        directory_scene_ids = load_directory_scene_ids(args.scene_dir)
    except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    overlap = sorted(split_scene_ids & directory_scene_ids)

    print(f"Split file: {split_path}")
    print(f"Split unique scene IDs: {len(split_scene_ids)}")
    print(f"Scene directory: {args.scene_dir}")
    print(f"Directory scene IDs: {len(directory_scene_ids)}")
    print(f"Overlap count: {len(overlap)}")

    if overlap:
        if args.show_overlap:
            print("Overlapping scene IDs:")
            for scene_id in overlap:
                print(scene_id)
        else:
            preview = ", ".join(overlap[:10])
            if len(overlap) > 10:
                preview += ", ..."
            print(f"Overlap preview: {preview}")
    else:
        print("No overlapping scenes found.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
