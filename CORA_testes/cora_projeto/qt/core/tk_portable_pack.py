"""Portable migration pack from Tk to Qt.

This module keeps the pieces that can be reused as-is from the Tk frontend:
- domain constants and presets
- dataclasses for grouped/processed items
- pure NumPy/OpenCV helpers (no Tk dependency)
- grouping helper functions that were methods in Tk
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ...matlab_style_cora import BASE_IMAGE_PATH, CORAArtifacts, AreaResults
from ...services.grouping_service import parse_group_image


# PT: DESTINO: | EN: TARGET:
# PT: - Este bloco de constantes/presets deve ser consumido por: | EN: - This constants/presets block should be consumed by:
# PT:   1) qt/cora_interface_qt.py (estado global e fluxo) | EN:   1) qt/cora_interface_qt.py (global state and workflow)
# PT:   2) qt/ui/roi_page_qt.py (labels/opcoes de preview, se necessario) | EN:   2) qt/ui/roi_page_qt.py (preview labels/options, if needed)
# PT:   3) qt/ui/roi_editor.py (parametros de edicao ROI) | EN:   3) qt/ui/roi_editor.py (ROI editing parameters)
SUPPORTED_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
TIME_ORDER = ("0h", "24h", "48h")
OUTPUT_DIR_NAME = "_cora_resultados"
ROI_OFFSET_STEP_PX = 4.0
ROI_POLY_MAX_POINTS = 600
ROI_LOCAL_DRAG_SIGMA_FRAC = 0.035
ROI_LOCAL_DRAG_MIN_SIGMA = 6.0
ROI_REGION_SIMPLIFY_STEP = 2
ROI_BRUSH_RADIUS_DEFAULT = 12
ROI_BRUSH_RADIUS_MIN = 2
ROI_BRUSH_RADIUS_MAX = 120
ROI_EDITOR_MAX_FPS = 45.0
ROI_CURSOR_MAX_FPS = 45.0
ROI_DRAG_POINT_REFRESH_FPS = 12.0
ROI_PERIMETER_POINT_CAP = 1800
ROI_PERIMETER_POINT_SIZE = 14
ROI_CONTOUR_THICKNESS_PX = 2
# ROI editor work scale (0.50 = 50% of original resolution).
ROI_EDITOR_WORK_SCALE = 0.50
ROI_EDITOR_PROFILE_PRESETS: dict[str, dict[str, object]] = {
    "rapido_50": {
        "label": "Rapido (50%)",
        "work_scale": 0.50,
        "editor_max_fps": 45.0,
        "cursor_max_fps": 45.0,
        "drag_point_refresh_fps": 10.0,
        "perimeter_point_cap": 1000,
        "perimeter_point_size": 12,
        "default_contour_mode": "contour",
    },
    "balanceado_75": {
        "label": "Balanceado (75%)",
        "work_scale": 0.75,
        "editor_max_fps": 45.0,
        "cursor_max_fps": 45.0,
        "drag_point_refresh_fps": 12.0,
        "perimeter_point_cap": 1600,
        "perimeter_point_size": 13,
        "default_contour_mode": "contour",
    },
    "qualidade_100": {
        "label": "Qualidade (100%)",
        "work_scale": 1.00,
        "editor_max_fps": 45.0,
        "cursor_max_fps": 45.0,
        "drag_point_refresh_fps": 14.0,
        "perimeter_point_cap": 2400,
        "perimeter_point_size": 14,
        "default_contour_mode": "points",
    },
}
ROI_EDITOR_PROFILE_DEFAULT = "rapido_50"
ROI_EDITOR_PROFILE_LABEL_TO_KEY = {
    str(cfg.get("label", "")).strip().lower(): key
    for key, cfg in ROI_EDITOR_PROFILE_PRESETS.items()
}
PREVIEW_MODE_PRESETS: dict[str, str] = {
    "contour": "So contorno",
    "filled": "Area preenchida",
}
PREVIEW_MODE_DEFAULT = "contour"
PREVIEW_MODE_LABEL_TO_KEY = {
    label.strip().lower(): key
    for key, label in PREVIEW_MODE_PRESETS.items()
}
PREVIEW_COLOR_MODE_PRESETS: dict[str, str] = {
    "shared": "Cor unica (todas)",
    "per_time": "Cores separadas por tipo",
}
PREVIEW_COLOR_MODE_DEFAULT = "shared"
PREVIEW_COLOR_MODE_LABEL_TO_KEY = {
    label.strip().lower(): key
    for key, label in PREVIEW_COLOR_MODE_PRESETS.items()
}
PREVIEW_CONTOUR_COLOR_SHARED_DEFAULT: tuple[int, int, int] = (255, 0, 0)
PREVIEW_CONTOUR_COLORS_DEFAULT: dict[str, tuple[int, int, int]] = {
    "0h": PREVIEW_CONTOUR_COLOR_SHARED_DEFAULT,
    "24h": PREVIEW_CONTOUR_COLOR_SHARED_DEFAULT,
    "48h": PREVIEW_CONTOUR_COLOR_SHARED_DEFAULT,
}
APP_ID = "CORA"


@dataclass
class ProcessedTimepoint:
    path: Path
    results: AreaResults
    artifacts: CORAArtifacts


@dataclass
class GroupingEntry:
    path: Path
    rel_path: str
    group_key: str
    time_tag: str | None
    selected: bool
    source: str


# PT: DESTINO: | EN: TARGET:
# PT: - Helpers de mascara/geometria abaixo devem ir para o editor de ROI Qt (qt/ui/roi_editor.py) quando voce portar _open_roi_editor.
# EN: - The mask/geometry helpers below should move to the Qt ROI editor (qt/ui/roi_editor.py) when _open_roi_editor is ported.
def strip_accents(text: str) -> str:
    """Remove accents for robust comparison/sorting."""
    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


# Backward-compatible alias (same name used in Tk class).
_strip_accents = strip_accents


def mask_to_polygon(mask: np.ndarray) -> np.ndarray:
    """Convert binary mask to closed polygon (largest external contour)."""
    bin255 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(bin255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), dtype=np.float32)

    poly = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float32)
    if poly.shape[0] > ROI_POLY_MAX_POINTS:
        sample_idx = np.floor(
            np.linspace(0.0, float(poly.shape[0]), num=ROI_POLY_MAX_POINTS, endpoint=False)
        ).astype(np.int32)
        poly = poly[sample_idx]
    return poly


def polygon_to_mask(shape: tuple[int, int], verts: np.ndarray) -> np.ndarray:
    """Rasterize polygon vertices into a binary mask with the requested shape."""
    h, w = shape
    if verts.shape[0] < 3:
        return np.zeros(shape, dtype=bool)

    pts = np.round(verts).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    mask_u8 = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask_u8, [pts], 255)
    return mask_u8 > 0


def resize_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize mask with nearest-neighbor interpolation to preserve labels."""
    th, tw = target_shape
    if mask.shape == target_shape:
        return mask.astype(bool)
    return cv2.resize(mask.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST) > 0


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component."""
    src = np.asarray(mask, dtype=bool)
    if src.ndim != 2:
        raise ValueError("mask must be 2D.")
    if not np.any(src):
        return np.zeros_like(src, dtype=bool)

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(src.astype(np.uint8), connectivity=8)
    if labels_count <= 2:
        return src

    areas = stats[1:, cv2.CC_STAT_AREA]
    keep_label = int(np.argmax(areas) + 1)
    return labels == keep_label


def offset_mask_sdf(mask: np.ndarray, offset_px: float, smooth_sigma: float = 0.0) -> np.ndarray:
    """Expand/shrink mask via signed distance field with optional smoothing."""
    src = np.asarray(mask).astype(bool)
    if src.ndim != 2:
        raise ValueError("mask must be 2D.")
    if not np.any(src):
        return np.zeros_like(src, dtype=bool)

    src_u8 = src.astype(np.uint8) * 255
    src_pad = cv2.copyMakeBorder(src_u8, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    inv_pad = cv2.bitwise_not(src_pad)

    dist_in = cv2.distanceTransform(src_pad, cv2.DIST_L2, 5)[1:-1, 1:-1]
    dist_out = cv2.distanceTransform(inv_pad, cv2.DIST_L2, 5)[1:-1, 1:-1]
    sdf = dist_in - dist_out

    sigma = float(smooth_sigma)
    if sigma > 0.0:
        sdf = cv2.GaussianBlur(
            sdf,
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REPLICATE,
        )

    return sdf >= (-float(offset_px))


def default_folder_from_base_path() -> str:
    """Derive default input folder from BASE_IMAGE_PATH."""
    base = str(BASE_IMAGE_PATH).strip()
    if not base:
        return ""
    path = Path(base)
    if path.suffix:
        return str(path.parent)
    return str(path)


# PT: DESTINO: | EN: TARGET:
# PT: - Regras de agrupamento abaixo devem ser chamadas pelo orquestrador em qt/cora_interface_qt.py durante o scan.
# EN: - The grouping rules below should be called by the orchestrator in qt/cora_interface_qt.py during scanning.
def sorted_image_paths_for_scan(folder: Path) -> list[Path]:
    """List supported image files in folder, sorted by filename."""
    paths = [
        path
        for path in folder.glob("*")
        if path.is_file() and (path.suffix.lower() in SUPPORTED_EXTS)
    ]
    return sorted(paths, key=lambda p: p.name.lower())


def collect_grouping_entries(
    folder: Path,
    groups: dict[str, dict[str, Path]],
    duplicates: list[tuple[str, str, Path, Path]],
) -> list[GroupingEntry]:
    """Build review entries from auto-grouping and duplicate hints."""
    all_paths = sorted_image_paths_for_scan(folder)
    selected_by_path: dict[Path, tuple[str, str]] = {}
    for group_key, time_map in groups.items():
        for time_tag, path in time_map.items():
            selected_by_path[path] = (group_key, time_tag)

    duplicate_by_path: dict[Path, tuple[str, str]] = {}
    for group_key, time_tag, _kept, ignored in duplicates:
        if ignored not in duplicate_by_path:
            duplicate_by_path[ignored] = (group_key, time_tag)

    entries: list[GroupingEntry] = []
    for path in all_paths:
        rel_path = str(path.relative_to(folder))
        selected_info = selected_by_path.get(path)
        if selected_info is not None:
            group_key, time_tag = selected_info
            entries.append(
                GroupingEntry(
                    path=path,
                    rel_path=rel_path,
                    group_key=group_key,
                    time_tag=time_tag,
                    selected=True,
                    source="auto",
                )
            )
            continue

        duplicate_info = duplicate_by_path.get(path)
        if duplicate_info is not None:
            group_key, time_tag = duplicate_info
            entries.append(
                GroupingEntry(
                    path=path,
                    rel_path=rel_path,
                    group_key=group_key,
                    time_tag=(None if time_tag == "sem_tempo" else time_tag),
                    selected=False,
                    source="duplicata",
                )
            )
            continue

        parsed = parse_group_image(path, base_folder=folder)
        if parsed is not None:
            group_key = f"{parsed.list_tag} {parsed.image_id}"
            time_tag = parsed.time_tag
        else:
            stem_clean = " ".join(strip_accents(path.stem).split()).strip()
            group_key = stem_clean if stem_clean else "grupo"
            time_tag = None

        entries.append(
            GroupingEntry(
                path=path,
                rel_path=rel_path,
                group_key=group_key,
                time_tag=time_tag,
                selected=False,
                source="sugestao",
            )
        )

    return entries


def build_groups_from_entries(
    entries: list[GroupingEntry],
) -> tuple[dict[str, dict[str, Path]], list[tuple[str, str, Path, Path]], list[str]]:
    """Build final group map after user review."""
    groups: dict[str, dict[str, Path]] = {}
    duplicates: list[tuple[str, str, Path, Path]] = []
    warnings: list[str] = []
    pending_untimed: dict[str, list[Path]] = {}

    ordered_entries = sorted(entries, key=lambda e: e.rel_path.lower())
    for entry in ordered_entries:
        if not entry.selected:
            continue
        group_key = " ".join(str(entry.group_key).split()).strip()
        if not group_key:
            warnings.append(f"Grupo vazio em: {entry.rel_path}")
            continue

        time_tag = str(entry.time_tag or "").strip().lower()
        if time_tag in TIME_ORDER:
            slot = groups.setdefault(group_key, {})
            if time_tag not in slot:
                slot[time_tag] = entry.path
            else:
                duplicates.append((group_key, time_tag, slot[time_tag], entry.path))
            continue

        if time_tag:
            warnings.append(f"Tempo invalido '{time_tag}' em: {entry.rel_path}")
        pending_untimed.setdefault(group_key, []).append(entry.path)

    for group_key, pending_paths in pending_untimed.items():
        slot = groups.setdefault(group_key, {})
        for pending_path in pending_paths:
            next_time = next((t for t in TIME_ORDER if t not in slot), None)
            if next_time is None:
                kept_path = slot.get(TIME_ORDER[-1]) or next(iter(slot.values()), pending_path)
                duplicates.append((group_key, "sem_tempo", kept_path, pending_path))
                continue
            slot[next_time] = pending_path

    return groups, duplicates, warnings


def all_items(
    group_order: list[str],
    group_files: dict[str, dict[str, Path]],
) -> list[tuple[str, str, Path]]:
    """Expand grouped files into a flat processing queue."""
    items: list[tuple[str, str, Path]] = []
    for group_key in group_order:
        for time_tag in TIME_ORDER:
            path = group_files.get(group_key, {}).get(time_tag)
            if path is not None:
                items.append((group_key, time_tag, path))
    return items


# PT: DESTINO: | EN: TARGET:
# PT: - Regras de metrica abaixo devem ser chamadas no resumo/resultado final, normalmente em metodos de metricas do orquestrador Qt.
# EN: - The metric rules below should be called in the final summary/result, usually from the Qt orchestrator's metric methods.
def area_auto_from_processed(
    processed_by_group: dict[str, dict[str, ProcessedTimepoint]],
    group_key: str,
    time_tag: str,
) -> float | None:
    """Read area_auto from processed item safely."""
    proc = processed_by_group.get(group_key, {}).get(time_tag)
    if proc is None:
        return None
    try:
        return float(proc.results.area_auto)
    except Exception:
        return None


def closure_vs_0h(
    processed_by_group: dict[str, dict[str, ProcessedTimepoint]],
    group_key: str,
    time_tag: str,
) -> tuple[float | None, float | None]:
    """Compute absolute/percentage closure against 0h."""
    area_base = area_auto_from_processed(processed_by_group, group_key, "0h")
    area_ref = area_auto_from_processed(processed_by_group, group_key, time_tag)
    if area_base is None or area_ref is None:
        return None, None
    fechamento_px = float(area_base - area_ref)
    if area_base <= 0.0:
        return fechamento_px, None
    fechamento_pct = 100.0 * fechamento_px / area_base
    return fechamento_px, fechamento_pct


__all__ = [
    "APP_ID",
    "GroupingEntry",
    "OUTPUT_DIR_NAME",
    "PREVIEW_COLOR_MODE_DEFAULT",
    "PREVIEW_COLOR_MODE_LABEL_TO_KEY",
    "PREVIEW_COLOR_MODE_PRESETS",
    "PREVIEW_CONTOUR_COLORS_DEFAULT",
    "PREVIEW_CONTOUR_COLOR_SHARED_DEFAULT",
    "PREVIEW_MODE_DEFAULT",
    "PREVIEW_MODE_LABEL_TO_KEY",
    "PREVIEW_MODE_PRESETS",
    "ProcessedTimepoint",
    "ROI_BRUSH_RADIUS_DEFAULT",
    "ROI_BRUSH_RADIUS_MAX",
    "ROI_BRUSH_RADIUS_MIN",
    "ROI_CONTOUR_THICKNESS_PX",
    "ROI_CURSOR_MAX_FPS",
    "ROI_DRAG_POINT_REFRESH_FPS",
    "ROI_EDITOR_MAX_FPS",
    "ROI_EDITOR_PROFILE_DEFAULT",
    "ROI_EDITOR_PROFILE_LABEL_TO_KEY",
    "ROI_EDITOR_PROFILE_PRESETS",
    "ROI_EDITOR_WORK_SCALE",
    "ROI_LOCAL_DRAG_MIN_SIGMA",
    "ROI_LOCAL_DRAG_SIGMA_FRAC",
    "ROI_OFFSET_STEP_PX",
    "ROI_PERIMETER_POINT_CAP",
    "ROI_PERIMETER_POINT_SIZE",
    "ROI_POLY_MAX_POINTS",
    "ROI_REGION_SIMPLIFY_STEP",
    "SUPPORTED_EXTS",
    "TIME_ORDER",
    "all_items",
    "area_auto_from_processed",
    "build_groups_from_entries",
    "closure_vs_0h",
    "collect_grouping_entries",
    "default_folder_from_base_path",
    "keep_largest_component",
    "mask_to_polygon",
    "offset_mask_sdf",
    "polygon_to_mask",
    "resize_mask",
    "sorted_image_paths_for_scan",
    "strip_accents",
]
