"""Core helpers for Qt frontend migration.

DIRECIONAMENTO:
- Importe daqui no orquestrador qt/cora_interface_qt.py.
- Evite importar regras de negocio diretamente em qt/ui/*.
"""

from .tk_portable_pack import (
    APP_ID,
    GroupingEntry,
    OUTPUT_DIR_NAME,
    ProcessedTimepoint,
    SUPPORTED_EXTS,
    TIME_ORDER,
    all_items,
    area_auto_from_processed,
    build_groups_from_entries,
    closure_vs_0h,
    collect_grouping_entries,
    default_folder_from_base_path,
    keep_largest_component,
    mask_to_polygon,
    offset_mask_sdf,
    polygon_to_mask,
    resize_mask,
    sorted_image_paths_for_scan,
    strip_accents,
)

__all__ = [
    "APP_ID",
    "GroupingEntry",
    "OUTPUT_DIR_NAME",
    "ProcessedTimepoint",
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
