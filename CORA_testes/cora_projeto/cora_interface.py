"""Interface grafica principal do fluxo de avaliacao de area por grupos e tempos."""

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import queue
import re
import cProfile
import threading
import time
import tkinter as tk
import unicodedata
from tkinter import colorchooser, filedialog, font as tkfont, messagebox, ttk
from typing import Callable

import cv2
from matplotlib.patches import Circle
from matplotlib.widgets import Button
import numpy as np
try:
    from PIL import Image, ImageOps, ImageTk
except Exception:
    Image = None
    ImageOps = None
    ImageTk = None

try:
    if __package__:
        from .matlab_style_cora import (
            BASE_IMAGE_PATH,
            CORAArtifacts,
            CORAConfig,
            AreaResults,
            clahe_matlab_like,
            contour_mask_with_thickness,
            ensure_gray,
            homomorphic_filter,
            mat2gray,
            morph_matlab_like,
            overlay_mask_alpha,
            overlay_perimeter,
            read_image,
            run_cora,
            to_uint8_01,
        )
    else:
        from cora_projeto.matlab_style_cora import (
            BASE_IMAGE_PATH,
            CORAArtifacts,
            CORAConfig,
            AreaResults,
            clahe_matlab_like,
            contour_mask_with_thickness,
            ensure_gray,
            homomorphic_filter,
            mat2gray,
            morph_matlab_like,
            overlay_mask_alpha,
            overlay_perimeter,
            read_image,
            run_cora,
            to_uint8_01,
        )
except ImportError:
    from matlab_style_cora import (
        BASE_IMAGE_PATH,
        CORAArtifacts,
        CORAConfig,
        AreaResults,
        clahe_matlab_like,
        contour_mask_with_thickness,
        ensure_gray,
        homomorphic_filter,
        mat2gray,
        morph_matlab_like,
        overlay_mask_alpha,
        overlay_perimeter,
        read_image,
        run_cora,
        to_uint8_01,
    )

try:
    if __package__:
        from .pages.main_page import build_main_layout
        from .pages.test_selection_page import build_test_selection_layout
        from .pages.roi_page import build_roi_editor_enhanced_image as build_roi_editor_enhanced_image_page
        from .services.grouping_service import discover_image_groups, parse_group_image
        from .services.performance_service import (
            ResourcePeaks,
            append_batch_test_report,
            append_test_report,
            read_batch_test_report,
            read_test_report,
        )
        from .services.processing_service import process_single_item, run_processing_worker
        from .services import export_service
    else:
        from cora_projeto.pages.main_page import build_main_layout
        from cora_projeto.pages.test_selection_page import build_test_selection_layout
        from cora_projeto.pages.roi_page import build_roi_editor_enhanced_image as build_roi_editor_enhanced_image_page
        from cora_projeto.services.grouping_service import discover_image_groups, parse_group_image
        from cora_projeto.services.performance_service import (
            ResourcePeaks,
            append_batch_test_report,
            append_test_report,
            read_batch_test_report,
            read_test_report,
        )
        from cora_projeto.services.processing_service import process_single_item, run_processing_worker
        from cora_projeto.services import export_service
except ImportError:
    from pages.main_page import build_main_layout
    from pages.test_selection_page import build_test_selection_layout
    from pages.roi_page import build_roi_editor_enhanced_image as build_roi_editor_enhanced_image_page
    from services.grouping_service import discover_image_groups, parse_group_image
    from services.performance_service import (
        ResourcePeaks,
        append_batch_test_report,
        append_test_report,
        read_batch_test_report,
        read_test_report,
    )
    from services.processing_service import process_single_item, run_processing_worker
    import services.export_service as export_service


SUPPORTED_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
TIME_ORDER = ("0h", "24h", "48h")
SINGLE_IMAGE_TEST_REPETITIONS = 30
BATCH_TEST_ITERATIONS = 10
BATCH_TEST_SIZES = (10, 24, 60)
OUTPUT_DIR_NAME = "_cora_resultados"
ROI_OFFSET_STEP_PX = 4.0
ROI_POLY_MAX_POINTS = 600
ROI_LOCAL_DRAG_SIGMA_FRAC = 0.035
ROI_LOCAL_DRAG_MIN_SIGMA = 6.0
ROI_BRUSH_RADIUS_DEFAULT = 12
ROI_BRUSH_RADIUS_MIN = 2
ROI_BRUSH_RADIUS_MAX = 120
ROI_EDITOR_MAX_FPS = 45.0
ROI_CURSOR_MAX_FPS = 45.0
ROI_CONTOUR_THICKNESS_PX = 2
# PT: Escala de trabalho do editor de ROI (0.50 = 50% da resolucao original). | EN: ROI editor work scale (0.50 = 50% of the original resolution).
# PT: Para voltar ao comportamento anterior (100%), ajuste para 1.0. | EN: Set it to 1.0 to restore the previous behavior (100%).
ROI_EDITOR_WORK_SCALE = 0.50
ROI_EDITOR_PROFILE_PRESETS: dict[str, dict[str, object]] = {
    # PT: Perfil atual padrao (mais leve), mantendo o comportamento solicitado de 50%. | EN: Current default profile (lighter) while preserving the requested 50% behavior.
    "rapido_50": {
        "label": "Rapido (50%)",
        "work_scale": 0.50,
        "editor_max_fps": 45.0,
        "cursor_max_fps": 45.0,
    },
}
ROI_EDITOR_PROFILE_DEFAULT = "rapido_50"
ROI_EDITOR_PROFILE_LABEL_TO_KEY = {
    str(cfg.get("label", "")).strip().lower(): key
    for key, cfg in ROI_EDITOR_PROFILE_PRESETS.items()
}
ROI_EDITOR_PROFILE_LABEL_TO_KEY.update({"fast (50%)": "rapido_50"})
PREVIEW_MODE_PRESETS: dict[str, str] = {
    "contour": "So contorno",
    "filled": "Area preenchida",
}
PREVIEW_MODE_DEFAULT = "filled"
PREVIEW_MODE_LABEL_TO_KEY = {label.strip().lower(): key for key, label in PREVIEW_MODE_PRESETS.items()}
PREVIEW_MODE_LABEL_TO_KEY.update(
    {
        "contour only": "contour",
        "filled area": "filled",
    }
)
PREVIEW_SOURCE_MODE_PRESETS: dict[str, str] = {
    "original": "Original",
    "enhanced": "Imagem realçada (Cinza)",
}
PREVIEW_SOURCE_MODE_DEFAULT = "original"
PREVIEW_SOURCE_MODE_LABEL_TO_KEY = {label.strip().lower(): key for key, label in PREVIEW_SOURCE_MODE_PRESETS.items()}
PREVIEW_SOURCE_MODE_LABEL_TO_KEY.update(
    {
        "enhanced": "enhanced",
        "mais definida": "enhanced",
        "contrast enchanced (grayscale)": "enhanced",
    }
)
PREVIEW_COLOR_MODE_PRESETS: dict[str, str] = {
    "shared": "Cor unica",
    "per_time": "Cores separadas por tipo",
}
PREVIEW_COLOR_MODE_DEFAULT = "shared"
PREVIEW_COLOR_MODE_LABEL_TO_KEY = {label.strip().lower(): key for key, label in PREVIEW_COLOR_MODE_PRESETS.items()}
PREVIEW_COLOR_MODE_LABEL_TO_KEY.update(
    {
        "single color": "shared",
        "separate colors by type": "per_time",
    }
)
PREVIEW_COMPARE_PROCESSING_COLOR_RGB: tuple[int, int, int] = (0, 190, 255)
PREVIEW_COMPARE_EDITED_COLOR_RGB: tuple[int, int, int] = (255, 0, 255)
PREVIEW_CONTOUR_COLOR_SHARED_DEFAULT: tuple[int, int, int] = (255, 0, 0)
PREVIEW_CONTOUR_COLORS_DEFAULT: dict[str, tuple[int, int, int]] = {
    "0h": PREVIEW_CONTOUR_COLOR_SHARED_DEFAULT,
    "24h": PREVIEW_CONTOUR_COLOR_SHARED_DEFAULT,
    "48h": PREVIEW_CONTOUR_COLOR_SHARED_DEFAULT,
}
APP_ID = "CORA"
UI_THEME_FAMILY_ORDER = ("rose", "standard", "blue", "green", "purple", "graphite")
UI_THEME_MODE_ORDER = ("light", "dark")
UI_THEME_FAMILY_NAMES: dict[str, dict[str, str]] = {
    "standard": {"pt": "Padrao", "en": "Standard"},
    "rose": {"pt": "Rosa", "en": "Rose"},
    "blue": {"pt": "Azul", "en": "Blue"},
    "green": {"pt": "Verde", "en": "Green"},
    "purple": {"pt": "Roxo", "en": "Purple"},
    "graphite": {"pt": "Grafite", "en": "Graphite"},
}
UI_THEME_PRESETS: dict[str, dict[str, object]] = {
    "standard_light": {
        "app_bg": "#F3F5F8",
        "panel_bg": "#FFFFFF",
        "fg": "#111827",
        "fg_disabled": "#8F97A4",
        "field_bg": "#FFFFFF",
        "field_bg_disabled": "#EEF2F7",
        "field_fg": "#111827",
        "button_bg": "#E8EEF7",
        "button_bg_active": "#DCE5F2",
        "button_bg_disabled": "#E8EEF7",
        "button_fg": "#111827",
        "button_fg_disabled": "#8F97A4",
        "header_bg": "#E5E7EB",
        "border": "#C5CEDA",
        "select_bg": "#2F6FED",
        "select_fg": "#FFFFFF",
        "tooltip_bg": "#FFF9D7",
        "tooltip_fg": "#111111",
        "text_bg": "#FFFFFF",
        "text_fg": "#111827",
        "menu_bg": "#FFFFFF",
        "menu_fg": "#111827",
        "progress_bg": "#2F6FED",
        "progress_trough": "#EEF2F7",
        "bg_blend_rgb": (255, 255, 255),
    },
    "standard_dark": {
        "app_bg": "#1B202B",
        "panel_bg": "#252B36",
        "fg": "#E7ECF3",
        "fg_disabled": "#7D8696",
        "field_bg": "#202733",
        "field_bg_disabled": "#2A313D",
        "field_fg": "#E7ECF3",
        "button_bg": "#313B4A",
        "button_bg_active": "#3D4A60",
        "button_bg_disabled": "#2A3240",
        "button_fg": "#E7ECF3",
        "button_fg_disabled": "#7D8696",
        "header_bg": "#2D3441",
        "border": "#445065",
        "select_bg": "#4A7DFF",
        "select_fg": "#FFFFFF",
        "tooltip_bg": "#2E3746",
        "tooltip_fg": "#E7ECF3",
        "text_bg": "#202733",
        "text_fg": "#E7ECF3",
        "menu_bg": "#2A313D",
        "menu_fg": "#E7ECF3",
        "progress_bg": "#4A7DFF",
        "progress_trough": "#2A313D",
        "bg_blend_rgb": (22, 26, 34),
    },
    "rose_light": {
        "app_bg": "#E8A5B0",
        "panel_bg": "#FFF1F4",
        "fg": "#3A1522",
        "fg_disabled": "#A87380",
        "field_bg": "#FFFAFB",
        "field_bg_disabled": "#F6D9DF",
        "field_fg": "#3A1522",
        "button_bg": "#8A214E",
        "button_bg_active": "#A73562",
        "button_bg_disabled": "#E7B6C0",
        "button_fg": "#FFFFFF",
        "button_fg_disabled": "#F8E3E8",
        "header_bg": "#F8D7DE",
        "border": "#D98A99",
        "select_bg": "#8A214E",
        "select_fg": "#FFFFFF",
        "tooltip_bg": "#FFF5F7",
        "tooltip_fg": "#3A1522",
        "text_bg": "#FFFAFB",
        "text_fg": "#3A1522",
        "menu_bg": "#FFF1F4",
        "menu_fg": "#3A1522",
        "progress_bg": "#8A214E",
        "progress_trough": "#F6D9DF",
        "bg_blend_rgb": (232, 165, 176),
    },
    "rose_dark": {
        "app_bg": "#5A2234",
        "panel_bg": "#703047",
        "fg": "#FFF0F3",
        "fg_disabled": "#D8A1AC",
        "field_bg": "#461829",
        "field_bg_disabled": "#5C2437",
        "field_fg": "#FFF0F3",
        "button_bg": "#E8A5B0",
        "button_bg_active": "#F0BBC4",
        "button_bg_disabled": "#87445A",
        "button_fg": "#3A1522",
        "button_fg_disabled": "#D8A1AC",
        "header_bg": "#8B4058",
        "border": "#C56C7C",
        "select_bg": "#E8A5B0",
        "select_fg": "#FFFFFF",
        "tooltip_bg": "#7E3850",
        "tooltip_fg": "#FFF0F3",
        "text_bg": "#461829",
        "text_fg": "#FFF0F3",
        "menu_bg": "#703047",
        "menu_fg": "#FFF0F3",
        "progress_bg": "#E8A5B0",
        "progress_trough": "#87445A",
        "bg_blend_rgb": (90, 34, 52),
    },
    "blue_light": {
        "app_bg": "#6EA8D7",
        "panel_bg": "#EAF4FB",
        "fg": "#102A3A",
        "fg_disabled": "#6F8795",
        "field_bg": "#F7FCFF",
        "field_bg_disabled": "#CDE4F4",
        "field_fg": "#102A3A",
        "button_bg": "#1F5D85",
        "button_bg_active": "#2E78A8",
        "button_bg_disabled": "#A7CDE4",
        "button_fg": "#FFFFFF",
        "button_fg_disabled": "#EAF4FB",
        "header_bg": "#C8E2F2",
        "border": "#6EA8D7",
        "select_bg": "#1F5D85",
        "select_fg": "#FFFFFF",
        "tooltip_bg": "#EFF9FF",
        "tooltip_fg": "#102A3A",
        "text_bg": "#F7FCFF",
        "text_fg": "#102A3A",
        "menu_bg": "#EAF4FB",
        "menu_fg": "#102A3A",
        "progress_bg": "#1F5D85",
        "progress_trough": "#CDE4F4",
        "bg_blend_rgb": (110, 168, 215),
    },
    "blue_dark": {
        "app_bg": "#123247",
        "panel_bg": "#1B425A",
        "fg": "#EAF6FF",
        "fg_disabled": "#8CB1C8",
        "field_bg": "#0F2738",
        "field_bg_disabled": "#18394E",
        "field_fg": "#EAF6FF",
        "button_bg": "#6EA8D7",
        "button_bg_active": "#8ABCE3",
        "button_bg_disabled": "#315C74",
        "button_fg": "#102A3A",
        "button_fg_disabled": "#8CB1C8",
        "header_bg": "#27536D",
        "border": "#4D8DB6",
        "select_bg": "#6EA8D7",
        "select_fg": "#102A3A",
        "tooltip_bg": "#234D66",
        "tooltip_fg": "#EAF6FF",
        "text_bg": "#0F2738",
        "text_fg": "#EAF6FF",
        "menu_bg": "#1B425A",
        "menu_fg": "#EAF6FF",
        "progress_bg": "#6EA8D7",
        "progress_trough": "#315C74",
        "bg_blend_rgb": (18, 50, 71),
    },
    "green_light": {
        "app_bg": "#8BBF9F",
        "panel_bg": "#EFF8F1",
        "fg": "#143524",
        "fg_disabled": "#6F8D7A",
        "field_bg": "#F8FFF9",
        "field_bg_disabled": "#D4E9DA",
        "field_fg": "#143524",
        "button_bg": "#2F6B4F",
        "button_bg_active": "#408563",
        "button_bg_disabled": "#B1D4BD",
        "button_fg": "#FFFFFF",
        "button_fg_disabled": "#EFF8F1",
        "header_bg": "#D0E7D7",
        "border": "#77AA8A",
        "select_bg": "#2F6B4F",
        "select_fg": "#FFFFFF",
        "tooltip_bg": "#F2FFF5",
        "tooltip_fg": "#143524",
        "text_bg": "#F8FFF9",
        "text_fg": "#143524",
        "menu_bg": "#EFF8F1",
        "menu_fg": "#143524",
        "progress_bg": "#2F6B4F",
        "progress_trough": "#D4E9DA",
        "bg_blend_rgb": (139, 191, 159),
    },
    "green_dark": {
        "app_bg": "#173B2B",
        "panel_bg": "#214D39",
        "fg": "#EAF7EE",
        "fg_disabled": "#94B8A1",
        "field_bg": "#112E21",
        "field_bg_disabled": "#1C4030",
        "field_fg": "#EAF7EE",
        "button_bg": "#8BBF9F",
        "button_bg_active": "#A4D0B3",
        "button_bg_disabled": "#38684C",
        "button_fg": "#143524",
        "button_fg_disabled": "#94B8A1",
        "header_bg": "#2E6148",
        "border": "#5A9470",
        "select_bg": "#8BBF9F",
        "select_fg": "#143524",
        "tooltip_bg": "#2A5A42",
        "tooltip_fg": "#EAF7EE",
        "text_bg": "#112E21",
        "text_fg": "#EAF7EE",
        "menu_bg": "#214D39",
        "menu_fg": "#EAF7EE",
        "progress_bg": "#8BBF9F",
        "progress_trough": "#38684C",
        "bg_blend_rgb": (23, 59, 43),
    },
    "purple_light": {
        "app_bg": "#B794D6",
        "panel_bg": "#F5ECFB",
        "fg": "#2F1B45",
        "fg_disabled": "#88709A",
        "field_bg": "#FCF8FF",
        "field_bg_disabled": "#E4D2F1",
        "field_fg": "#2F1B45",
        "button_bg": "#6A3D8F",
        "button_bg_active": "#8154A8",
        "button_bg_disabled": "#CBB4DD",
        "button_fg": "#FFFFFF",
        "button_fg_disabled": "#F5ECFB",
        "header_bg": "#E2D0F0",
        "border": "#9B75BD",
        "select_bg": "#6A3D8F",
        "select_fg": "#FFFFFF",
        "tooltip_bg": "#FBF4FF",
        "tooltip_fg": "#2F1B45",
        "text_bg": "#FCF8FF",
        "text_fg": "#2F1B45",
        "menu_bg": "#F5ECFB",
        "menu_fg": "#2F1B45",
        "progress_bg": "#6A3D8F",
        "progress_trough": "#E4D2F1",
        "bg_blend_rgb": (183, 148, 214),
    },
    "purple_dark": {
        "app_bg": "#311D45",
        "panel_bg": "#42265C",
        "fg": "#F5EAFE",
        "fg_disabled": "#B79ACA",
        "field_bg": "#261637",
        "field_bg_disabled": "#38204F",
        "field_fg": "#F5EAFE",
        "button_bg": "#B794D6",
        "button_bg_active": "#C9ACE3",
        "button_bg_disabled": "#5A3875",
        "button_fg": "#2F1B45",
        "button_fg_disabled": "#B79ACA",
        "header_bg": "#563473",
        "border": "#8763A6",
        "select_bg": "#B794D6",
        "select_fg": "#2F1B45",
        "tooltip_bg": "#51316B",
        "tooltip_fg": "#F5EAFE",
        "text_bg": "#261637",
        "text_fg": "#F5EAFE",
        "menu_bg": "#42265C",
        "menu_fg": "#F5EAFE",
        "progress_bg": "#B794D6",
        "progress_trough": "#5A3875",
        "bg_blend_rgb": (49, 29, 69),
    },
    "graphite_light": {
        "app_bg": "#AEB7C2",
        "panel_bg": "#F2F5F7",
        "fg": "#1D2730",
        "fg_disabled": "#727D88",
        "field_bg": "#FFFFFF",
        "field_bg_disabled": "#D9DEE4",
        "field_fg": "#1D2730",
        "button_bg": "#44546A",
        "button_bg_active": "#5A6C83",
        "button_bg_disabled": "#B9C1CA",
        "button_fg": "#FFFFFF",
        "button_fg_disabled": "#F2F5F7",
        "header_bg": "#DDE3E9",
        "border": "#8F9BA7",
        "select_bg": "#44546A",
        "select_fg": "#FFFFFF",
        "tooltip_bg": "#F8FAFC",
        "tooltip_fg": "#1D2730",
        "text_bg": "#FFFFFF",
        "text_fg": "#1D2730",
        "menu_bg": "#F2F5F7",
        "menu_fg": "#1D2730",
        "progress_bg": "#44546A",
        "progress_trough": "#D9DEE4",
        "bg_blend_rgb": (174, 183, 194),
    },
    "graphite_dark": {
        "app_bg": "#202733",
        "panel_bg": "#2C3440",
        "fg": "#E8EDF3",
        "fg_disabled": "#8894A3",
        "field_bg": "#171D26",
        "field_bg_disabled": "#242C38",
        "field_fg": "#E8EDF3",
        "button_bg": "#AEB7C2",
        "button_bg_active": "#C2CAD4",
        "button_bg_disabled": "#4E5967",
        "button_fg": "#1D2730",
        "button_fg_disabled": "#8894A3",
        "header_bg": "#3A4553",
        "border": "#5D6978",
        "select_bg": "#AEB7C2",
        "select_fg": "#1D2730",
        "tooltip_bg": "#3A4553",
        "tooltip_fg": "#E8EDF3",
        "text_bg": "#171D26",
        "text_fg": "#E8EDF3",
        "menu_bg": "#2C3440",
        "menu_fg": "#E8EDF3",
        "progress_bg": "#AEB7C2",
        "progress_trough": "#4E5967",
        "bg_blend_rgb": (32, 39, 51),
    },
}
UI_THEME_DEFAULT = "rose_light"
UI_LANGUAGE_DEFAULT = "pt"
UI_LANGUAGE_NAMES: dict[str, dict[str, str]] = {
    "pt": {"pt": "Portugues", "en": "Portuguese"},
    "en": {"pt": "Ingles", "en": "English"},
}
UI_TEXT: dict[str, dict[str, str]] = {
    "app.title": {
        "pt": "CORA - Revisao em Lote por Grupos",
        "en": "CORA - Batch Review by Groups",
    },
    "status.ready": {"pt": "Pronto", "en": "Ready"},
    "status.language_changed": {
        "pt": "Idioma alterado para {language}.",
        "en": "Language changed to {language}.",
    },
    "language.switch_to_en": {"pt": "English", "en": "English"},
    "language.switch_to_pt": {"pt": "Portugues", "en": "Portuguese"},
    "theme.switch_to_dark": {"pt": "Tema escuro", "en": "Dark theme"},
    "theme.switch_to_light": {"pt": "Tema claro", "en": "Light theme"},
    "theme.menu_label": {"pt": "Tema: {theme}", "en": "Theme: {theme}"},
    "theme.mode_dark": {"pt": "escuro", "en": "dark"},
    "theme.mode_light": {"pt": "claro", "en": "light"},
    "status.theme_changed": {
        "pt": "Tema visual alterado para {mode}.",
        "en": "Visual theme changed to {mode}.",
    },
    "button.guide": {"pt": "Guia", "en": "Guide"},
    "button.close": {"pt": "Fechar", "en": "Close"},
    "button.close_esc": {"pt": "Fechar (Esc)", "en": "Close (Esc)"},
    "button.browse": {"pt": "Procurar", "en": "Browse"},
    "button.browse_folder": {"pt": "Procurar pasta", "en": "Browse folder"},
    "button.choose_output": {"pt": "Escolher saida", "en": "Choose output"},
    "button.load_groups": {"pt": "Carregar grupos", "en": "Load groups"},
    "button.configure_contours": {"pt": "Configurar contornos", "en": "Configure contours"},
    "button.compare_areas": {"pt": "Comparar areas", "en": "Compare areas"},
    "button.hide_results": {"pt": "Ocultar resultados", "en": "Hide results"},
    "button.show_results": {"pt": "Mostrar resultados", "en": "Show results"},
    "button.prev_group": {"pt": "Grupo anterior", "en": "Previous group"},
    "button.next_group": {"pt": "Proximo grupo", "en": "Next group"},
    "button.edit_masks": {"pt": "Editar mascaras", "en": "Edit masks"},
    "button.restore_auto": {"pt": "Restaurar auto", "en": "Restore auto"},
    "button.save_result": {"pt": "Salvar resultado", "en": "Save result"},
    "button.cancel_processing": {"pt": "Cancelar processamento", "en": "Cancel processing"},
    "button.fullscreen": {"pt": "Tela cheia", "en": "Full screen"},
    "button.view_enhanced": {"pt": "Imagem realçada (Cinza)", "en": "Contrast enchanced (Grayscale)"},
    "button.view_original": {"pt": "Ver original", "en": "View original"},
    "button.apply_group_time": {"pt": "Aplicar grupo/tempo", "en": "Apply group/time"},
    "button.mark_all": {"pt": "Marcar todos", "en": "Select all"},
    "button.unmark_all": {"pt": "Desmarcar todos", "en": "Clear all"},
    "button.auto_fill_times": {"pt": "Auto preencher tempos vazios", "en": "Auto fill missing times"},
    "button.cancel": {"pt": "Cancelar", "en": "Cancel"},
    "button.ok": {"pt": "OK", "en": "OK"},
    "button.start_processing": {"pt": "Iniciar processamento", "en": "Start processing"},
    "button.test_modes": {"pt": "Modos de teste", "en": "Test modes"},
    "button.choose_single_image": {
        "pt": "Selecionar imagem e iniciar 30 repeticoes",
        "en": "Select image and start 30 runs",
    },
    "button.open_batch": {"pt": "Abrir fluxo em lote", "en": "Open batch workflow"},
    "button.open_robotized_test": {
        "pt": "Robot test",
        "en": "Robot test",
    },
    "button.start_batch_test": {
        "pt": "Testar {count} imagens",
        "en": "Test {count} images",
    },
    "test.title": {
        "pt": "Escolha como deseja testar a aplicacao",
        "en": "Choose how you want to test the application",
    },
    "test.single_title": {"pt": "Teste com uma unica imagem", "en": "Single-image test"},
    "test.batch_title": {"pt": "Fluxo normal em lote", "en": "Normal batch workflow"},
    "test.single_description": {
        "pt": "Processa a mesma imagem 30 vezes no fluxo real do app e abre o ultimo resultado para revisao.\n\nMedicoes por repeticao:\n- tempo de carregamento\n- tempo de segmentacao\n- pico de memoria RAM\n- pico de uso da CPU\n\nAs 30 repeticoes sao acrescentadas aos relatorios CSV e Excel.",
        "en": "Processes the same image 30 times through the app's real workflow and opens the last result for review.\n\nMeasurements per run:\n- loading time\n- segmentation time\n- peak RAM\n- peak CPU usage\n\nAll 30 runs are appended to the CSV and Excel reports.",
    },
    "test.batch_description": {
        "pt": "Abre a configuracao existente para carregar pastas, revisar grupos e processar imagens nos tempos 0h, 24h e 48h.\n\nUse esta opcao para continuar trabalhando com o fluxo completo do app.",
        "en": "Opens the existing settings to load folders, review groups and process images at 0h, 24h and 48h.\n\nUse this option to continue with the app's complete workflow.",
    },
    "test.batch_test_title": {"pt": "Teste de batching", "en": "Batching test"},
    "test.batch_test_description": {
        "pt": "Escolha o tamanho do lote. O mesmo conjunto sera processado em 10 iteracoes, com uma linha por iteracao nos relatorios CSV e Excel.",
        "en": "Choose the batch size. The same set is processed for 10 iterations, with one row per iteration in the CSV and Excel reports.",
    },
    "test.robotized_title": {"pt": "Teste robotizado", "en": "Automated test"},
    "test.robotized_description": {
        "pt": "Direciona para a tela principal do aplicativo, pronta para o uso normal.",
        "en": "Opens the application's main screen, ready for normal use.",
    },
    "test.select_batch_folder": {
        "pt": "Selecione a pasta para o teste com {count} imagens",
        "en": "Select the folder for the {count}-image test",
    },
    "test.batch_not_enough": {
        "pt": "Foram encontradas apenas {found} imagens validas e agrupadas. O teste selecionado exige {required} imagens.",
        "en": "Only {found} valid grouped images were found. The selected test requires {required} images.",
    },
    "test.batch_progress_title": {
        "pt": "Teste de batching: {count} imagens x 10 iteracoes",
        "en": "Batching test: {count} images x 10 iterations",
    },
    "test.extensible_hint": {
        "pt": "Novos tipos de teste podem ser adicionados nesta tela sem alterar o fluxo principal.",
        "en": "New test types can be added here without changing the main workflow.",
    },
    "test.select_image_title": {
        "pt": "Selecione uma imagem para o teste",
        "en": "Select an image for the test",
    },
    "test.report_title": {"pt": "Resultado do teste de imagem unica", "en": "Single-image test result"},
    "test.progress_title": {"pt": "Teste de imagem unica", "en": "Single-image test"},
    "test.menu_choose": {"pt": "Escolher modo de teste", "en": "Choose test mode"},
    "status.test_modes_opened": {"pt": "Selecao de modo de teste aberta.", "en": "Test mode selection opened."},
    "config.initial_settings": {"pt": "Configuracoes iniciais", "en": "Initial settings"},
    "label.image_folder": {"pt": "Pasta de imagens:", "en": "Image folder:"},
    "label.output_folder": {"pt": "Pasta de saida:", "en": "Output folder:"},
    "label.output_folder_optional": {"pt": "Pasta de saida (opcional):", "en": "Output folder (optional):"},
    "label.group_in_review": {"pt": "Grupo em revisao:", "en": "Group under review:"},
    "label.include_save": {"pt": "Incluir no salvamento:", "en": "Include in save:"},
    "label.no_area": {"pt": "Sem area:", "en": "No area:"},
    "label.group_results": {"pt": "Resultados do grupo", "en": "Group results"},
    "label.group_images": {"pt": "Imagens do grupo", "en": "Group images"},
    "label.processing_progress_inline": {
        "pt": "Andamento do processamento (popup fechado)",
        "en": "Processing progress (popup closed)",
    },
    "label.processing_progress": {"pt": "Progresso do processamento", "en": "Processing progress"},
    "label.current_image": {"pt": "Imagem atual:", "en": "Current image:"},
    "label.remaining": {"pt": "Faltando:", "en": "Remaining:"},
    "label.eta": {"pt": "ETA:", "en": "ETA:"},
    "label.group": {"pt": "Grupo:", "en": "Group:"},
    "label.time": {"pt": "Tempo:", "en": "Time:"},
    "label.selected_image": {"pt": "Imagem selecionada", "en": "Selected image"},
    "menu.configurations_button": {"pt": "Configuracoes v", "en": "Settings v"},
    "menu.paths": {"pt": "Pastas de entrada/saida", "en": "Input/output folders"},
    "menu.group_navigation": {"pt": "Navegacao de grupos", "en": "Group navigation"},
    "menu.contour_colors": {"pt": "Cores dos contornos", "en": "Contour colors"},
    "menu.open_full_settings": {"pt": "Abrir configuracoes completas", "en": "Open full settings"},
    "tab.results": {"pt": "Resultados", "en": "Results"},
    "title.quick_paths": {"pt": "Configuracoes rapidas - Pastas", "en": "Quick settings - Folders"},
    "title.quick_groups": {"pt": "Configuracoes rapidas - Grupos", "en": "Quick settings - Groups"},
    "title.contour_settings": {
        "pt": "Configuracoes dos contornos automaticos",
        "en": "Automatic contour settings",
    },
    "title.theme_settings": {"pt": "Configuracoes de tema", "en": "Theme settings"},
    "title.guide": {"pt": "Guia", "en": "Guide"},
    "title.progress_full": {"pt": "Processamento em lote", "en": "Batch processing"},
    "title.progress_reprocess": {"pt": "Reprocessando imagens", "en": "Reprocessing images"},
    "label.initial_preview": {"pt": "Visualizacao inicial:", "en": "Initial preview:"},
    "label.background_image": {"pt": "Imagem de fundo:", "en": "Background image:"},
    "label.color_mode": {"pt": "Modo de cores:", "en": "Color mode:"},
    "label.theme_pattern": {"pt": "Padrao de cores:", "en": "Color pattern:"},
    "label.theme_mode": {"pt": "Versao:", "en": "Version:"},
    "label.single_color": {"pt": "Cor unica:", "en": "Single color:"},
    "label.colors_by_type": {"pt": "Cores por tipo:", "en": "Colors by type:"},
    "color.all": {"pt": "Todas", "en": "All"},
    "color.choose_all_title": {
        "pt": "Escolha a cor do contorno (todas as imagens)",
        "en": "Choose contour color (all images)",
    },
    "color.choose_time_title": {
        "pt": "Escolha a cor do contorno ({time})",
        "en": "Choose contour color ({time})",
    },
    "status.shared_color_changed": {
        "pt": "Cor unica dos contornos alterada para {color}.",
        "en": "Single contour color changed to {color}.",
    },
    "status.time_color_changed": {
        "pt": "Cor do contorno {time} alterada para {color}.",
        "en": "{time} contour color changed to {color}.",
    },
    "status.config_page_opened": {"pt": "Tela de configuracoes aberta.", "en": "Settings page opened."},
    "status.viewer_page_opened": {"pt": "Visualizacao das imagens aberta.", "en": "Image viewer opened."},
    "status.results_hidden": {"pt": "Painel de resultados oculto.", "en": "Results panel hidden."},
    "status.results_shown": {"pt": "Painel de resultados exibido.", "en": "Results panel shown."},
    "preview.mode.contour": {"pt": "So contorno", "en": "Contour only"},
    "preview.mode.filled": {"pt": "Area preenchida", "en": "Filled area"},
    "preview.source.original": {"pt": "Original", "en": "Original"},
    "preview.source.enhanced": {"pt": "Imagem realçada (Cinza)", "en": "Contrast enchanced (Grayscale)"},
    "preview.color.shared": {"pt": "Cor unica", "en": "Single color"},
    "preview.color.per_time": {"pt": "Cores separadas por tipo", "en": "Separate colors by type"},
    "dialog.warning": {"pt": "Aviso", "en": "Warning"},
    "dialog.error": {"pt": "Erro", "en": "Error"},
    "dialog.saved": {"pt": "Salvo", "en": "Saved"},
    "dialog.done": {"pt": "Concluido", "en": "Done"},
    "dialog.preview": {"pt": "Preview", "en": "Preview"},
    "dialog.review": {"pt": "Revisao", "en": "Review"},
    "common.more_items": {"pt": "... +{count} item(ns)", "en": "... +{count} item(s)"},
    "common.more_warnings": {"pt": "... +{count} aviso(s)", "en": "... +{count} warning(s)"},
    "common.more_errors": {"pt": "... +{count} erro(s)", "en": "... +{count} error(s)"},
    "review.select_preview": {"pt": "Selecione uma imagem para visualizar.", "en": "Select an image to preview."},
    "review.select_row_preview": {
        "pt": "Selecione uma linha para ver a imagem aqui.",
        "en": "Select a row to view the image here.",
    },
    "review.preview_unavailable": {
        "pt": "Preview indisponivel.\nInstale Pillow para mostrar a imagem.",
        "en": "Preview unavailable.\nInstall Pillow to show the image.",
    },
    "review.loading": {"pt": "Carregando...", "en": "Loading..."},
    "review.file": {"pt": "Arquivo", "en": "File"},
    "review.group": {"pt": "Grupo", "en": "Group"},
    "review.time": {"pt": "Tempo", "en": "Time"},
    "review.source": {"pt": "Origem", "en": "Source"},
    "review.view": {"pt": "Visualizacao", "en": "View"},
    "review.source.auto": {"pt": "auto", "en": "auto"},
    "review.source.duplicate": {"pt": "duplicata", "en": "duplicate"},
    "review.source.suggestion": {"pt": "sugestao", "en": "suggestion"},
    "review.heading.include": {"pt": "Incluir", "en": "Include"},
    "review.heading.file": {"pt": "Arquivo", "en": "File"},
    "review.heading.source": {"pt": "Origem", "en": "Source"},
    "review.summary": {
        "pt": "Selecionadas: {selected}/{total} | Grupos: {groups} | Sem tempo definido: {missing}",
        "en": "Selected: {selected}/{total} | Groups: {groups} | Missing time: {missing}",
    },
    "review.include_tooltip": {
        "pt": "Coluna 'Incluir': clique para marcar ou desmarcar as imagens que entram no processamento.",
        "en": "'Include' column: click to check or uncheck images included in processing.",
    },
    "review.fullscreen_tooltip": {
        "pt": "Tela cheia: abre a imagem selecionada em visualizacao ampliada.",
        "en": "Full screen: opens the selected image in an enlarged view.",
    },
    "review.toggle_preview_tooltip": {
        "pt": "Imagem realçada (Cinza)/Ver original: alterna a imagem de preview entre versoes.",
        "en": "Contrast enchanced (Grayscale)/View original: switches the preview image between versions.",
    },
    "review.apply_tooltip": {
        "pt": "Aplicar grupo/tempo: aplica os campos editados nas linhas selecionadas.",
        "en": "Apply group/time: applies the edited fields to selected rows.",
    },
    "review.mark_all_tooltip": {"pt": "Marcar todos: inclui todas as linhas no processamento.", "en": "Select all: includes every row in processing."},
    "review.unmark_all_tooltip": {
        "pt": "Desmarcar todos: remove todas as linhas do processamento.",
        "en": "Clear all: removes every row from processing.",
    },
    "review.auto_fill_tooltip": {
        "pt": "Auto preencher tempos vazios: completa tempos faltantes (0h/24h/48h) dentro de cada grupo.",
        "en": "Auto fill missing times: completes missing times (0h/24h/48h) inside each group.",
    },
    "review.cancel_tooltip": {"pt": "Cancelar: fecha a revisao sem aplicar este agrupamento.", "en": "Cancel: closes the review without applying this grouping."},
    "review.start_tooltip": {
        "pt": "Iniciar processamento: confirma o agrupamento atual e inicia o processamento em seguida.",
        "en": "Start processing: confirms the current grouping and starts processing.",
    },
    "review.select_image_full": {"pt": "Selecione uma imagem para ampliar.", "en": "Select an image to enlarge."},
    "review.pillow_unavailable": {
        "pt": "Pillow nao disponivel para visualizar imagem.",
        "en": "Pillow is not available to display the image.",
    },
    "review.open_image_failed": {
        "pt": "Nao foi possivel abrir a imagem:\n{name}",
        "en": "Could not open the image:\n{name}",
    },
    "review.no_selection_apply": {
        "pt": "Selecione ao menos uma linha para aplicar.",
        "en": "Select at least one row to apply.",
    },
    "review.empty_group_warning": {
        "pt": "Existem imagens selecionadas com grupo vazio.\nDefina o grupo ou desmarque essas linhas antes de iniciar o processamento.\n\n{details}",
        "en": "Some selected images have an empty group.\nSet the group or uncheck those rows before starting processing.\n\n{details}",
    },
    "review.no_selected_confirm": {
        "pt": "Nenhuma imagem esta selecionada. Deseja continuar assim?",
        "en": "No image is selected. Do you want to continue anyway?",
    },
    "review.cancelled": {
        "pt": "A revisao dos grupos foi cancelada.\nClique em 'Carregar grupos' para tentar novamente.",
        "en": "Group review was cancelled.\nClick 'Load groups' to try again.",
    },
    "review.confirmed_loading": {"pt": "Agrupamento confirmado. Carregando grupos...", "en": "Grouping confirmed. Loading groups..."},
    "scan.preparing": {
        "pt": "Escaneando pasta e preparando agrupamento...",
        "en": "Scanning folder and preparing grouping...",
    },
    "scan.folder_required": {"pt": "Informe a pasta de imagens.", "en": "Enter the image folder."},
    "scan.invalid_folder": {"pt": "Pasta invalida: {folder}", "en": "Invalid folder: {folder}"},
    "scan.scanning_folder": {"pt": "Escaneando pasta...", "en": "Scanning folder..."},
    "scan.scanning_files": {"pt": "Escaneando arquivos...", "en": "Scanning files..."},
    "scan.grouping_progress": {
        "pt": "Separando imagens em grupos: {done}/{total}",
        "en": "Separating images into groups: {done}/{total}",
    },
    "scan.review_prompt": {
        "pt": "Revise o agrupamento e clique em 'Iniciar processamento'.",
        "en": "Review the grouping and click 'Start processing'.",
    },
    "scan.cancelled": {"pt": "Carregamento de grupos cancelado.", "en": "Group loading cancelled."},
    "scan.no_group_found_status": {"pt": "Nenhum grupo encontrado", "en": "No group found"},
    "scan.no_group_found_message": {
        "pt": "Nenhum grupo valido foi encontrado.\nRevise os nomes dos arquivos/pastas e tente novamente.",
        "en": "No valid group was found.\nReview file/folder names and try again.",
    },
    "scan.no_group_found_warning": {
        "pt": "Nenhuma imagem agrupavel foi encontrada.\nUse nomes/pastas contendo i|ni (ou equivalentes) e 0h/24h/48h.",
        "en": "No groupable image was found.\nUse names/folders containing i|ni (or equivalents) and 0h/24h/48h.",
    },
    "scan.groups_found": {
        "pt": "{count} grupo(s) encontrado(s). Agrupamento confirmado; preparando processamento.",
        "en": "{count} group(s) found. Grouping confirmed; preparing processing.",
    },
    "scan.duplicates_title": {"pt": "Duplicatas encontradas", "en": "Duplicates found"},
    "scan.duplicates_message": {
        "pt": "Foram encontradas imagens duplicadas para o mesmo grupo/tempo.\nO processamento vai usar o primeiro arquivo encontrado em ordem alfabetica.\n\n{details}",
        "en": "Duplicate images were found for the same group/time.\nProcessing will use the first file found in alphabetical order.\n\n{details}",
    },
    "scan.duplicate_line": {
        "pt": "{group} {time}: mantendo '{kept}', ignorando '{ignored}'",
        "en": "{group} {time}: keeping '{kept}', ignoring '{ignored}'",
    },
    "scan.more_duplicates": {"pt": "... +{count} duplicata(s)", "en": "... +{count} duplicate(s)"},
    "scan.empty_group_adjustment": {"pt": "Grupo vazio em: {path}", "en": "Empty group in: {path}"},
    "scan.invalid_time_adjustment": {"pt": "Tempo invalido '{time}' em: {path}", "en": "Invalid time '{time}' in: {path}"},
    "scan.grouping_summary_log": {"pt": "Resumo do agrupamento confirmado:", "en": "Confirmed grouping summary:"},
    "scan.duplicates_log_warning": {
        "pt": "Aviso: {count} conflito(s) de grupo/tempo ficaram de fora.",
        "en": "Warning: {count} group/time conflict(s) were left out.",
    },
    "scan.grouping_warnings_title": {"pt": "Avisos no agrupamento", "en": "Grouping warnings"},
    "scan.grouping_warnings_message": {
        "pt": "Alguns ajustes automaticos foram aplicados:\n\n{details}",
        "en": "Some automatic adjustments were applied:\n\n{details}",
    },
    "scan.loaded_starting": {
        "pt": "{count} grupo(s) carregado(s). Iniciando processamento...",
        "en": "{count} group(s) loaded. Starting processing...",
    },
    "scan.loaded_not_started": {
        "pt": "{count} grupo(s) carregado(s). Processamento nao iniciado automaticamente.",
        "en": "{count} group(s) loaded. Processing was not started automatically.",
    },
    "save.no_results": {"pt": "Nao ha resultados para salvar.", "en": "There are no results to save."},
    "save.failed": {"pt": "Falha ao salvar resultados atuais:\n{error}", "en": "Failed to save current results:\n{error}"},
    "save.write_failed": {
        "pt": "Falha ao salvar arquivo: {name}\nCaminho: {path}",
        "en": "Failed to save file: {name}\nPath: {path}",
    },
    "save.saved_status": {"pt": "Resultados salvos em: {path}", "en": "Results saved to: {path}"},
    "save.saved_message": {"pt": "Resultados salvos em:\n{path}", "en": "Results saved to:\n{path}"},
    "processing.load_groups_first": {"pt": "Carregue os grupos antes de processar.", "en": "Load groups before processing."},
    "processing.no_valid_images": {"pt": "Nao ha imagens validas para processar.", "en": "There are no valid images to process."},
    "processing.no_images": {"pt": "Nao ha imagens para processar.", "en": "There are no images to process."},
    "processing.full_initial": {
        "pt": "Processando todos os grupos (0/{total})...",
        "en": "Processing all groups (0/{total})...",
    },
    "processing.reprocess_initial": {
        "pt": "Reprocessando nao selecionadas (0/{total})...",
        "en": "Reprocessing unchecked images (0/{total})...",
    },
    "processing.waiting_first": {"pt": "Aguardando primeira imagem...", "en": "Waiting for first image..."},
    "processing.prefix_full": {"pt": "Lote", "en": "Batch"},
    "processing.prefix_reprocess": {"pt": "Refazendo", "en": "Reprocessing"},
    "processing.stage_status": {
        "pt": "{prefix}: {group} {time} | {stage} ({index}/{total})",
        "en": "{prefix}: {group} {time} | {stage} ({index}/{total})",
    },
    "processing.stage.preparing_image": {"pt": "Preparando imagem", "en": "Preparing image"},
    "processing.stage.loading_image": {"pt": "Carregando imagem", "en": "Loading image"},
    "processing.stage.configuring_params": {"pt": "Configurando parametros", "en": "Configuring parameters"},
    "processing.stage.avg_brightness": {"pt": "Analisando brilho medio", "en": "Analyzing mean brightness"},
    "processing.stage.prepare_roi": {"pt": "Preparando ROI", "en": "Preparing ROI"},
    "processing.stage.preprocess": {"pt": "Pre-processando imagem", "en": "Preprocessing image"},
    "processing.stage.estimate_gabor": {"pt": "Estimando iteracoes de Gabor", "en": "Estimating Gabor iterations"},
    "processing.stage.run_gabor": {"pt": "Executando filtros de Gabor", "en": "Running Gabor filters"},
    "processing.stage.texture_map": {"pt": "Montando mapa de textura", "en": "Building texture map"},
    "processing.stage.estimate_kmeans": {"pt": "Estimando iteracoes do k-means", "en": "Estimating k-means iterations"},
    "processing.stage.kmeans_regions": {"pt": "Separando regioes (k-means)", "en": "Separating regions (k-means)"},
    "processing.stage.postprocess_mask": {"pt": "Pos-processando mascara", "en": "Postprocessing mask"},
    "processing.stage.refine_adaptive": {"pt": "Refinando mascara por modo adaptativo", "en": "Refining mask by adaptive mode"},
    "processing.stage.metrics": {"pt": "Calculando metricas", "en": "Calculating metrics"},
    "processing.stage.completed": {"pt": "Concluido", "en": "Completed"},
    "processing.stage.finished": {"pt": "Finalizado", "en": "Finished"},
    "processing.stage.special_24h_roi": {
        "pt": "Preparando ROI especial de 24h ({origin})",
        "en": "Preparing special 24h ROI ({origin})",
    },
    "processing.stage.adaptive_mode": {
        "pt": "Modo adaptativo: {mode} (media={mean})",
        "en": "Adaptive mode: {mode} (mean={mean})",
    },
    "processing.error_read_image": {
        "pt": "Falha ao ler imagem: {name}\n{error}",
        "en": "Failed to read image: {name}\n{error}",
    },
    "processing.error_processing_image": {
        "pt": "Falha no processamento: {name}\n{error}",
        "en": "Processing failed: {name}\n{error}",
    },
    "processing.error_unexpected": {
        "pt": "Falha inesperada no processamento: {error}",
        "en": "Unexpected processing failure: {error}",
    },
    "processing.read.access_failed": {
        "pt": "Nao consegui acessar a imagem: {path}",
        "en": "Could not access the image: {path}",
    },
    "processing.read.empty_or_corrupt": {
        "pt": "Arquivo de imagem vazio ou corrompido: {path}",
        "en": "Empty or corrupted image file: {path}",
    },
    "processing.read.decode_failed": {
        "pt": "Nao consegui decodificar a imagem: {path}",
        "en": "Could not decode the image: {path}",
    },
    "processing.read.unsupported_format": {
        "pt": "Formato de imagem nao suportado: {path}",
        "en": "Unsupported image format: {path}",
    },
    "processing.temp_save_failed": {
        "pt": "Nao foi possivel salvar imagem temporaria adaptada: {path}",
        "en": "Could not save adapted temporary image: {path}",
    },
    "processing.general": {"pt": "geral", "en": "general"},
    "processing.item_done": {
        "pt": "{prefix}: {group} {time} concluida ({index}/{total})",
        "en": "{prefix}: {group} {time} completed ({index}/{total})",
    },
    "processing.running_title": {"pt": "Processamento em andamento", "en": "Processing in progress"},
    "processing.close_confirm": {
        "pt": "Ha um processamento em andamento. Deseja cancelar e fechar o aplicativo?",
        "en": "Processing is in progress. Do you want to cancel and close the application?",
    },
    "processing.cancel_requested": {
        "pt": "Cancelamento solicitado. Aguardando terminar a imagem atual...",
        "en": "Cancellation requested. Waiting for the current image to finish...",
    },
    "processing.cancelled_status": {
        "pt": "Processamento cancelado pelo usuario. Concluidas {done}/{total} imagem(ns).",
        "en": "Processing cancelled by the user. Completed {done}/{total} image(s).",
    },
    "processing.cancelled_message": {
        "pt": "Processamento cancelado.\nConcluidas: {done}/{total} imagem(ns).",
        "en": "Processing cancelled.\nCompleted: {done}/{total} image(s).",
    },
    "processing.errors_registered": {"pt": "Erros registrados:\n{details}", "en": "Registered errors:\n{details}"},
    "processing.cancelled_with_warnings": {
        "pt": "Processamento cancelado com avisos",
        "en": "Processing cancelled with warnings",
    },
    "processing.cancelled_title": {"pt": "Processamento cancelado", "en": "Processing cancelled"},
    "processing.completed_with_errors": {
        "pt": "Processamento em lote concluido com {count} erro(s).",
        "en": "Batch processing completed with {count} error(s).",
    },
    "processing.completed_with_warnings": {
        "pt": "Processamento concluido com avisos",
        "en": "Processing completed with warnings",
    },
    "processing.completed_review": {
        "pt": "Processamento em lote concluido. Inicie a revisao dos grupos.",
        "en": "Batch processing completed. Start reviewing the groups.",
    },
    "processing.reprocess_cancelled": {
        "pt": "Reprocessamento cancelado. Concluidas {done}/{total} imagem(ns).",
        "en": "Reprocessing cancelled. Completed {done}/{total} image(s).",
    },
    "processing.reprocess_cancelled_with_warnings": {
        "pt": "Reprocessamento cancelado com avisos",
        "en": "Reprocessing cancelled with warnings",
    },
    "processing.reprocess_cancelled_title": {"pt": "Reprocessamento cancelado", "en": "Reprocessing cancelled"},
    "processing.reprocess_completed_warnings_status": {
        "pt": "Reprocessamento concluido com avisos.",
        "en": "Reprocessing completed with warnings.",
    },
    "processing.reprocess_completed_warnings": {
        "pt": "Reprocessamento concluido com avisos",
        "en": "Reprocessing completed with warnings",
    },
    "processing.reprocess_save_failed": {
        "pt": "Falha ao salvar resultados: {error}",
        "en": "Failed to save results: {error}",
    },
    "processing.reprocess_saved_status": {
        "pt": "Reprocessamento concluido. Resultados salvos em: {path}",
        "en": "Reprocessing completed. Results saved to: {path}",
    },
    "processing.reprocess_saved_message": {
        "pt": "Reprocessamento concluido e salvo em:\n{path}",
        "en": "Reprocessing completed and saved to:\n{path}",
    },
    "processing.reprocess_completed": {"pt": "Reprocessamento concluido.", "en": "Reprocessing completed."},
    "status.preview_mode_changed": {"pt": "Visualizacao inicial: {label}.", "en": "Initial preview: {label}."},
    "status.preview_source_changed": {"pt": "Imagem de fundo: {label}.", "en": "Background image: {label}."},
    "status.preview_color_mode_changed": {"pt": "Modo de cor dos contornos: {label}.", "en": "Contour color mode: {label}."},
    "status.compare_area_overlay_changed": {
        "pt": "Comparacao de areas {state}.",
        "en": "Area comparison {state}.",
    },
    "state.enabled": {"pt": "ativada", "en": "enabled"},
    "state.disabled": {"pt": "desativada", "en": "disabled"},
    "status.review_progress": {
        "pt": "Revisao: grupo {index}/{total} | grupos vistos: {reviewed}/{total}",
        "en": "Review: group {index}/{total} | viewed groups: {reviewed}/{total}",
    },
    "folder.changed_reload": {
        "pt": "Pasta alterada. Clique em 'Carregar grupos' para carregar somente a pasta atual.",
        "en": "Folder changed. Click 'Load groups' to load only the current folder.",
    },
    "folder.select_images_title": {"pt": "Selecione a pasta com as imagens", "en": "Select the folder with images"},
    "folder.select_output_title": {"pt": "Selecione a pasta para salvar os resultados", "en": "Select the folder to save results"},
    "guide.title": {"pt": "----------- GUIA -----------", "en": "----------- GUIDE -----------"},
    "guide.hover": {
        "pt": "Passe o mouse sobre os botoes para ver a explicacao rapida.",
        "en": "Hover over buttons to see a quick explanation.",
    },
    "guide.flow": {"pt": "Fluxo sugerido:", "en": "Suggested flow:"},
    "guide.buttons": {"pt": "Botoes:", "en": "Buttons:"},
    "guide.group_review": {"pt": "Revisao de agrupamento:", "en": "Grouping review:"},
    "tooltip.guide": {"pt": "Guia: abre o guia completo com instrucoes e atalhos.", "en": "Guide: opens full instructions and shortcuts."},
    "tooltip.theme": {"pt": "Tema: escolhe um dos padroes de cor em versao clara ou escura.", "en": "Theme: chooses one color pattern in light or dark mode."},
    "tooltip.language": {
        "pt": "Idioma: alterna os textos da interface entre portugues e ingles.",
        "en": "Language: switches interface text between Portuguese and English.",
    },
    "help.browse_folder": {"pt": "Escolhe a pasta com as imagens de entrada.", "en": "Chooses the folder with input images."},
    "help.browse_output": {"pt": "Define onde salvar os resultados.", "en": "Sets where results will be saved."},
    "help.load_groups": {
        "pt": "Faz o agrupamento automatico e abre a revisao em tabela para ajustes antes de carregar.",
        "en": "Groups images automatically and opens the review table for adjustments before loading.",
    },
    "help.settings": {
        "pt": "Abre menu rapido (pastas, navegacao de grupos, cores dos contornos e configuracoes completas).",
        "en": "Opens the quick menu for folders, group navigation, contour colors, and full settings.",
    },
    "help.theme": {"pt": "Escolhe entre 6 padroes de cor, cada um com versao clara e escura.", "en": "Chooses among 6 color patterns, each with light and dark versions."},
    "help.language": {
        "pt": "Alterna os textos do aplicativo entre portugues e ingles.",
        "en": "Switches application text between Portuguese and English.",
    },
    "help.guide": {"pt": "Abre o guia completo com instrucoes e atalhos.", "en": "Opens the full guide with instructions and shortcuts."},
    "help.contours": {
        "pt": "Abre os ajustes de visualizacao e das cores dos contornos.",
        "en": "Opens preview and contour color settings.",
    },
    "help.compare_areas": {
        "pt": "Sobrepoe a area gerada pelo processamento e a area editada, quando houver ROI personalizada.",
        "en": "Overlays the processing-generated area and the edited area when a custom ROI exists.",
    },
    "help.toggle_results": {"pt": "Mostra ou oculta o painel de resultados.", "en": "Shows or hides the results panel."},
    "help.prev_group": {"pt": "Volta para o grupo anterior.", "en": "Goes back to the previous group."},
    "help.next_group": {"pt": "Avanca para o proximo grupo.", "en": "Moves to the next group."},
    "help.edit_masks": {
        "pt": "Abre uma fila para editar as mascaras das imagens desmarcadas.",
        "en": "Opens a queue to edit masks for unchecked images.",
    },
    "help.restore_auto": {
        "pt": "Remove ROI personalizada das desmarcadas e restaura a segmentacao automatica.",
        "en": "Removes custom ROI from unchecked images and restores automatic segmentation.",
    },
    "help.save_result": {"pt": "Salva CSV/XLSX e overlays na pasta de saida.", "en": "Saves CSV/XLSX files and overlays to the output folder."},
    "help.cancel_processing": {"pt": "Interrompe o processamento em lote em andamento.", "en": "Stops the current batch processing."},
    "guide.step1": {
        "pt": "1) Escolha pasta de imagens e, se quiser, pasta de saida.",
        "en": "1) Choose the image folder and, if needed, the output folder.",
    },
    "guide.step2": {
        "pt": "2) Clique em 'Carregar grupos' para revisar e ajustar o agrupamento.",
        "en": "2) Click 'Load groups' to review and adjust grouping.",
    },
    "guide.step3": {
        "pt": "3) Na revisao, clique em 'Iniciar processamento' para confirmar e processar.",
        "en": "3) In the review, click 'Start processing' to confirm and process.",
    },
    "guide.step4": {
        "pt": "4) Revise mascaras/resultados e clique em 'Salvar resultado'.",
        "en": "4) Review masks/results and click 'Save result'.",
    },
    "guide.review1": {
        "pt": "- Selecione linhas e clique na coluna 'Incluir' para marcar/desmarcar.",
        "en": "- Select rows and click the 'Include' column to check/uncheck.",
    },
    "guide.review2": {
        "pt": "- Use os botoes de edicao para ajustar grupo e tempo antes de iniciar.",
        "en": "- Use the edit buttons to adjust group and time before starting.",
    },
    "guide.review3": {
        "pt": "- Clique em 'Iniciar processamento' para confirmar e seguir.",
        "en": "- Click 'Start processing' to confirm and continue.",
    },
    "metrics.no_group": {"pt": "Sem grupo selecionado.", "en": "No group selected."},
    "metrics.header": {"pt": "=========== RESULTADOS (GRUPO) ===========", "en": "=========== RESULTS (GROUP) ==========="},
    "metrics.reviewed_groups": {"pt": "Grupos revisados", "en": "Reviewed groups"},
    "metrics.no_images_group": {"pt": "Sem imagens no grupo.", "en": "No images in group."},
    "metrics.selected_save": {"pt": "Selecionada para salvar", "en": "Selected for save"},
    "metrics.pending_roi": {"pt": "Pendente redefinicao ROI", "en": "Pending ROI redefinition"},
    "metrics.custom_roi": {"pt": "ROI personalizada", "en": "Custom ROI"},
    "metrics.no_area_marked": {"pt": "Sem area marcada", "en": "No area marked"},
    "metrics.status": {"pt": "Status", "en": "Status"},
    "metrics.no_result": {"pt": "SEM RESULTADO", "en": "NO RESULT"},
    "metrics.not_processed": {"pt": "Nao processada.", "en": "Not processed."},
    "metrics.auto_area": {"pt": "Area automatica", "en": "Automatic area"},
    "metrics.closure_vs_0h": {"pt": "Fechamento vs 0h", "en": "Closure vs 0h"},
    "metrics.processing_time": {"pt": "Tempo proc.", "en": "Processing time"},
    "figure.no_group": {"pt": "Sem grupo selecionado", "en": "No group selected"},
    "figure.no_images_group": {"pt": "Sem imagens para este grupo", "en": "No images for this group"},
    "figure.area": {"pt": "area", "en": "area"},
    "figure.closure": {"pt": "fech", "en": "closure"},
    "figure.pending_roi": {"pt": "pendente ROI", "en": "pending ROI"},
    "figure.no_area": {"pt": "sem area", "en": "no area"},
    "figure.no_area_zeroed": {"pt": "SEM AREA (ZERADA)", "en": "NO AREA (ZEROED)"},
    "figure.processing_area": {"pt": "Processamento", "en": "Processing"},
    "figure.edited_area": {"pt": "Editada", "en": "Edited"},
    "figure.no_result": {"pt": "Sem resultado", "en": "No result"},
    "figure.no_preview": {"pt": "Sem preview disponivel.", "en": "No preview available."},
    "roi.tools_help": {
        "pt": "Ferramentas/atalhos:\nB: pincel\nE: borracha\nX: excluir componente\n[ ]: raio do pincel\nV: cursor ON/OFF\nT: alterna visao\nC: alterna area/contorno\n+ / -: expande/contrai ROI\nDel: limpa tudo\nCtrl+Z: desfaz\nCtrl+Y: refaz\nEnter: salva e avanca (fecha na ultima)\nEsc: cancela\nBotao direito: move trecho local",
        "en": "Tools/shortcuts:\nB: brush\nE: eraser\nX: remove component\n[ ]: brush radius\nV: cursor ON/OFF\nT: toggle view\nC: toggle filled/contour\n+ / -: expand/shrink ROI\nDel: clear all\nCtrl+Z: undo\nCtrl+Y: redo\nEnter: save and advance (close on last)\nEsc: cancel\nRight button: move local segment",
    },
    "roi.tool_desc_title": {"pt": "Descricao da ferramenta:", "en": "Tool description:"},
    "roi.minimize_description": {"pt": "Minimizar descricao", "en": "Minimize description"},
    "roi.show_description": {"pt": "Mostrar descricao", "en": "Show description"},
    "roi.switch_image": {"pt": "Alternar imagem", "en": "Switch image"},
    "roi.tool.brush": {"pt": "Pincel", "en": "Brush"},
    "roi.tool.eraser": {"pt": "Borracha", "en": "Eraser"},
    "roi.tool.component_remove": {"pt": "Excluir componente", "en": "Remove component"},
    "roi.profile.fast50": {"pt": "Rapido (50%)", "en": "Fast (50%)"},
    "roi.desc.eraser": {
        "pt": "Remove partes da mascara. Clique e arraste com o botao esquerdo para apagar com o raio atual.",
        "en": "Removes parts of the mask. Click and drag with the left button to erase with the current radius.",
    },
    "roi.desc.component_remove": {
        "pt": "Remove o componente conectado clicado. Use para excluir ilhas inteiras sem apagar manualmente.",
        "en": "Removes the clicked connected component. Use it to delete whole islands without manual erasing.",
    },
    "roi.desc.brush": {
        "pt": "Adiciona mascara na regiao desejada. Clique e arraste com o botao esquerdo para pintar com o raio atual.",
        "en": "Adds mask in the desired region. Click and drag with the left button to paint with the current radius.",
    },
    "roi.reprocess.file_not_found": {"pt": "Arquivo nao encontrado para {group} {time}.", "en": "File not found for {group} {time}."},
    "roi.reprocess.no_previous_processing": {
        "pt": "Sem processamento previo para {group} {time}.",
        "en": "No previous processing for {group} {time}.",
    },
    "roi.redefine.no_previous_processing": {
        "pt": "Sem processamento previo para redefinir ROI.",
        "en": "No previous processing to redefine ROI.",
    },
    "roi.redefine.retry_title": {"pt": "Redefinicao de ROI", "en": "ROI redefinition"},
    "roi.redefine.retry_message": {
        "pt": "A ROI de {group} {time} nao foi confirmada.\nDeseja tentar novamente?\n\nClique 'Nao' para cancelar toda a fila.",
        "en": "The ROI for {group} {time} was not confirmed.\nDo you want to try again?\n\nClick 'No' to cancel the entire queue.",
    },
    "roi.apply.title_single": {"pt": "Aplicando nova ROI", "en": "Applying new ROI"},
    "roi.apply.title_plural": {"pt": "Aplicando novas ROIs", "en": "Applying new ROIs"},
    "roi.apply.preparing": {
        "pt": "Preparando atualizacao de ROI (0/{total})...",
        "en": "Preparing ROI update (0/{total})...",
    },
    "roi.apply.applying": {
        "pt": "ROI: {group} {time} | aplicando ROI como mascara final ({index}/{total})",
        "en": "ROI: {group} {time} | applying ROI as final mask ({index}/{total})",
    },
    "roi.apply.failed_log": {
        "pt": "Falha ao atualizar {group} {time}: {error}",
        "en": "Failed to update {group} {time}: {error}",
    },
    "roi.apply.done": {
        "pt": "ROI: {group} {time} | mascara final e area (pela ROI) atualizadas ({index}/{total})",
        "en": "ROI: {group} {time} | final mask and area (from ROI) updated ({index}/{total})",
    },
    "roi.apply.failures_warning": {
        "pt": "Algumas ROIs foram definidas, mas falharam na aplicacao da ROI como mascara/area:\n\n{details}",
        "en": "Some ROIs were defined, but failed while applying the ROI as mask/area:\n\n{details}",
    },
    "roi.edit.load_groups_first": {"pt": "Carregue os grupos antes de editar as mascaras.", "en": "Load groups before editing masks."},
    "roi.edit.process_first": {"pt": "Processe as imagens antes de editar as mascaras.", "en": "Process the images before editing masks."},
    "roi.edit.no_unselected": {
        "pt": "Nao ha imagens desmarcadas para editar.\nDesmarque ao menos uma imagem em 'Incluir no salvamento' e tente novamente.",
        "en": "There are no unchecked images to edit.\nUncheck at least one image in 'Include in save' and try again.",
    },
    "roi.edit.incomplete_review_title": {"pt": "Revisao incompleta", "en": "Incomplete review"},
    "roi.edit.incomplete_review_message": {
        "pt": "Antes de editar mascaras, visualize cada imagem ao menos uma vez.\nPendentes de visualizacao: {items}",
        "en": "Before editing masks, view each image at least once.\nPending view: {items}",
    },
    "roi.edit.queue_title": {"pt": "Editar mascaras em fila", "en": "Edit masks in queue"},
    "roi.edit.queue_message": {
        "pt": "Imagens desmarcadas para editar: {count}\n\nO editor sera aberto em sequencia para todas as imagens pendentes.\nAo confirmar uma ROI, a proxima imagem da fila abre automaticamente.\n\nDeseja continuar?",
        "en": "Unchecked images to edit: {count}\n\nThe editor will open sequentially for all pending images.\nAfter confirming an ROI, the next image in the queue opens automatically.\n\nDo you want to continue?",
    },
    "roi.edit.queue_cancelled_status": {"pt": "Edicao de mascaras em fila cancelada.", "en": "Queued mask editing cancelled."},
    "roi.edit.cancelled_status": {"pt": "Redefinicao de ROI cancelada.", "en": "ROI redefinition cancelled."},
    "roi.edit.pending_without_redefinition": {
        "pt": "{count} imagem(ns) pendente(s) ficaram sem redefinicao de ROI.",
        "en": "{count} image(s) remained pending without ROI redefinition.",
    },
    "roi.edit.updated_status": {
        "pt": "ROI editada e area substituida em {count} imagem(ns) desmarcada(s).",
        "en": "ROI edited and area replaced in {count} unchecked image(s).",
    },
    "roi.edit.no_confirmed_status": {
        "pt": "Nenhuma ROI foi confirmada; nenhuma imagem desmarcada foi alterada.",
        "en": "No ROI was confirmed; no unchecked image was changed.",
    },
    "roi.restore.no_group": {
        "pt": "Selecione um grupo para restaurar o automatico das desmarcadas.",
        "en": "Select a group to restore automatic segmentation for unchecked images.",
    },
    "roi.restore.no_unselected": {
        "pt": "Este botao so funciona para imagens desmarcadas em 'Selecionar para salvar'.\nDesmarque ao menos uma imagem (0h/24h/48h) e tente novamente.",
        "en": "This button only works for images unchecked in 'Select for save'.\nUncheck at least one image (0h/24h/48h) and try again.",
    },
    "roi.restore.confirm_title": {"pt": "Restaurar segmentacao automatica", "en": "Restore automatic segmentation"},
    "roi.restore.confirm_message": {
        "pt": "Grupo: {group}\nImagens desmarcadas: {times}\n\nA ROI personalizada sera removida e a segmentacao automatica sera recalculada.\nAs imagens continuarao desmarcadas para salvar.\n\nDeseja continuar?",
        "en": "Group: {group}\nUnchecked images: {times}\n\nThe custom ROI will be removed and automatic segmentation will be recalculated.\nThe images will remain unchecked for saving.\n\nDo you want to continue?",
    },
    "roi.restore.cancelled_status": {
        "pt": "Restauracao automatica das desmarcadas cancelada.",
        "en": "Automatic restoration for unchecked images cancelled.",
    },
    "roi.restore.progress_title": {"pt": "Restaurando segmentacao automatica", "en": "Restoring automatic segmentation"},
    "roi.restore.progress_initial": {
        "pt": "Restaurando segmentacao automatica (0/{total})...",
        "en": "Restoring automatic segmentation (0/{total})...",
    },
    "roi.restore.progress_item": {
        "pt": "Restaurando {group} {time} ({index}/{total})",
        "en": "Restoring {group} {time} ({index}/{total})",
    },
    "roi.restore.failed_log": {
        "pt": "Falha ao restaurar {group} {time}: {error}",
        "en": "Failed to restore {group} {time}: {error}",
    },
    "roi.restore.progress_done": {"pt": "Restaurado {group} {time} ({index}/{total})", "en": "Restored {group} {time} ({index}/{total})"},
    "roi.restore.failures_status": {
        "pt": "ROI personalizada removida em {removed} item(ns); automatico restaurado em {restored}/{total}.",
        "en": "Custom ROI removed from {removed} item(s); automatic segmentation restored in {restored}/{total}.",
    },
    "roi.restore.failures_title": {
        "pt": "Restauracao automatica com avisos",
        "en": "Automatic restoration with warnings",
    },
    "roi.restore.failures_message": {
        "pt": "Algumas imagens nao puderam ser restauradas para a segmentacao automatica:\n\n{details}",
        "en": "Some images could not be restored to automatic segmentation:\n\n{details}",
    },
    "roi.restore.done_status": {
        "pt": "ROI personalizada removida em {removed} item(ns) e automatico restaurado em {restored} imagem(ns).",
        "en": "Custom ROI removed from {removed} item(s) and automatic segmentation restored in {restored} image(s).",
    },
    "roi.no_area.no_processing": {
        "pt": "Sem processamento para marcar 'sem area': {group} {time}.",
        "en": "No processing available to mark 'no area': {group} {time}.",
    },
    "roi.editor.unavailable_title": {"pt": "Editor de ROI indisponivel", "en": "ROI editor unavailable"},
    "roi.editor.unavailable_message": {
        "pt": "Nao foi possivel carregar matplotlib.pyplot.\n\nDetalhes: {error}",
        "en": "Could not load matplotlib.pyplot.\n\nDetails: {error}",
    },
    "roi.editor.invalid_image": {"pt": "Imagem invalida para editor de ROI.", "en": "Invalid image for ROI editor."},
    "roi.status_current": {"pt": "Status atual:", "en": "Current status:"},
    "roi.profile": {"pt": "Perfil", "en": "Profile"},
    "roi.scale": {"pt": "Escala", "en": "Scale"},
    "roi.next": {"pt": "prox.", "en": "next"},
    "roi.tool": {"pt": "Ferramenta", "en": "Tool"},
    "roi.radius": {"pt": "Raio", "en": "Radius"},
    "roi.cursor": {"pt": "Cursor", "en": "Cursor"},
    "roi.view": {"pt": "Visao", "en": "View"},
    "roi.edge": {"pt": "Borda", "en": "Edge"},
    "roi.contour": {"pt": "Contorno", "en": "Contour"},
    "yes": {"pt": "SIM", "en": "YES"},
    "no": {"pt": "NAO", "en": "NO"},
}
UI_LITERAL_PT_TO_EN: dict[str, str] = {
    text["pt"]: text["en"]
    for text in UI_TEXT.values()
    if "pt" in text and "en" in text and text["pt"] != text["en"]
}
UI_LITERAL_EN_TO_PT: dict[str, str] = {en: pt for pt, en in UI_LITERAL_PT_TO_EN.items()}
_PYPLOT_MODULE = None
_PYPLOT_IMPORT_ERROR: Exception | None = None


def _get_pyplot():
    """Importa pyplot apenas quando necessario (evita custo na abertura da GUI)."""
    global _PYPLOT_MODULE, _PYPLOT_IMPORT_ERROR
    if _PYPLOT_MODULE is not None:
        return _PYPLOT_MODULE
    if _PYPLOT_IMPORT_ERROR is not None:
        raise _PYPLOT_IMPORT_ERROR

    try:
        import matplotlib.pyplot as pyplot
    except Exception as exc:
        _PYPLOT_IMPORT_ERROR = exc
        raise

    _PYPLOT_MODULE = pyplot
    return _PYPLOT_MODULE


def _close_all_pyplot_figures() -> None:
    """Fecha figuras abertas pelo pyplot sem disparar novo import."""
    if _PYPLOT_MODULE is None:
        return
    try:
        _PYPLOT_MODULE.close("all")
    except Exception:
        pass


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


def _strip_accents(text: str) -> str:
    """Remove acentos para comparacoes mais robustas em exibicao/ordenacao."""
    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def mask_to_polygon(mask: np.ndarray) -> np.ndarray:
    """Converte mascara binaria em poligono fechado (contorno externo principal)."""
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
    """Rasteriza vertices de poligono para mascara binaria no tamanho informado."""
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
    """Redimensiona mascara usando vizinho mais proximo para preservar rotulos."""
    th, tw = target_shape
    if mask.shape == target_shape:
        return mask.astype(bool)
    return cv2.resize(mask.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST) > 0


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Mantem apenas o maior componente conexo da mascara."""
    src = np.asarray(mask, dtype=bool)
    if src.ndim != 2:
        raise ValueError("mask deve ser 2D.")
    if not np.any(src):
        return np.zeros_like(src, dtype=bool)

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(src.astype(np.uint8), connectivity=8)
    if labels_count <= 2:
        return src

    areas = stats[1:, cv2.CC_STAT_AREA]
    keep_label = int(np.argmax(areas) + 1)
    return labels == keep_label


def offset_mask_sdf(mask: np.ndarray, offset_px: float, smooth_sigma: float = 0.0) -> np.ndarray:
    """Expande/contrai mascara via distancia assinada com suavizacao opcional."""
    src = np.asarray(mask).astype(bool)
    if src.ndim != 2:
        raise ValueError("mask deve ser 2D.")
    if not np.any(src):
        return np.zeros_like(src, dtype=bool)

    src_u8 = src.astype(np.uint8) * 255
    # PT: Pad para manter distancia assinada consistente mesmo quando a ROI toca a borda. | EN: Pads the mask to keep the signed distance consistent even when the ROI touches the border.
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
    """Deriva a pasta inicial de busca a partir de BASE_IMAGE_PATH."""
    base = str(BASE_IMAGE_PATH).strip()
    if not base:
        return ""
    p = Path(base)
    if p.suffix:
        return str(p.parent)
    return str(p)


class CORAApp:
    """Controlador da GUI: estado da sessao, eventos de UI e orquestracao do processamento."""

    def __init__(
        self,
        folder_path: str = "",
        output_folder_path: str = "",
    ) -> None:
        folder_init = str(folder_path).strip()
        output_init = str(output_folder_path).strip()

        self._folder_from_cli = bool(folder_init)
        self._output_from_cli = bool(output_init)
        self.settings_path = self._resolve_settings_path()
        self.ui_language_key = UI_LANGUAGE_DEFAULT

        self.root = tk.Tk()
        self.root.title(self._t("app.title"))
        self.root.geometry("1440x860")
        self.root.minsize(1180, 760)
        self.style = ttk.Style(self.root)

        self.folder_var = tk.StringVar(value=folder_init)
        self.output_folder_var = tk.StringVar(value=output_init)
        self.group_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value=self._t("status.ready"))
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_percent_var = tk.StringVar(value="0.0%")
        self.progress_current_image_var = tk.StringVar(value="-")
        self.progress_remaining_var = tk.StringVar(value="-")
        self.progress_eta_var = tk.StringVar(value="--")
        self.popup_status_var = tk.StringVar(value="")
        self.popup_progress_var = tk.DoubleVar(value=0.0)
        self.popup_percent_var = tk.StringVar(value="0.0%")
        self.editor_profile_key = ROI_EDITOR_PROFILE_DEFAULT
        self.editor_profile_var = tk.StringVar(value=self._roi_editor_profile_label(self.editor_profile_key))
        self.preview_mode_key = PREVIEW_MODE_DEFAULT
        self.preview_mode_var = tk.StringVar(value=self._preview_mode_label(self.preview_mode_key))
        self.preview_source_mode_key = PREVIEW_SOURCE_MODE_DEFAULT
        self.preview_source_mode_var = tk.StringVar(value=self._preview_source_mode_label(self.preview_source_mode_key))
        self.preview_color_mode_key = PREVIEW_COLOR_MODE_DEFAULT
        self.preview_color_mode_var = tk.StringVar(value=self._preview_color_mode_label(self.preview_color_mode_key))
        self.compare_area_overlay_var = tk.BooleanVar(value=False)
        self.ui_theme_key = UI_THEME_DEFAULT
        self.preview_shared_contour_color: tuple[int, int, int] = PREVIEW_CONTOUR_COLOR_SHARED_DEFAULT
        self.preview_contour_colors: dict[str, tuple[int, int, int]] = dict(PREVIEW_CONTOUR_COLORS_DEFAULT)
        self.preview_original_image_cache: dict[str, np.ndarray] = {}
        self.preview_enhanced_image_cache: dict[str, np.ndarray] = {}
        self._roi_editor_window_state: dict[str, bool] = {"fullscreen": False, "zoomed": False}

        self.config = CORAConfig()
        self.group_files: dict[str, dict[str, Path]] = {}
        self.group_labels: dict[str, str] = {}
        self.label_to_group: dict[str, str] = {}
        self.group_order: list[str] = []
        self.current_group_key: str | None = None

        self.roi_by_item: dict[tuple[str, str], np.ndarray] = {}
        self.processing_mask_by_item: dict[tuple[str, str], np.ndarray] = {}
        self.processed_by_group: dict[str, dict[str, ProcessedTimepoint]] = {}
        self.processing_errors: dict[tuple[str, str], str] = {}
        self.pending_reprocess: set[tuple[str, str]] = set()
        self.reprocessed_history: set[tuple[str, str]] = set()
        self.reviewed_groups: set[str] = set()
        self.reviewed_items: set[tuple[str, str]] = set()
        self.no_area_items: set[tuple[str, str]] = set()
        self.no_area_backup: dict[tuple[str, str], tuple[ProcessedTimepoint | None, str | None]] = {}
        # PT: O backend ainda exige o campo area_manual para compor AreaResults, mas a interface nao depende mais desse dado.
        # EN: The backend still requires the area_manual field to build AreaResults, but the interface no longer depends on this value.
        self._area_reference_px = 1

        self.ui_busy = False
        self.worker_thread: threading.Thread | None = None
        self.worker_queue: queue.Queue[tuple] = queue.Queue()
        self.cancel_event = threading.Event()
        self.active_processing_mode: str | None = None
        self.processing_started_at: float | None = None
        self.processing_total_items = 0
        self._start_processing_after_grouping = False
        self.progress_popup: tk.Toplevel | None = None
        self.cancel_btn: ttk.Button | None = None
        self.toggle_results_btn: ttk.Button | None = None
        self.open_config_btn: ttk.Menubutton | None = None
        self.process_all_btn: ttk.Button | None = None
        self.go_viewer_btn: ttk.Button | None = None
        self.test_selection_page: ttk.Frame | None = None
        self.single_image_test_btn: ttk.Button | None = None
        self.batch_mode_btn: ttk.Button | None = None
        self.robotized_test_btn: ttk.Button | None = None
        self.folder_entry: ttk.Entry | None = None
        self.output_folder_entry: ttk.Entry | None = None
        self.batch_test_buttons: list[ttk.Button] = []
        self.test_modes_btn_config: ttk.Button | None = None
        self.config_page: ttk.Frame | None = None
        self.viewer_page: ttk.Frame | None = None
        self.viewer_bottom: ttk.Frame | None = None
        self.body_paned: ttk.Panedwindow | None = None
        self.left_panel: ttk.Frame | None = None
        self.right_panel: ttk.Frame | None = None
        self.results_panel_visible = True
        self.side_notebook: ttk.Notebook | None = None
        self.metrics_tab: ttk.Frame | None = None
        self.metrics_text: tk.Text | None = None
        self.progress_tab: ttk.Frame | None = None
        self.config_guide_btn: ttk.Button | None = None
        self.viewer_guide_btn: ttk.Button | None = None
        self.theme_toggle_btn: ttk.Button | None = None
        self.theme_toggle_btn_config: ttk.Button | None = None
        self.theme_settings_popup: tk.Toplevel | None = None
        self.theme_family_var = tk.StringVar()
        self.theme_mode_var = tk.StringVar()
        self.theme_family_combo: ttk.Combobox | None = None
        self.theme_mode_combo: ttk.Combobox | None = None
        self.language_toggle_btn_config: ttk.Button | None = None
        self.guide_popup: tk.Toplevel | None = None
        self.guide_popup_text: tk.Text | None = None
        self.inline_progress_frame: ttk.Frame | None = None
        self.progress_log_text: tk.Text | None = None
        self.progress_log_last_status = ""
        self._progress_popup_dismissed = False
        self._guide_text_cache = ""
        self._tooltip_popup: tk.Toplevel | None = None
        self._tooltip_after_id: str | None = None
        self._tooltip_text_by_widget: dict[str, str] = {}
        self._tooltip_bound_widgets: set[str] = set()
        self.editor_profile_combo: ttk.Combobox | None = None
        self.contour_settings_btn: ttk.Button | None = None
        self.contour_settings_popup: tk.Toplevel | None = None
        self.paths_mini_popup: tk.Toplevel | None = None
        self.group_navigation_popup: tk.Toplevel | None = None
        self.preview_mode_combo: ttk.Combobox | None = None
        self.preview_source_mode_combo: ttk.Combobox | None = None
        self.preview_color_mode_combo: ttk.Combobox | None = None
        self.preview_shared_color_button: tk.Button | None = None
        self.preview_shared_color_label: ttk.Label | None = None
        self.preview_shared_color_row: ttk.Frame | None = None
        self.preview_per_time_color_label: ttk.Label | None = None
        self.preview_per_time_color_row: ttk.Frame | None = None
        self.preview_color_buttons: dict[str, tk.Button] = {}
        self.compare_area_overlay_check: ttk.Checkbutton | None = None
        self.open_config_menu: tk.Menu | None = None
        self.language_toggle_menu_index: int | None = None
        self.bg_image_label: tk.Label | None = None
        self.bg_source_image = None
        self.bg_photo = None
        self.bg_last_size: tuple[int, int] = (0, 0)
        self.bg_resize_after_id: str | None = None
        self._shutdown_in_progress = False
        self._force_exit_timer: threading.Timer | None = None
        self._single_test_context: dict[str, object] | None = None
        self._single_test_resource_peaks = ResourcePeaks(None, None)
        self._single_test_runs: list[dict[str, object]] = []
        self._batch_test_context: dict[str, object] | None = None
        self._batch_test_rows: list[dict[str, object]] = []
        self._processing_metrics_by_item: dict[tuple[str, str], dict[str, object]] = {}
        self._page_time_started_at: datetime | None = None
        self._active_page_name: str | None = None
        self._active_page_started_at: float | None = None
        self._page_time_totals: dict[str, float] = {
            "selecao_de_teste": 0.0,
            "configuracoes": 0.0,
            "visualizacao": 0.0,
            "espera_processamento_imagem": 0.0,
            "edicao_de_mascara": 0.0,
        }
        self._page_visit_counts: dict[str, int] = {
            page_name: 0 for page_name in self._page_time_totals
        }
        self._robot_review_actions: dict[str, Callable[[], None]] = {}
        self._robot_save_selection_index = 0
        self._robot_paths_confirmed: set[str] = set()
        self._robotized_test_active = False

        self.refazer_vars: dict[str, tk.BooleanVar] = {}
        self.refazer_checks: dict[str, ttk.Checkbutton] = {}
        self.no_area_vars: dict[str, tk.BooleanVar] = {}
        self.no_area_checks: dict[str, ttk.Checkbutton] = {}
        self._folder_var_trace_id: str | None = None
        self._last_folder_entry_value = ""
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._apply_ui_theme_styles()
        self._load_user_settings()
        if not self.folder_var.get().strip():
            self.folder_var.set(default_folder_from_base_path())

        self._setup_background_image()
        self._build_layout()
        self.root.bind("<Control-r>", self._open_robot_test_shortcut, add="+")
        self.root.bind("<F2>", self._open_robot_test_shortcut, add="+")
        self.root.bind("<F6>", lambda _e: self._set_robot_field_from_clipboard("images"), add="+")
        self.root.bind("<F7>", lambda _e: self._set_robot_field_from_clipboard("output"), add="+")
        self.root.bind("<F8>", self._robot_load_groups_shortcut, add="+")
        self.root.bind("<F9>", lambda _e: self._run_robot_review_action("mark_all"), add="+")
        self.root.bind("<F10>", lambda _e: self._run_robot_review_action("auto_fill"), add="+")
        self.root.bind("<F11>", lambda _e: self._run_robot_review_action("start"), add="+")
        self.root.bind("<F12>", self._robot_select_next_save_item, add="+")
        self.root.bind("<Shift-F6>", self._robot_next_group_shortcut, add="+")
        self.root.bind("<Shift-F7>", self._robot_edit_masks_shortcut, add="+")
        self.root.bind("<Shift-F12>", self._robot_save_results_shortcut, add="+")
        self._start_page_timing("selecao_de_teste")
        self._apply_ui_theme_styles()
        self._refresh_language_texts()
        self._last_folder_entry_value = self._normalize_folder_entry(self.folder_var.get())
        self._folder_var_trace_id = self.folder_var.trace_add("write", self._on_folder_var_changed)
        self._hide_results_panel_on_startup()
        self._set_ui_busy(False)
        self._refresh_metrics()
        self._refresh_figure()

    def _center_toplevel_on_root(
        self,
        popup: tk.Toplevel,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        """Centraliza uma janela filha sobre a janela principal, com fallback para a tela."""
        try:
            self.root.update_idletasks()
            popup.update_idletasks()

            popup_width = int(width or max(popup.winfo_width(), popup.winfo_reqwidth(), 1))
            popup_height = int(height or max(popup.winfo_height(), popup.winfo_reqheight(), 1))

            root_width = int(max(self.root.winfo_width(), 1))
            root_height = int(max(self.root.winfo_height(), 1))
            root_x = int(self.root.winfo_rootx())
            root_y = int(self.root.winfo_rooty())

            if root_width <= 1 or root_height <= 1:
                screen_x = int(popup.winfo_vrootx())
                screen_y = int(popup.winfo_vrooty())
                screen_width = int(popup.winfo_vrootwidth() or popup.winfo_screenwidth())
                screen_height = int(popup.winfo_vrootheight() or popup.winfo_screenheight())
                x = screen_x + max(0, (screen_width - popup_width) // 2)
                y = screen_y + max(0, (screen_height - popup_height) // 2)
            else:
                x = root_x + max(0, (root_width - popup_width) // 2)
                y = root_y + max(0, (root_height - popup_height) // 2)

            screen_x = int(popup.winfo_vrootx())
            screen_y = int(popup.winfo_vrooty())
            screen_width = int(popup.winfo_vrootwidth() or popup.winfo_screenwidth())
            screen_height = int(popup.winfo_vrootheight() or popup.winfo_screenheight())
            x = max(screen_x, min(x, screen_x + max(0, screen_width - popup_width)))
            y = max(screen_y, min(y, screen_y + max(0, screen_height - popup_height)))

            popup.geometry(f"{popup_width}x{popup_height}+{x}+{y}")
        except Exception:
            pass

    def _resolve_background_image_path(self) -> Path | None:
        candidates = [
            Path.cwd() / "img",
            Path(__file__).resolve().parent / "img",
            Path(__file__).resolve().parent.parent / "img",
        ]
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
        for img_dir in candidates:
            if not img_dir.exists() or not img_dir.is_dir():
                continue
            files = sorted(
                [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts],
                key=lambda p: p.name.lower(),
            )
            if files:
                return files[0]
        return None

    def _setup_background_image(self) -> None:
        if Image is None or ImageTk is None or ImageOps is None:
            return
        path = self._resolve_background_image_path()
        if path is None:
            return
        try:
            self.bg_source_image = Image.open(path).convert("RGB")
        except Exception:
            self.bg_source_image = None
            return
        self.bg_image_label = tk.Label(self.root, borderwidth=0, highlightthickness=0)
        self.bg_image_label.place(x=0, y=0, relwidth=1, relheight=1)
        self.bg_image_label.lower()
        self.root.bind("<Configure>", self._on_root_resized, add="+")
        self.root.after(10, self._render_background_image)

    def _on_root_resized(self, event=None) -> None:
        if event is not None and event.widget is not self.root:
            return
        if self.bg_resize_after_id is not None:
            try:
                self.root.after_cancel(self.bg_resize_after_id)
            except Exception:
                pass
        self.bg_resize_after_id = self.root.after(40, self._render_background_image)

    def _render_background_image(self) -> None:
        self.bg_resize_after_id = None
        if Image is None or ImageTk is None or ImageOps is None:
            return
        label = self.bg_image_label
        source = self.bg_source_image
        if label is None or source is None:
            return
        try:
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            w = max(1, int(self.root.winfo_width()))
            h = max(1, int(self.root.winfo_height()))
            if (w, h) == self.bg_last_size:
                return
            self.bg_last_size = (w, h)
            fitted = ImageOps.fit(source, (w, h), method=resampling, centering=(0.5, 0.5))
            # PT: 20% de visibilidade da imagem no fundo. | EN: 20% visibility for the background image.
            palette = self._theme_palette()
            blend_color = tuple(int(np.clip(c, 0, 255)) for c in palette.get("bg_blend_rgb", (255, 255, 255)))
            base = Image.new("RGB", (w, h), blend_color)
            faded = Image.blend(base, fitted, 0.20)
            self.bg_photo = ImageTk.PhotoImage(faded)
            label.configure(image=self.bg_photo)
            label.lower()
        except Exception:
            return

    def _resolve_settings_path(self) -> Path:
        local_appdata = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if local_appdata:
            base_dir = Path(local_appdata) / APP_ID
        else:
            base_dir = Path.home() / f".{APP_ID.lower()}"
        return base_dir / "settings.json"

    def _load_user_settings(self) -> None:
        """Restaura preferencias persistidas da interface (perfil, cores e modos)."""
        path = self.settings_path
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, dict):
            return

        language_raw = data.get("ui_language")
        if language_raw is not None:
            self.ui_language_key = self._normalize_ui_language_key(language_raw)

        theme_raw = data.get("ui_theme")
        if theme_raw is not None:
            self._set_ui_theme(theme_raw, save=False, update_status=False)

        profile_raw = data.get("roi_editor_profile")
        if profile_raw is not None:
            self._set_roi_editor_profile(profile_raw, save=False, update_status=False)

        preview_mode_raw = data.get("preview_mode")
        if preview_mode_raw is not None:
            self._set_preview_mode(preview_mode_raw, save=False, update_status=False)

        # PT: Sempre iniciar exibindo a imagem original. | EN: Always start by displaying the original image.
        self._set_preview_source_mode("original", save=False, update_status=False)

        preview_color_mode_raw = data.get("preview_color_mode")
        if preview_color_mode_raw is not None:
            self._set_preview_color_mode(preview_color_mode_raw, save=False, update_status=False)

        compare_area_overlay_raw = data.get("compare_area_overlay")
        if compare_area_overlay_raw is not None:
            self._set_compare_area_overlay(bool(compare_area_overlay_raw), save=False, update_status=False)

        preview_shared_color = data.get("preview_contour_color_shared")
        parsed_shared = self._normalize_rgb_triplet(preview_shared_color)
        if parsed_shared is not None:
            self.preview_shared_contour_color = parsed_shared

        preview_colors = data.get("preview_contour_colors")
        if isinstance(preview_colors, dict):
            for time_tag in TIME_ORDER:
                parsed = self._normalize_rgb_triplet(preview_colors.get(time_tag))
                if parsed is not None:
                    self.preview_contour_colors[time_tag] = parsed

    def _save_user_settings(self) -> None:
        """Salva preferencias de interface para reaproveitamento nas proximas execucoes."""
        payload = {
            "ui_language": self.ui_language_key,
            "ui_theme": self.ui_theme_key,
            "roi_editor_profile": self.editor_profile_key,
            "preview_mode": self.preview_mode_key,
            "preview_source_mode": self.preview_source_mode_key,
            "preview_color_mode": self.preview_color_mode_key,
            "compare_area_overlay": bool(self.compare_area_overlay_var.get()),
            "preview_contour_color_shared": list(self.preview_shared_contour_color),
            "preview_contour_colors": {
                time_tag: list(
                    tuple(int(np.clip(c, 0, 255)) for c in self.preview_contour_colors.get(time_tag, self.preview_shared_contour_color))
                )
                for time_tag in TIME_ORDER
            },
            "updated_at": int(time.time()),
        }

        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            self.settings_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    @staticmethod
    def _normalize_ui_language_key(raw_value: object) -> str:
        text = _strip_accents(str(raw_value or "").strip().lower())
        if text in {"en", "eng", "english", "ingles", "ing"}:
            return "en"
        if text in {"pt", "por", "portugues", "portuguese", "br", "pt-br", "pt_br"}:
            return "pt"
        return UI_LANGUAGE_DEFAULT

    def _t(self, key: str, **kwargs: object) -> str:
        language = self._normalize_ui_language_key(getattr(self, "ui_language_key", UI_LANGUAGE_DEFAULT))
        entry = UI_TEXT.get(str(key), {})
        text = str(entry.get(language) or entry.get("pt") or key)
        if kwargs:
            try:
                return text.format(**kwargs)
            except Exception:
                return text
        return text

    def _ui_language_name(self, language_key: str | None = None) -> str:
        key = self._normalize_ui_language_key(language_key or self.ui_language_key)
        language = self._normalize_ui_language_key(self.ui_language_key)
        return UI_LANGUAGE_NAMES.get(key, UI_LANGUAGE_NAMES[UI_LANGUAGE_DEFAULT]).get(language, key)

    def _ui_language_toggle_label(self) -> str:
        key = self._normalize_ui_language_key(self.ui_language_key)
        return self._t("language.switch_to_pt" if key == "en" else "language.switch_to_en")

    def _translate_literal(self, text: object) -> str:
        raw = str(text)
        language = self._normalize_ui_language_key(self.ui_language_key)
        if language == "en":
            return UI_LITERAL_PT_TO_EN.get(raw, raw)
        return UI_LITERAL_EN_TO_PT.get(raw, raw)

    def _translate_processing_stage(self, stage: object) -> str:
        raw = str(stage)
        exact_keys = {
            "Preparando imagem": "processing.stage.preparing_image",
            "Carregando imagem": "processing.stage.loading_image",
            "Configurando parametros": "processing.stage.configuring_params",
            "Analisando brilho medio": "processing.stage.avg_brightness",
            "Preparando ROI": "processing.stage.prepare_roi",
            "Pre-processando imagem": "processing.stage.preprocess",
            "Estimando iteracoes de Gabor": "processing.stage.estimate_gabor",
            "Executando filtros de Gabor": "processing.stage.run_gabor",
            "Montando mapa de textura": "processing.stage.texture_map",
            "Estimando iteracoes do k-means": "processing.stage.estimate_kmeans",
            "Separando regioes (k-means)": "processing.stage.kmeans_regions",
            "Pos-processando mascara": "processing.stage.postprocess_mask",
            "Refinando mascara por modo adaptativo": "processing.stage.refine_adaptive",
            "Calculando metricas": "processing.stage.metrics",
            "Concluido": "processing.stage.completed",
            "Finalizado": "processing.stage.finished",
        }
        key = exact_keys.get(raw)
        if key is not None:
            return self._t(key)

        match = re.fullmatch(r"Preparando ROI especial de 24h \((.+)\)", raw)
        if match:
            return self._t("processing.stage.special_24h_roi", origin=match.group(1))

        match = re.fullmatch(r"Modo adaptativo: (.+) \(media=([^)]+)\)", raw)
        if match:
            return self._t("processing.stage.adaptive_mode", mode=match.group(1), mean=match.group(2))

        return self._translate_literal(raw)

    def _translate_widget_tree(self, widget: tk.Widget | None) -> None:
        if widget is None:
            return
        try:
            if not widget.winfo_exists():
                return
        except Exception:
            return

        try:
            current = widget.cget("text")
            translated = self._translate_literal(current)
            if translated != current:
                widget.configure(text=translated)
        except Exception:
            pass

        try:
            children = list(widget.winfo_children())
        except Exception:
            children = []
        for child in children:
            self._translate_widget_tree(child)

    def _refresh_language_toggle_controls(self) -> None:
        label = self._ui_language_toggle_label()
        btn = self.language_toggle_btn_config
        if btn is not None:
            try:
                btn.configure(text=label)
            except Exception:
                pass

        menu = self.open_config_menu
        idx = self.language_toggle_menu_index
        if menu is not None and idx is not None:
            try:
                menu.entryconfigure(idx, label=label)
            except Exception:
                pass

    def _refresh_open_config_menu_labels(self) -> None:
        if self.open_config_btn is not None:
            try:
                self.open_config_btn.configure(text=self._t("menu.configurations_button"))
            except Exception:
                pass

        menu = self.open_config_menu
        if menu is None:
            return
        entries = {
            0: "menu.paths",
            1: "menu.group_navigation",
            2: "menu.contour_colors",
            4: "menu.open_full_settings",
            6: "test.menu_choose",
        }
        for idx, key in entries.items():
            try:
                menu.entryconfigure(idx, label=self._t(key))
            except Exception:
                pass
        self._refresh_language_toggle_controls()

    def _results_toggle_label(self) -> str:
        return self._t("button.hide_results" if self.results_panel_visible else "button.show_results")

    def _refresh_results_toggle_button(self) -> None:
        if self.toggle_results_btn is None:
            return
        try:
            self.toggle_results_btn.configure(text=self._results_toggle_label())
        except Exception:
            pass

    def _refresh_compare_area_overlay_check(self) -> None:
        if self.compare_area_overlay_check is None:
            return
        try:
            self.compare_area_overlay_check.configure(text=self._t("button.compare_areas"))
        except Exception:
            pass

    def _refresh_preview_combo_labels(self) -> None:
        self.editor_profile_var.set(self._roi_editor_profile_label(self.editor_profile_key))
        if self.editor_profile_combo is not None:
            try:
                self.editor_profile_combo.set(self._roi_editor_profile_label(self.editor_profile_key))
            except Exception:
                pass

        self.preview_mode_var.set(self._preview_mode_label(self.preview_mode_key))
        if self.preview_mode_combo is not None:
            try:
                self.preview_mode_combo.configure(values=self._preview_mode_values())
                self.preview_mode_combo.set(self._preview_mode_label(self.preview_mode_key))
            except Exception:
                pass

        self.preview_source_mode_var.set(self._preview_source_mode_label(self.preview_source_mode_key))
        if self.preview_source_mode_combo is not None:
            try:
                self.preview_source_mode_combo.configure(values=self._preview_source_mode_values())
                self.preview_source_mode_combo.set(self._preview_source_mode_label(self.preview_source_mode_key))
            except Exception:
                pass

        self.preview_color_mode_var.set(self._preview_color_mode_label(self.preview_color_mode_key))
        if self.preview_color_mode_combo is not None:
            try:
                self.preview_color_mode_combo.configure(values=self._preview_color_mode_values())
                self.preview_color_mode_combo.set(self._preview_color_mode_label(self.preview_color_mode_key))
            except Exception:
                pass
        self._update_preview_color_buttons()

    def _refresh_popup_titles(self) -> None:
        title_pairs = (
            (self.guide_popup, "title.guide"),
            (self.paths_mini_popup, "title.quick_paths"),
            (self.group_navigation_popup, "title.quick_groups"),
            (self.theme_settings_popup, "title.theme_settings"),
            (self.contour_settings_popup, "title.contour_settings"),
        )
        for popup, key in title_pairs:
            if popup is None:
                continue
            try:
                if popup.winfo_exists():
                    popup.title(self._t(key))
            except Exception:
                pass

    def _refresh_language_texts(self) -> None:
        self.ui_language_key = self._normalize_ui_language_key(self.ui_language_key)
        try:
            self.root.title(self._t("app.title"))
        except Exception:
            pass

        try:
            current_status = self.status_var.get()
            translated_status = self._translate_literal(current_status)
            if translated_status != current_status:
                self.status_var.set(translated_status)
        except Exception:
            pass

        self._translate_widget_tree(self.root)
        self._refresh_theme_toggle_buttons()
        self._refresh_language_toggle_controls()
        self._refresh_open_config_menu_labels()
        self._refresh_results_toggle_button()
        self._refresh_compare_area_overlay_check()
        self._refresh_preview_combo_labels()
        self._refresh_popup_titles()

        if self.side_notebook is not None and self.metrics_tab is not None:
            try:
                self.side_notebook.tab(self.metrics_tab, text=self._t("tab.results"))
            except Exception:
                pass

        self._refresh_help_tab()
        if self.metrics_text is not None:
            try:
                if self.metrics_text.winfo_exists():
                    self._refresh_metrics(self.current_group_key)
            except Exception:
                pass
        if hasattr(self, "figure") and hasattr(self, "canvas"):
            try:
                self._refresh_figure(self.current_group_key)
            except Exception:
                pass

    def _set_ui_language(self, raw_value: object, save: bool = False, update_status: bool = False) -> None:
        key = self._normalize_ui_language_key(raw_value)
        self.ui_language_key = key
        self._refresh_language_texts()
        if update_status:
            self.status_var.set(self._t("status.language_changed", language=self._ui_language_name(key)))
        if save:
            self._save_user_settings()

    def _toggle_ui_language(self) -> None:
        current = self._normalize_ui_language_key(self.ui_language_key)
        target = "pt" if current == "en" else "en"
        self._set_ui_language(target, save=True, update_status=True)

    @staticmethod
    def _normalize_ui_theme_key(raw_value: object) -> str:
        text = _strip_accents(str(raw_value or "").strip().lower())
        text = text.replace("-", "_").replace(" ", "_")
        if text in UI_THEME_PRESETS:
            return text

        legacy_aliases = {
            "claro": "rose_light",
            "light": "rose_light",
            "tema_claro": "rose_light",
            "escuro": "rose_dark",
            "dark": "rose_dark",
            "tema_escuro": "rose_dark",
            "padrao": "standard_light",
            "standard": "standard_light",
            "default": "standard_light",
        }
        if text in legacy_aliases:
            return legacy_aliases[text]

        family_aliases = {
            "standard": ("standard", "padrao", "default", "biblioteca", "library"),
            "rose": ("rose", "rosa"),
            "blue": ("blue", "azul"),
            "green": ("green", "verde"),
            "purple": ("purple", "roxo"),
            "graphite": ("graphite", "grafite"),
        }
        mode_aliases = {
            "light": ("light", "claro"),
            "dark": ("dark", "escuro"),
        }
        for family_key, family_names in family_aliases.items():
            for mode_key, mode_names in mode_aliases.items():
                theme_key = f"{family_key}_{mode_key}"
                for family_name in family_names:
                    for mode_name in mode_names:
                        if text in {f"{family_name}_{mode_name}", f"{mode_name}_{family_name}"}:
                            return theme_key
        return UI_THEME_DEFAULT

    @staticmethod
    def _ui_theme_family_key(theme_key: str) -> str:
        key = CORAApp._normalize_ui_theme_key(theme_key)
        family, _sep, _mode = key.rpartition("_")
        return family if family in UI_THEME_FAMILY_ORDER else UI_THEME_FAMILY_ORDER[0]

    @staticmethod
    def _ui_theme_mode_key(theme_key: str) -> str:
        key = CORAApp._normalize_ui_theme_key(theme_key)
        _family, _sep, mode = key.rpartition("_")
        return mode if mode in UI_THEME_MODE_ORDER else UI_THEME_MODE_ORDER[0]

    def _theme_palette(self) -> dict[str, object]:
        key = self._normalize_ui_theme_key(self.ui_theme_key)
        palette = UI_THEME_PRESETS.get(key)
        if palette is None:
            palette = UI_THEME_PRESETS[UI_THEME_DEFAULT]
        return dict(palette)

    def _ui_theme_family_name(self, family_key: str) -> str:
        language = self._normalize_ui_language_key(getattr(self, "ui_language_key", UI_LANGUAGE_DEFAULT))
        names = UI_THEME_FAMILY_NAMES.get(family_key, UI_THEME_FAMILY_NAMES[UI_THEME_FAMILY_ORDER[0]])
        return names.get(language, names.get("pt", family_key))

    def _ui_theme_name(self, theme_key: str | None = None) -> str:
        key = self._normalize_ui_theme_key(theme_key or self.ui_theme_key)
        family = self._ui_theme_family_key(key)
        mode = self._ui_theme_mode_key(key)
        mode_label = self._t("theme.mode_dark" if mode == "dark" else "theme.mode_light")
        return f"{self._ui_theme_family_name(family)} {mode_label}"

    def _ui_theme_toggle_label(self) -> str:
        return self._t("theme.menu_label", theme=self._ui_theme_name())

    def _refresh_theme_toggle_buttons(self) -> None:
        label = self._ui_theme_toggle_label()
        for btn in (self.theme_toggle_btn_config, self.theme_toggle_btn):
            if btn is None:
                continue
            try:
                btn.configure(text=label)
            except Exception:
                pass
        self._sync_theme_settings_controls()

    def _theme_family_values(self) -> list[str]:
        return [self._ui_theme_family_name(key) for key in UI_THEME_FAMILY_ORDER]

    def _theme_mode_label(self, mode_key: str) -> str:
        return self._t("theme.mode_dark" if mode_key == "dark" else "theme.mode_light")

    def _theme_mode_values(self) -> list[str]:
        return [self._theme_mode_label(key) for key in UI_THEME_MODE_ORDER]

    def _theme_family_from_label(self, raw_value: object) -> str:
        text = _strip_accents(str(raw_value or "").strip().lower())
        for key in UI_THEME_FAMILY_ORDER:
            if text == _strip_accents(self._ui_theme_family_name(key).lower()):
                return key
        return self._ui_theme_family_key(self.ui_theme_key)

    def _theme_mode_from_label(self, raw_value: object) -> str:
        text = _strip_accents(str(raw_value or "").strip().lower())
        for key in UI_THEME_MODE_ORDER:
            if text == _strip_accents(self._theme_mode_label(key).lower()):
                return key
        return self._ui_theme_mode_key(self.ui_theme_key)

    def _sync_theme_settings_controls(self) -> None:
        key = self._normalize_ui_theme_key(self.ui_theme_key)
        family = self._ui_theme_family_key(key)
        mode = self._ui_theme_mode_key(key)
        self.theme_family_var.set(self._ui_theme_family_name(family))
        self.theme_mode_var.set(self._theme_mode_label(mode))
        if self.theme_family_combo is not None:
            try:
                self.theme_family_combo.configure(values=self._theme_family_values())
                self.theme_family_combo.set(self._ui_theme_family_name(family))
            except Exception:
                pass
        if self.theme_mode_combo is not None:
            try:
                self.theme_mode_combo.configure(values=self._theme_mode_values())
                self.theme_mode_combo.set(self._theme_mode_label(mode))
            except Exception:
                pass

    def _apply_theme_selection_from_controls(self) -> None:
        family = self._theme_family_from_label(self.theme_family_var.get())
        mode = self._theme_mode_from_label(self.theme_mode_var.get())
        self._set_ui_theme(f"{family}_{mode}", save=True, update_status=True)

    def _on_theme_settings_popup_close_requested(self) -> None:
        popup = self.theme_settings_popup
        self.theme_settings_popup = None
        self.theme_family_combo = None
        self.theme_mode_combo = None
        if popup is not None and popup.winfo_exists():
            try:
                popup.destroy()
            except Exception:
                pass

    def _open_theme_settings_popup(self) -> None:
        popup = self.theme_settings_popup
        if popup is not None and popup.winfo_exists():
            popup.deiconify()
            popup.lift()
            self._sync_theme_settings_controls()
            try:
                popup.focus_force()
            except Exception:
                pass
            return

        popup = tk.Toplevel(self.root)
        self.theme_settings_popup = popup
        popup.title(self._t("title.theme_settings"))
        popup.resizable(False, False)
        popup.transient(self.root)
        popup.protocol("WM_DELETE_WINDOW", self._on_theme_settings_popup_close_requested)

        frame = ttk.Frame(popup, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text=self._t("label.theme_pattern")).grid(row=0, column=0, sticky=tk.W)
        self.theme_family_combo = ttk.Combobox(
            frame,
            textvariable=self.theme_family_var,
            values=self._theme_family_values(),
            state="readonly",
            width=24,
        )
        self.theme_family_combo.grid(row=0, column=1, sticky=tk.W, padx=(8, 0))
        self.theme_family_combo.bind("<<ComboboxSelected>>", lambda _event: self._apply_theme_selection_from_controls())

        ttk.Label(frame, text=self._t("label.theme_mode")).grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.theme_mode_combo = ttk.Combobox(
            frame,
            textvariable=self.theme_mode_var,
            values=self._theme_mode_values(),
            state="readonly",
            width=24,
        )
        self.theme_mode_combo.grid(row=1, column=1, sticky=tk.W, padx=(8, 0), pady=(10, 0))
        self.theme_mode_combo.bind("<<ComboboxSelected>>", lambda _event: self._apply_theme_selection_from_controls())

        ttk.Button(
            frame,
            text=self._t("button.ok"),
            command=self._on_theme_settings_popup_close_requested,
        ).grid(row=2, column=1, sticky=tk.E, pady=(12, 0))
        frame.columnconfigure(1, weight=1)

        self._sync_theme_settings_controls()
        self._translate_widget_tree(popup)
        self._center_toplevel_on_root(popup)

    def _apply_ui_theme_styles(self) -> None:
        self.ui_theme_key = self._normalize_ui_theme_key(self.ui_theme_key)
        palette = self._theme_palette()

        app_bg = str(palette.get("app_bg", "#F3F5F8"))
        panel_bg = str(palette.get("panel_bg", "#FFFFFF"))
        fg = str(palette.get("fg", "#111827"))
        fg_disabled = str(palette.get("fg_disabled", "#8F97A4"))
        field_bg = str(palette.get("field_bg", "#FFFFFF"))
        field_bg_disabled = str(palette.get("field_bg_disabled", "#EEF2F7"))
        field_fg = str(palette.get("field_fg", "#111827"))
        button_bg = str(palette.get("button_bg", "#E8EEF7"))
        button_bg_active = str(palette.get("button_bg_active", "#DCE5F2"))
        button_bg_disabled = str(palette.get("button_bg_disabled", "#E8EEF7"))
        button_fg = str(palette.get("button_fg", fg))
        button_fg_disabled = str(palette.get("button_fg_disabled", fg_disabled))
        header_bg = str(palette.get("header_bg", "#E5EBF4"))
        border = str(palette.get("border", "#C5CEDA"))
        select_bg = str(palette.get("select_bg", "#2F6FED"))
        select_fg = str(palette.get("select_fg", "#FFFFFF"))
        text_bg = str(palette.get("text_bg", panel_bg))
        text_fg = str(palette.get("text_fg", fg))
        menu_bg = str(palette.get("menu_bg", panel_bg))
        menu_fg = str(palette.get("menu_fg", fg))
        progress_bg = str(palette.get("progress_bg", select_bg))
        progress_trough = str(palette.get("progress_trough", field_bg_disabled))

        style = self.style
        try:
            if "clam" in style.theme_names() and style.theme_use() != "clam":
                style.theme_use("clam")
        except Exception:
            pass

        try:
            self.root.configure(bg=app_bg)
            self.root.option_add("*Menu.background", menu_bg)
            self.root.option_add("*Menu.foreground", menu_fg)
            self.root.option_add("*Menu.activeBackground", select_bg)
            self.root.option_add("*Menu.activeForeground", select_fg)
        except Exception:
            pass

        for style_name, kwargs in (
            (".", {"background": app_bg, "foreground": fg}),
            ("TFrame", {"background": app_bg}),
            ("TLabel", {"background": app_bg, "foreground": fg}),
            ("TLabelframe", {"background": app_bg, "foreground": fg}),
            ("TLabelframe.Label", {"background": app_bg, "foreground": fg}),
            ("TButton", {"background": button_bg, "foreground": button_fg, "bordercolor": border}),
            ("TMenubutton", {"background": button_bg, "foreground": button_fg, "bordercolor": border}),
            ("TCheckbutton", {"background": app_bg, "foreground": fg}),
            ("TRadiobutton", {"background": app_bg, "foreground": fg}),
            ("TEntry", {"fieldbackground": field_bg, "foreground": field_fg}),
            ("TCombobox", {"fieldbackground": field_bg, "background": field_bg, "foreground": field_fg}),
            ("TNotebook", {"background": app_bg, "bordercolor": border}),
            ("TNotebook.Tab", {"background": panel_bg, "foreground": fg}),
            ("Treeview", {"background": panel_bg, "fieldbackground": panel_bg, "foreground": fg}),
            ("Treeview.Heading", {"background": header_bg, "foreground": fg}),
            ("Vertical.TScrollbar", {"background": panel_bg}),
            ("Horizontal.TScrollbar", {"background": panel_bg}),
            (
                "Horizontal.TProgressbar",
                {"background": progress_bg, "troughcolor": progress_trough, "bordercolor": border},
            ),
        ):
            try:
                style.configure(style_name, **kwargs)
            except Exception:
                pass

        for style_name, kwargs in (
            (
                "TButton",
                {
                    "background": [("disabled", button_bg_disabled), ("active", button_bg_active)],
                    "foreground": [("disabled", button_fg_disabled), ("active", button_fg)],
                },
            ),
            (
                "TMenubutton",
                {
                    "background": [("disabled", button_bg_disabled), ("active", button_bg_active)],
                    "foreground": [("disabled", button_fg_disabled), ("active", button_fg)],
                },
            ),
            ("TCheckbutton", {"foreground": [("disabled", fg_disabled)]}),
            (
                "TEntry",
                {
                    "fieldbackground": [("disabled", field_bg_disabled)],
                    "foreground": [("disabled", fg_disabled)],
                },
            ),
            (
                "TCombobox",
                {
                    "fieldbackground": [("readonly", field_bg), ("disabled", field_bg_disabled)],
                    "background": [("readonly", field_bg), ("disabled", field_bg_disabled)],
                    "foreground": [("readonly", field_fg), ("disabled", fg_disabled)],
                    "selectbackground": [("readonly", select_bg)],
                    "selectforeground": [("readonly", select_fg)],
                },
            ),
            ("Treeview", {"background": [("selected", select_bg)], "foreground": [("selected", select_fg)]}),
            (
                "TNotebook.Tab",
                {
                    "background": [("selected", button_bg_active), ("active", button_bg)],
                    "foreground": [("selected", button_fg), ("active", button_fg), ("disabled", fg_disabled)],
                },
            ),
        ):
            try:
                style.map(style_name, **kwargs)
            except Exception:
                pass

        for menu in (self.open_config_menu,):
            if menu is None:
                continue
            try:
                menu.configure(
                    bg=menu_bg,
                    fg=menu_fg,
                    activebackground=select_bg,
                    activeforeground=select_fg,
                )
            except Exception:
                pass

        for text_widget in (self.metrics_text, self.guide_popup_text, self.progress_log_text):
            if text_widget is None:
                continue
            try:
                if not text_widget.winfo_exists():
                    continue
                text_widget.configure(
                    bg=text_bg,
                    fg=text_fg,
                    insertbackground=text_fg,
                    selectbackground=select_bg,
                    selectforeground=select_fg,
                    highlightbackground=border,
                    highlightcolor=select_bg,
                )
            except Exception:
                pass

        for popup in (
            self.progress_popup,
            self.guide_popup,
            self.paths_mini_popup,
            self.group_navigation_popup,
            self.theme_settings_popup,
            self.contour_settings_popup,
        ):
            if popup is None:
                continue
            try:
                if popup.winfo_exists():
                    popup.configure(bg=app_bg)
            except Exception:
                pass

        self._refresh_theme_toggle_buttons()
        self.bg_last_size = (0, 0)
        self._render_background_image()

    def _set_ui_theme(self, raw_value: object, save: bool = False, update_status: bool = False) -> None:
        key = self._normalize_ui_theme_key(raw_value)
        self.ui_theme_key = key
        self._apply_ui_theme_styles()
        if update_status:
            mode_txt = self._ui_theme_name(key)
            self.status_var.set(self._t("status.theme_changed", mode=mode_txt))
        if save:
            self._save_user_settings()
        self._refresh_help_tab()

    def _toggle_ui_theme(self) -> None:
        current = self._normalize_ui_theme_key(self.ui_theme_key)
        family = self._ui_theme_family_key(current)
        mode = self._ui_theme_mode_key(current)
        target_mode = "dark" if mode == "light" else "light"
        target = f"{family}_{target_mode}"
        self._set_ui_theme(target, save=True, update_status=True)

    @staticmethod
    def _normalize_roi_editor_profile_key(raw_value: object) -> str:
        text = str(raw_value or "").strip().lower()
        if text in ROI_EDITOR_PROFILE_PRESETS:
            return text
        mapped = ROI_EDITOR_PROFILE_LABEL_TO_KEY.get(text)
        if mapped is not None:
            return mapped
        return ROI_EDITOR_PROFILE_DEFAULT

    def _roi_editor_profile_label(self, profile_key: str) -> str:
        key = self._normalize_roi_editor_profile_key(profile_key)
        if key == "rapido_50":
            return self._t("roi.profile.fast50")
        preset = ROI_EDITOR_PROFILE_PRESETS.get(key) or ROI_EDITOR_PROFILE_PRESETS[ROI_EDITOR_PROFILE_DEFAULT]
        return str(preset.get("label", self._t("roi.profile.fast50")))

    def _set_roi_editor_profile(self, raw_value: object, save: bool = False, update_status: bool = False) -> None:
        key = self._normalize_roi_editor_profile_key(raw_value)
        self.editor_profile_key = key
        label = self._roi_editor_profile_label(key)
        self.editor_profile_var.set(label)
        if self.editor_profile_combo is not None:
            self.editor_profile_combo.set(label)
        if update_status:
            self.status_var.set(f"{self._t('roi.profile')}: {label}")
        self._refresh_help_tab()
        if save:
            self._save_user_settings()

    @staticmethod
    def _normalize_preview_mode_key(raw_value: object) -> str:
        text = str(raw_value or "").strip().lower()
        if text in PREVIEW_MODE_PRESETS:
            return text
        mapped = PREVIEW_MODE_LABEL_TO_KEY.get(text)
        if mapped is not None:
            return mapped
        return PREVIEW_MODE_DEFAULT

    def _preview_mode_label(self, mode_key: str) -> str:
        key = self._normalize_preview_mode_key(mode_key)
        return self._t(f"preview.mode.{key}")

    def _preview_mode_values(self) -> list[str]:
        return [self._preview_mode_label("contour"), self._preview_mode_label("filled")]

    @staticmethod
    def _normalize_preview_source_mode_key(raw_value: object) -> str:
        text = str(raw_value or "").strip().lower()
        if text in PREVIEW_SOURCE_MODE_PRESETS:
            return text
        mapped = PREVIEW_SOURCE_MODE_LABEL_TO_KEY.get(text)
        if mapped is not None:
            return mapped
        return PREVIEW_SOURCE_MODE_DEFAULT

    def _preview_source_mode_label(self, mode_key: str) -> str:
        key = self._normalize_preview_source_mode_key(mode_key)
        return self._t(f"preview.source.{key}")

    def _preview_source_mode_values(self) -> list[str]:
        return [self._preview_source_mode_label("original"), self._preview_source_mode_label("enhanced")]

    @staticmethod
    def _normalize_preview_color_mode_key(raw_value: object) -> str:
        text = str(raw_value or "").strip().lower()
        if text in PREVIEW_COLOR_MODE_PRESETS:
            return text
        mapped = PREVIEW_COLOR_MODE_LABEL_TO_KEY.get(text)
        if mapped is not None:
            return mapped
        return PREVIEW_COLOR_MODE_DEFAULT

    def _preview_color_mode_label(self, mode_key: str) -> str:
        key = self._normalize_preview_color_mode_key(mode_key)
        return self._t(f"preview.color.{key}")

    def _preview_color_mode_values(self) -> list[str]:
        return [self._preview_color_mode_label("shared"), self._preview_color_mode_label("per_time")]

    @staticmethod
    def _normalize_rgb_triplet(raw_value: object) -> tuple[int, int, int] | None:
        if isinstance(raw_value, str):
            txt = raw_value.strip()
            if re.fullmatch(r"#[0-9A-Fa-f]{6}", txt):
                return (int(txt[1:3], 16), int(txt[3:5], 16), int(txt[5:7], 16))
            parts = [p for p in re.split(r"[,; ]+", txt) if p]
            if len(parts) >= 3:
                try:
                    vals = [int(round(float(parts[i]))) for i in range(3)]
                except Exception:
                    return None
                return tuple(int(np.clip(v, 0, 255)) for v in vals)
            return None

        if isinstance(raw_value, (list, tuple)) and len(raw_value) >= 3:
            try:
                vals = [int(round(float(raw_value[i]))) for i in range(3)]
            except Exception:
                return None
            return tuple(int(np.clip(v, 0, 255)) for v in vals)
        return None

    @staticmethod
    def _rgb_to_hex(color_rgb: tuple[int, int, int]) -> str:
        r, g, b = (int(np.clip(c, 0, 255)) for c in color_rgb)
        return f"#{r:02X}{g:02X}{b:02X}"

    @staticmethod
    def _color_text_fg(color_rgb: tuple[int, int, int]) -> str:
        r, g, b = (int(np.clip(c, 0, 255)) for c in color_rgb)
        luma = (0.299 * r) + (0.587 * g) + (0.114 * b)
        return "black" if luma >= 160.0 else "white"

    def _set_preview_mode(self, raw_value: object, save: bool = False, update_status: bool = False) -> None:
        key = self._normalize_preview_mode_key(raw_value)
        self.preview_mode_key = key
        label = self._preview_mode_label(key)
        self.preview_mode_var.set(label)
        if self.preview_mode_combo is not None:
            self.preview_mode_combo.configure(values=self._preview_mode_values())
            self.preview_mode_combo.set(label)
        if update_status:
            self.status_var.set(self._t("status.preview_mode_changed", label=label))
        if hasattr(self, "figure") and hasattr(self, "canvas"):
            self._refresh_figure(self.current_group_key)
        if save:
            self._save_user_settings()

    def _set_preview_source_mode(self, raw_value: object, save: bool = False, update_status: bool = False) -> None:
        key = self._normalize_preview_source_mode_key(raw_value)
        self.preview_source_mode_key = key
        label = self._preview_source_mode_label(key)
        self.preview_source_mode_var.set(label)
        if self.preview_source_mode_combo is not None:
            self.preview_source_mode_combo.configure(values=self._preview_source_mode_values())
            self.preview_source_mode_combo.set(label)
        if update_status:
            self.status_var.set(self._t("status.preview_source_changed", label=label))
        if hasattr(self, "figure") and hasattr(self, "canvas"):
            self._refresh_figure(self.current_group_key)
        if save:
            self._save_user_settings()

    def _set_preview_color_mode(self, raw_value: object, save: bool = False, update_status: bool = False) -> None:
        key = self._normalize_preview_color_mode_key(raw_value)
        self.preview_color_mode_key = key
        label = self._preview_color_mode_label(key)
        self.preview_color_mode_var.set(label)
        if self.preview_color_mode_combo is not None:
            self.preview_color_mode_combo.configure(values=self._preview_color_mode_values())
            self.preview_color_mode_combo.set(label)
        self._refresh_contour_color_controls_state()
        if update_status:
            self.status_var.set(self._t("status.preview_color_mode_changed", label=label))
        if hasattr(self, "figure") and hasattr(self, "canvas"):
            self._refresh_figure(self.current_group_key)
        if save:
            self._save_user_settings()

    def _set_compare_area_overlay(self, enabled: bool, save: bool = False, update_status: bool = False) -> None:
        value = bool(enabled)
        self.compare_area_overlay_var.set(value)
        if update_status:
            state_key = "state.enabled" if value else "state.disabled"
            self.status_var.set(self._t("status.compare_area_overlay_changed", state=self._t(state_key)))
        if hasattr(self, "figure") and hasattr(self, "canvas"):
            self._refresh_figure(self.current_group_key)
        if save:
            self._save_user_settings()

    def _on_compare_area_overlay_changed(self) -> None:
        self._set_compare_area_overlay(
            bool(self.compare_area_overlay_var.get()),
            save=True,
            update_status=True,
        )

    def _on_preview_mode_selected(self, _event=None) -> None:
        self._set_preview_mode(self.preview_mode_var.get(), save=True, update_status=True)

    def _on_preview_source_mode_selected(self, _event=None) -> None:
        self._set_preview_source_mode(self.preview_source_mode_var.get(), save=True, update_status=True)

    def _on_preview_color_mode_selected(self, _event=None) -> None:
        self._set_preview_color_mode(self.preview_color_mode_var.get(), save=True, update_status=True)

    def _preview_contour_color(self, time_tag: str) -> tuple[int, int, int]:
        color_mode = self._normalize_preview_color_mode_key(self.preview_color_mode_key)
        if color_mode != "per_time":
            return tuple(int(np.clip(c, 0, 255)) for c in self.preview_shared_contour_color)
        raw = self.preview_contour_colors.get(str(time_tag).lower())
        if raw is None:
            return tuple(int(np.clip(c, 0, 255)) for c in self.preview_shared_contour_color)
        return tuple(int(np.clip(c, 0, 255)) for c in raw)

    def _update_preview_color_buttons(self) -> None:
        if self.preview_shared_color_button is not None:
            shared_rgb = tuple(int(np.clip(c, 0, 255)) for c in self.preview_shared_contour_color)
            shared_hex = self._rgb_to_hex(shared_rgb)
            shared_fg = self._color_text_fg(shared_rgb)
            self.preview_shared_color_button.configure(
                text=f"{self._t('color.all')} {shared_hex}",
                bg=shared_hex,
                activebackground=shared_hex,
                fg=shared_fg,
                activeforeground=shared_fg,
                highlightbackground=shared_hex,
            )

        for time_tag, btn in self.preview_color_buttons.items():
            raw_rgb = self.preview_contour_colors.get(time_tag, self.preview_shared_contour_color)
            rgb = tuple(int(np.clip(c, 0, 255)) for c in raw_rgb)
            color_hex = self._rgb_to_hex(rgb)
            fg = self._color_text_fg(rgb)
            btn.configure(
                text=f"{time_tag.upper()} {color_hex}",
                bg=color_hex,
                activebackground=color_hex,
                fg=fg,
                activeforeground=fg,
                highlightbackground=color_hex,
            )

    def _refresh_contour_color_controls_state(self) -> None:
        color_mode = self._normalize_preview_color_mode_key(self.preview_color_mode_key)
        if self.preview_color_mode_combo is not None:
            self.preview_color_mode_combo.configure(state=("disabled" if self.ui_busy else "readonly"))
        if self.preview_source_mode_combo is not None:
            self.preview_source_mode_combo.configure(state=("disabled" if self.ui_busy else "readonly"))

        if self.preview_shared_color_label is not None and self.preview_shared_color_row is not None:
            if color_mode == "shared":
                self.preview_shared_color_label.grid()
                self.preview_shared_color_row.grid()
            else:
                self.preview_shared_color_label.grid_remove()
                self.preview_shared_color_row.grid_remove()

        if self.preview_per_time_color_label is not None and self.preview_per_time_color_row is not None:
            if color_mode == "per_time":
                self.preview_per_time_color_label.grid()
                self.preview_per_time_color_row.grid()
            else:
                self.preview_per_time_color_label.grid_remove()
                self.preview_per_time_color_row.grid_remove()

        shared_state = tk.NORMAL if ((not self.ui_busy) and color_mode == "shared") else tk.DISABLED
        if self.preview_shared_color_button is not None:
            self.preview_shared_color_button.configure(state=shared_state)

        per_time_state = tk.NORMAL if ((not self.ui_busy) and color_mode == "per_time") else tk.DISABLED
        for btn in self.preview_color_buttons.values():
            btn.configure(state=per_time_state)

        self._fit_contour_settings_popup(recenter=False)

    def _fit_contour_settings_popup(self, recenter: bool = False) -> None:
        popup = self.contour_settings_popup
        if popup is None or (not popup.winfo_exists()):
            return

        try:
            popup.update_idletasks()
        except Exception:
            return

        req_w = max(int(popup.winfo_reqwidth()), 420)
        req_h = int(popup.winfo_reqheight())

        if recenter:
            x = self.root.winfo_rootx() + max(0, (self.root.winfo_width() - req_w) // 2)
            y = self.root.winfo_rooty() + max(0, (self.root.winfo_height() - req_h) // 3)
        else:
            x = int(popup.winfo_x())
            y = int(popup.winfo_y())

        popup.geometry(f"{req_w}x{req_h}+{x}+{y}")

    def _choose_preview_shared_contour_color(self) -> None:
        current = tuple(int(np.clip(c, 0, 255)) for c in self.preview_shared_contour_color)
        popup = self.contour_settings_popup
        chooser_parent = popup if (popup is not None and popup.winfo_exists()) else self.root
        chosen = colorchooser.askcolor(
            color=self._rgb_to_hex(current),
            title=self._t("color.choose_all_title"),
            parent=chooser_parent,
        )
        if not chosen or chosen[0] is None:
            return
        rgb = tuple(int(np.clip(round(v), 0, 255)) for v in chosen[0])
        self.preview_shared_contour_color = rgb
        self._update_preview_color_buttons()
        self._save_user_settings()
        self._refresh_figure(self.current_group_key)
        self.status_var.set(self._t("status.shared_color_changed", color=self._rgb_to_hex(rgb)))

    def _choose_preview_contour_color(self, time_tag: str) -> None:
        tag = str(time_tag).lower()
        current = self._preview_contour_color(tag)
        popup = self.contour_settings_popup
        chooser_parent = popup if (popup is not None and popup.winfo_exists()) else self.root
        chosen = colorchooser.askcolor(
            color=self._rgb_to_hex(current),
            title=self._t("color.choose_time_title", time=tag.upper()),
            parent=chooser_parent,
        )
        if not chosen or chosen[0] is None:
            return
        rgb = tuple(int(np.clip(round(v), 0, 255)) for v in chosen[0])
        self.preview_contour_colors[tag] = rgb
        self._update_preview_color_buttons()
        self._save_user_settings()
        self._refresh_figure(self.current_group_key)
        self.status_var.set(self._t("status.time_color_changed", time=tag.upper(), color=self._rgb_to_hex(rgb)))

    def _on_contour_settings_popup_close_requested(self) -> None:
        popup = self.contour_settings_popup
        self.contour_settings_popup = None
        self.preview_mode_combo = None
        self.preview_source_mode_combo = None
        self.preview_color_mode_combo = None
        self.preview_shared_color_button = None
        self.preview_shared_color_label = None
        self.preview_shared_color_row = None
        self.preview_per_time_color_label = None
        self.preview_per_time_color_row = None
        self.preview_color_buttons = {}
        if popup is not None and popup.winfo_exists():
            try:
                popup.destroy()
            except Exception:
                pass

    def _close_paths_mini_tab(self) -> None:
        popup = self.paths_mini_popup
        self.paths_mini_popup = None
        if popup is not None and popup.winfo_exists():
            try:
                popup.destroy()
            except Exception:
                pass

    def _open_paths_mini_tab(self) -> None:
        """Abre mini janela com ajustes de pastas de entrada/saida."""
        popup = self.paths_mini_popup
        if popup is not None and popup.winfo_exists():
            popup.deiconify()
            popup.lift()
            try:
                popup.focus_force()
            except Exception:
                pass
            return

        popup = tk.Toplevel(self.root)
        self.paths_mini_popup = popup
        popup.title(self._t("title.quick_paths"))
        popup.transient(self.root)
        popup.resizable(False, False)
        popup.protocol("WM_DELETE_WINDOW", self._close_paths_mini_tab)

        frame = ttk.Frame(popup, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text=self._t("label.image_folder")).grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.folder_var, width=56).grid(row=0, column=1, sticky=tk.EW, padx=(8, 0))
        ttk.Button(
            frame,
            text=self._t("button.browse"),
            width=12,
            command=self._browse_folder,
        ).grid(row=0, column=2, padx=(8, 0))

        ttk.Label(frame, text=self._t("label.output_folder")).grid(row=1, column=0, sticky=tk.W, pady=(8, 0))
        ttk.Entry(frame, textvariable=self.output_folder_var, width=56).grid(
            row=1, column=1, sticky=tk.EW, padx=(8, 0), pady=(8, 0)
        )
        ttk.Button(
            frame,
            text=self._t("button.browse"),
            width=12,
            command=self._browse_output_folder,
        ).grid(row=1, column=2, padx=(8, 0), pady=(8, 0))

        actions = ttk.Frame(frame)
        actions.grid(row=2, column=0, columnspan=3, sticky=tk.E, pady=(12, 0))
        ttk.Button(actions, text=self._t("button.load_groups"), command=self._scan_folder).pack(side=tk.RIGHT)
        ttk.Button(actions, text=self._t("button.close"), command=self._close_paths_mini_tab).pack(side=tk.RIGHT, padx=(0, 8))

        frame.columnconfigure(1, weight=1)
        self._translate_widget_tree(popup)

    def _close_group_navigation_mini_tab(self) -> None:
        popup = self.group_navigation_popup
        self.group_navigation_popup = None
        if popup is not None and popup.winfo_exists():
            try:
                popup.destroy()
            except Exception:
                pass

    def _close_guide_popup(self) -> None:
        popup = self.guide_popup
        self.guide_popup = None
        self.guide_popup_text = None
        if popup is not None and popup.winfo_exists():
            try:
                popup.destroy()
            except Exception:
                pass

    def _open_guide_popup(self) -> None:
        popup = self.guide_popup
        if popup is not None and popup.winfo_exists():
            popup.deiconify()
            popup.lift()
            try:
                popup.focus_force()
            except Exception:
                pass
            self._refresh_help_tab()
            return

        popup = tk.Toplevel(self.root)
        self.guide_popup = popup
        popup.title(self._t("title.guide"))
        popup.transient(self.root)
        popup.resizable(True, True)
        popup.protocol("WM_DELETE_WINDOW", self._close_guide_popup)

        frame = ttk.Frame(popup, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        body = ttk.Frame(frame)
        body.pack(fill=tk.BOTH, expand=True)
        text = tk.Text(body, width=90, height=28, wrap=tk.WORD, state=tk.DISABLED)
        yscroll = ttk.Scrollbar(body, orient=tk.VERTICAL, command=text.yview)
        text.configure(yscrollcommand=yscroll.set)
        text.grid(row=0, column=0, sticky=tk.NSEW)
        yscroll.grid(row=0, column=1, sticky=tk.NS)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)

        ttk.Button(frame, text=self._t("button.close"), command=self._close_guide_popup).pack(anchor=tk.E, pady=(10, 0))

        self.guide_popup_text = text
        self._refresh_help_tab()
        self._translate_widget_tree(popup)

        try:
            self.root.update_idletasks()
            popup.update_idletasks()
            width = max(700, int(popup.winfo_reqwidth()))
            height = max(520, int(popup.winfo_reqheight()))
            x = self.root.winfo_rootx() + max(0, (self.root.winfo_width() - width) // 2)
            y = self.root.winfo_rooty() + max(0, (self.root.winfo_height() - height) // 3)
            popup.geometry(f"{width}x{height}+{x}+{y}")
        except Exception:
            pass

    def _clear_tooltip_schedule(self) -> None:
        after_id = self._tooltip_after_id
        if after_id is None:
            return
        self._tooltip_after_id = None
        try:
            self.root.after_cancel(after_id)
        except Exception:
            pass

    def _hide_tooltip(self, _event=None) -> None:
        self._clear_tooltip_schedule()
        popup = self._tooltip_popup
        self._tooltip_popup = None
        if popup is not None and popup.winfo_exists():
            try:
                popup.destroy()
            except Exception:
                pass

    def _show_tooltip_for_widget(self, widget: tk.Widget) -> None:
        try:
            widget_exists = widget.winfo_exists()
        except Exception:
            widget_exists = False
        if not widget_exists:
            return
        key = str(widget)
        text = str(self._tooltip_text_by_widget.get(key, "")).strip()
        if not text:
            return

        self._hide_tooltip()

        popup = tk.Toplevel(self.root)
        self._tooltip_popup = popup
        popup.overrideredirect(True)
        popup.transient(self.root)
        try:
            popup.attributes("-topmost", True)
        except Exception:
            pass

        palette = self._theme_palette()
        label = tk.Label(
            popup,
            text=text,
            justify=tk.LEFT,
            anchor=tk.W,
            relief=tk.SOLID,
            borderwidth=1,
            padx=7,
            pady=5,
            background=str(palette.get("tooltip_bg", "#FFF9D7")),
            foreground=str(palette.get("tooltip_fg", "#111111")),
            wraplength=420,
        )
        label.pack(fill=tk.BOTH, expand=True)

        try:
            popup.update_idletasks()
            pointer_x = int(widget.winfo_pointerx())
            pointer_y = int(widget.winfo_pointery())
            tip_w = int(popup.winfo_reqwidth())
            tip_h = int(popup.winfo_reqheight())
            screen_w = int(widget.winfo_screenwidth())
            screen_h = int(widget.winfo_screenheight())

            x = pointer_x + 14
            y = pointer_y + 18
            if (x + tip_w) > (screen_w - 8):
                x = max(8, screen_w - tip_w - 8)
            if (y + tip_h) > (screen_h - 8):
                y = max(8, pointer_y - tip_h - 14)
            popup.geometry(f"+{x}+{y}")
        except Exception:
            pass

    def _schedule_tooltip_for_widget(self, widget: tk.Widget) -> None:
        self._hide_tooltip()
        try:
            widget_exists = widget.winfo_exists()
        except Exception:
            widget_exists = False
        if not widget_exists:
            return
        key = str(widget)
        text = str(self._tooltip_text_by_widget.get(key, "")).strip()
        if not text:
            return

        def _show() -> None:
            self._tooltip_after_id = None
            self._show_tooltip_for_widget(widget)

        self._tooltip_after_id = self.root.after(420, _show)

    def _register_tooltip(self, widget: tk.Widget | None, text: str) -> None:
        if widget is None:
            return
        key = str(widget)
        self._tooltip_text_by_widget[key] = str(text).strip()
        if key in self._tooltip_bound_widgets:
            return
        widget.bind("<Enter>", lambda _event, w=widget: self._schedule_tooltip_for_widget(w), add="+")
        widget.bind("<Leave>", self._hide_tooltip, add="+")
        widget.bind("<ButtonPress>", self._hide_tooltip, add="+")
        widget.bind("<Destroy>", self._hide_tooltip, add="+")
        self._tooltip_bound_widgets.add(key)

    def _open_group_navigation_mini_tab(self) -> None:
        """Abre mini janela com navegacao de grupos e acesso rapido a edicao completa."""
        popup = self.group_navigation_popup
        if popup is not None and popup.winfo_exists():
            popup.deiconify()
            popup.lift()
            try:
                popup.focus_force()
            except Exception:
                pass
            return

        popup = tk.Toplevel(self.root)
        self.group_navigation_popup = popup
        popup.title(self._t("title.quick_groups"))
        popup.transient(self.root)
        popup.resizable(False, False)
        popup.protocol("WM_DELETE_WINDOW", self._close_group_navigation_mini_tab)

        frame = ttk.Frame(popup, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text=self._t("label.group_in_review")).grid(row=0, column=0, sticky=tk.W)
        values = []
        try:
            values = list(self.group_combo.cget("values"))
        except Exception:
            values = []
        state = "readonly" if values else "disabled"
        combo = ttk.Combobox(frame, textvariable=self.group_var, values=values, state=state, width=50)
        combo.grid(row=0, column=1, columnspan=2, sticky=tk.EW, padx=(8, 0))
        combo.bind("<<ComboboxSelected>>", self._on_group_selected)

        nav = ttk.Frame(frame)
        nav.grid(row=1, column=0, columnspan=3, sticky=tk.EW, pady=(10, 0))
        ttk.Button(nav, text=self._t("button.prev_group"), command=self._show_prev_group).pack(side=tk.LEFT)
        ttk.Button(nav, text=self._t("button.next_group"), command=self._show_next_group).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Button(
            frame,
            text=self._t("menu.open_full_settings"),
            command=lambda: (self._close_group_navigation_mini_tab(), self._show_config_page()),
        ).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(12, 0))
        ttk.Button(frame, text=self._t("button.close"), command=self._close_group_navigation_mini_tab).grid(
            row=2, column=2, sticky=tk.E, pady=(12, 0)
        )

        frame.columnconfigure(1, weight=1)
        self._translate_widget_tree(popup)

    def _open_contour_settings_popup(self) -> None:
        popup = self.contour_settings_popup
        if popup is not None and popup.winfo_exists():
            popup.deiconify()
            popup.lift()
            self._fit_contour_settings_popup(recenter=False)
            try:
                popup.focus_force()
            except Exception:
                pass
            return

        popup = tk.Toplevel(self.root)
        self.contour_settings_popup = popup
        popup.title(self._t("title.contour_settings"))
        popup.resizable(False, False)
        popup.transient(self.root)
        popup.protocol("WM_DELETE_WINDOW", self._on_contour_settings_popup_close_requested)

        frame = ttk.Frame(popup, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text=self._t("label.initial_preview")).grid(row=0, column=0, sticky=tk.W)
        preview_values = self._preview_mode_values()
        self.preview_mode_combo = ttk.Combobox(
            frame,
            textvariable=self.preview_mode_var,
            state=("disabled" if self.ui_busy else "readonly"),
            values=preview_values,
            width=22,
        )
        self.preview_mode_combo.grid(row=0, column=1, sticky=tk.W, padx=(8, 0))
        self.preview_mode_combo.bind("<<ComboboxSelected>>", self._on_preview_mode_selected)
        self.preview_mode_combo.set(self._preview_mode_label(self.preview_mode_key))

        ttk.Label(frame, text=self._t("label.background_image")).grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        source_mode_values = self._preview_source_mode_values()
        self.preview_source_mode_combo = ttk.Combobox(
            frame,
            textvariable=self.preview_source_mode_var,
            state=("disabled" if self.ui_busy else "readonly"),
            values=source_mode_values,
            width=28,
        )
        self.preview_source_mode_combo.grid(row=1, column=1, sticky=tk.W, padx=(8, 0), pady=(10, 0))
        self.preview_source_mode_combo.bind("<<ComboboxSelected>>", self._on_preview_source_mode_selected)
        self.preview_source_mode_combo.set(self._preview_source_mode_label(self.preview_source_mode_key))

        ttk.Label(frame, text=self._t("label.color_mode")).grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        color_mode_values = self._preview_color_mode_values()
        self.preview_color_mode_combo = ttk.Combobox(
            frame,
            textvariable=self.preview_color_mode_var,
            state=("disabled" if self.ui_busy else "readonly"),
            values=color_mode_values,
            width=22,
        )
        self.preview_color_mode_combo.grid(row=2, column=1, sticky=tk.W, padx=(8, 0), pady=(10, 0))
        self.preview_color_mode_combo.bind("<<ComboboxSelected>>", self._on_preview_color_mode_selected)
        self.preview_color_mode_combo.set(self._preview_color_mode_label(self.preview_color_mode_key))

        self.preview_shared_color_label = ttk.Label(frame, text=self._t("label.single_color"))
        self.preview_shared_color_label.grid(row=3, column=0, sticky=tk.W, pady=(10, 0))
        self.preview_shared_color_row = ttk.Frame(frame)
        self.preview_shared_color_row.grid(row=3, column=1, sticky=tk.W, padx=(8, 0), pady=(10, 0))
        self.preview_shared_color_button = tk.Button(
            self.preview_shared_color_row,
            text=self._t("color.all"),
            width=16,
            command=self._choose_preview_shared_contour_color,
            relief=tk.RAISED,
            bd=1,
            padx=4,
            pady=2,
        )
        self.preview_shared_color_button.grid(row=0, column=0, sticky=tk.W)

        self.preview_per_time_color_label = ttk.Label(frame, text=self._t("label.colors_by_type"))
        self.preview_per_time_color_label.grid(row=4, column=0, sticky=tk.W, pady=(10, 0))
        self.preview_per_time_color_row = ttk.Frame(frame)
        self.preview_per_time_color_row.grid(row=4, column=1, sticky=tk.W, padx=(8, 0), pady=(10, 0))
        self.preview_color_buttons = {}
        for idx, time_tag in enumerate(TIME_ORDER):
            btn = tk.Button(
                self.preview_per_time_color_row,
                text=time_tag.upper(),
                width=16,
                command=lambda tag=time_tag: self._choose_preview_contour_color(tag),
                relief=tk.RAISED,
                bd=1,
                padx=4,
                pady=2,
            )
            btn.grid(row=idx, column=0, sticky=tk.W, pady=(0 if idx == 0 else 4, 0))
            self.preview_color_buttons[time_tag] = btn
        self._update_preview_color_buttons()
        self._refresh_contour_color_controls_state()

        ttk.Button(
            frame,
            text=self._t("button.ok"),
            command=self._on_contour_settings_popup_close_requested,
        ).grid(row=5, column=1, sticky=tk.E, pady=(12, 0))
        frame.columnconfigure(1, weight=1)
        self._translate_widget_tree(popup)

        self._fit_contour_settings_popup(recenter=True)

    @staticmethod
    def _image_cache_key(path: Path) -> str:
        try:
            return str(path.resolve()).lower()
        except Exception:
            return str(path).lower()

    @staticmethod
    def _to_rgb_u8(image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim == 2:
            gray = arr if arr.dtype == np.uint8 else to_uint8_01(mat2gray(arr))
            return cv2.cvtColor(np.asarray(gray, dtype=np.uint8), cv2.COLOR_GRAY2RGB)

        if arr.ndim == 3:
            rgb = np.asarray(arr[:, :, :3])
            if rgb.dtype != np.uint8:
                rgb = to_uint8_01(rgb)
            return np.asarray(rgb, dtype=np.uint8)

        raise ValueError("Imagem invalida para conversao RGB.")

    @staticmethod
    def _visibility_score(image_rgb_u8: np.ndarray) -> float:
        rgb = np.asarray(image_rgb_u8, dtype=np.uint8)
        try:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        except Exception:
            gray = np.mean(rgb.astype(np.float32), axis=2).astype(np.float32)
        contrast = float(np.std(gray))
        edge_var = float(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F).var())
        return contrast + (0.15 * float(np.sqrt(max(edge_var, 0.0))))

    def _best_defined_image(self, image_rgb_u8: np.ndarray, cache_key: str | None = None) -> np.ndarray:
        key = str(cache_key or "")
        if key and key in self.preview_enhanced_image_cache:
            return np.asarray(self.preview_enhanced_image_cache[key], dtype=np.uint8)

        base_rgb = self._to_rgb_u8(image_rgb_u8)
        enhanced_rgb = self._to_rgb_u8(self._build_roi_editor_enhanced_image(base_rgb))
        best_rgb = enhanced_rgb if self._visibility_score(enhanced_rgb) >= self._visibility_score(base_rgb) else base_rgb
        if key:
            self.preview_enhanced_image_cache[key] = np.asarray(best_rgb, dtype=np.uint8)
        return np.asarray(best_rgb, dtype=np.uint8)

    def _read_original_preview_image(self, path: Path) -> np.ndarray | None:
        key = self._image_cache_key(path)
        cached = self.preview_original_image_cache.get(key)
        if cached is not None:
            return np.asarray(cached, dtype=np.uint8)
        try:
            src_img = read_image(str(path))
            rgb = self._to_rgb_u8(src_img)
            self.preview_original_image_cache[key] = np.asarray(rgb, dtype=np.uint8)
            return np.asarray(rgb, dtype=np.uint8)
        except Exception:
            return None

    def _preview_base_for_path(self, path: Path) -> np.ndarray | None:
        original_rgb = self._read_original_preview_image(path)
        if original_rgb is None:
            return None
        source_mode = self._normalize_preview_source_mode_key(self.preview_source_mode_key)
        if source_mode == "original":
            return np.asarray(original_rgb, dtype=np.uint8)
        return self._best_defined_image(original_rgb, cache_key=self._image_cache_key(path))

    def _preview_base_for_processed_item(self, proc: ProcessedTimepoint) -> np.ndarray:
        preview_rgb = self._preview_base_for_path(proc.path)
        if preview_rgb is not None:
            return np.asarray(preview_rgb, dtype=np.uint8)

        source_mode = self._normalize_preview_source_mode_key(self.preview_source_mode_key)
        fallback_base = self._to_rgb_u8(np.asarray(proc.artifacts.base_rgb_u8, dtype=np.uint8))
        if source_mode == "original":
            return np.asarray(fallback_base, dtype=np.uint8)
        cache_key = f"processed::{self._image_cache_key(proc.path)}"
        return self._best_defined_image(fallback_base, cache_key=cache_key)

    def _build_filled_overlay_with_selected_color(
        self,
        proc: ProcessedTimepoint,
        time_tag: str,
        base_rgb_u8: np.ndarray | None = None,
    ) -> np.ndarray:
        if base_rgb_u8 is None:
            try:
                base_rgb_u8 = self._preview_base_for_processed_item(proc)
            except Exception:
                base_rgb_u8 = np.asarray(proc.artifacts.base_rgb_u8, dtype=np.uint8)
        target_shape = tuple(np.asarray(base_rgb_u8).shape[:2])
        mask_auto = np.asarray(proc.artifacts.mask_auto).astype(bool)
        if mask_auto.shape != target_shape:
            mask_auto = resize_mask(mask_auto, target_shape)
        contour_mask = np.asarray(proc.artifacts.contour_mask).astype(bool)
        if contour_mask.shape != target_shape:
            contour_mask = contour_mask_with_thickness(mask_auto, radius=self.config.r_auto)
        color_rgb = self._preview_contour_color(time_tag)
        filled = overlay_mask_alpha(np.asarray(base_rgb_u8, dtype=np.uint8), mask_auto, color_rgb, alpha=0.35)
        return overlay_perimeter(filled, contour_mask, color_rgb)

    def _compare_area_overlay_enabled(self) -> bool:
        try:
            return bool(self.compare_area_overlay_var.get())
        except Exception:
            return False

    def _compare_area_masks(
        self,
        proc: ProcessedTimepoint,
        group_key: str,
        time_tag: str,
        target_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if not self._compare_area_overlay_enabled():
            return None

        key = (group_key, time_tag)
        edited_mask_raw = self.roi_by_item.get(key)
        if edited_mask_raw is None:
            return None

        generated_mask_raw = self.processing_mask_by_item.get(key)
        if generated_mask_raw is None:
            generated_mask_raw = getattr(proc.artifacts, "mask_auto", None)
        if generated_mask_raw is None:
            return None

        try:
            generated_mask = np.asarray(generated_mask_raw).astype(bool)
            edited_mask = np.asarray(edited_mask_raw).astype(bool)
        except Exception:
            return None

        if generated_mask.ndim != 2 or edited_mask.ndim != 2:
            return None
        if generated_mask.shape != target_shape:
            generated_mask = resize_mask(generated_mask, target_shape)
        if edited_mask.shape != target_shape:
            edited_mask = resize_mask(edited_mask, target_shape)
        if (not np.any(generated_mask)) and (not np.any(edited_mask)):
            return None
        return generated_mask, edited_mask

    def _build_compare_area_overlay(
        self,
        base_rgb_u8: np.ndarray,
        generated_mask: np.ndarray,
        edited_mask: np.ndarray,
    ) -> np.ndarray:
        base_rgb = np.asarray(base_rgb_u8, dtype=np.uint8)
        generated = np.asarray(generated_mask).astype(bool)
        edited = np.asarray(edited_mask).astype(bool)

        out = overlay_mask_alpha(base_rgb, generated, PREVIEW_COMPARE_PROCESSING_COLOR_RGB, alpha=0.24)
        out = overlay_mask_alpha(out, edited, PREVIEW_COMPARE_EDITED_COLOR_RGB, alpha=0.34)
        radius = max(1, int(self.config.r_auto))
        generated_contour = contour_mask_with_thickness(generated, radius=radius)
        edited_contour = contour_mask_with_thickness(edited, radius=radius)
        out = overlay_perimeter(out, generated_contour, PREVIEW_COMPARE_PROCESSING_COLOR_RGB)
        out = overlay_perimeter(out, edited_contour, PREVIEW_COMPARE_EDITED_COLOR_RGB)
        return out

    def _draw_compare_area_legend(self, ax) -> None:
        legend_specs = (
            (self._t("figure.processing_area"), PREVIEW_COMPARE_PROCESSING_COLOR_RGB, 0.97),
            (self._t("figure.edited_area"), PREVIEW_COMPARE_EDITED_COLOR_RGB, 0.89),
        )
        for label, color_rgb, y_pos in legend_specs:
            ax.text(
                0.03,
                y_pos,
                label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color=self._rgb_to_hex(color_rgb),
                bbox={"facecolor": "black", "alpha": 0.62, "pad": 3, "edgecolor": "none"},
            )

    def _build_preview_image(self, proc: ProcessedTimepoint, group_key: str, time_tag: str) -> np.ndarray:
        try:
            base_rgb = self._preview_base_for_processed_item(proc)
            target_shape = tuple(np.asarray(base_rgb).shape[:2])
            compare_masks = self._compare_area_masks(proc, group_key, time_tag, target_shape)
            if compare_masks is not None:
                generated_mask, edited_mask = compare_masks
                return self._build_compare_area_overlay(base_rgb, generated_mask, edited_mask)

            contour_mask = np.asarray(proc.artifacts.contour_mask).astype(bool)
            if contour_mask.shape != target_shape:
                mask_auto = np.asarray(proc.artifacts.mask_auto).astype(bool)
                if mask_auto.shape != target_shape:
                    mask_auto = resize_mask(mask_auto, target_shape)
                contour_mask = contour_mask_with_thickness(mask_auto, radius=self.config.r_auto)
            color_rgb = self._preview_contour_color(time_tag)
            mode_key = self._normalize_preview_mode_key(self.preview_mode_key)
            if mode_key == "filled":
                return self._build_filled_overlay_with_selected_color(proc, time_tag, base_rgb)
            return overlay_perimeter(np.asarray(base_rgb, dtype=np.uint8), contour_mask, color_rgb)
        except Exception:
            mode_key = self._normalize_preview_mode_key(self.preview_mode_key)
            if mode_key == "filled":
                return np.asarray(proc.artifacts.mask_overlay_rgb_u8, dtype=np.uint8)
            return np.asarray(proc.artifacts.contour_overlay_rgb_u8, dtype=np.uint8)

    def _on_close(self) -> None:
        if self._shutdown_in_progress:
            return
        if self.ui_busy and (self.active_processing_mode is not None):
            confirm = messagebox.askyesno(
                self._t("processing.running_title"),
                self._t("processing.close_confirm"),
            )
            if not confirm:
                return
        self._shutdown_in_progress = True
        self.cancel_event.set()
        # PT: Watchdog de seguranca: evita processo preso por loops nativos/threads externos. | EN: Safety watchdog that prevents the process from hanging in native loops or external threads.
        self._arm_force_exit_watchdog(timeout_seconds=3.0)

        try:
            if self.bg_resize_after_id is not None:
                self.root.after_cancel(self.bg_resize_after_id)
                self.bg_resize_after_id = None
        except Exception:
            pass

        try:
            self._close_progress_popup()
        except Exception:
            pass

        self._close_paths_mini_tab()
        self._close_group_navigation_mini_tab()
        self._on_theme_settings_popup_close_requested()
        self._on_contour_settings_popup_close_requested()
        self._close_guide_popup()
        self._hide_tooltip()
        self._save_user_settings()

        _close_all_pyplot_figures()

        try:
            self.root.quit()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def _arm_force_exit_watchdog(self, timeout_seconds: float = 3.0) -> None:
        if self._force_exit_timer is not None:
            return
        force_exit_raw = str(os.environ.get("CORA_FORCE_EXIT_ON_CLOSE", "1")).strip().lower()
        if force_exit_raw in {"0", "false", "no", "off"}:
            return

        def _force_exit() -> None:
            os._exit(0)

        timer = threading.Timer(max(0.5, float(timeout_seconds)), _force_exit)
        timer.daemon = True
        self._force_exit_timer = timer
        timer.start()

    def _build_layout(self) -> None:
        build_main_layout(self, time_order=TIME_ORDER)
        build_test_selection_layout(self)

    def _start_page_timing(self, first_page_name: str) -> None:
        """Inicia a medicao da sessao quando a primeira pagina ja esta visivel."""
        now = time.perf_counter()
        self._page_time_started_at = datetime.now().astimezone()
        self._active_page_name = first_page_name
        self._active_page_started_at = now
        self._page_visit_counts[first_page_name] += 1

    def _track_page_transition(self, page_name: str) -> None:
        """Acumula o tempo da pagina anterior e inicia a medicao da nova pagina."""
        if self._page_time_started_at is None or page_name == self._active_page_name:
            return

        now = time.perf_counter()
        if self._active_page_name is not None and self._active_page_started_at is not None:
            elapsed = max(0.0, now - self._active_page_started_at)
            self._page_time_totals[self._active_page_name] += elapsed

        self._active_page_name = page_name
        self._active_page_started_at = now
        self._page_visit_counts[page_name] += 1

    def _page_time_snapshot(self) -> tuple[datetime, list[dict[str, object]]]:
        """Retorna uma copia dos tempos incluindo a pagina ativa ate este instante."""
        measured_at = datetime.now().astimezone()
        totals = dict(self._page_time_totals)
        now = time.perf_counter()
        if self._active_page_name is not None and self._active_page_started_at is not None:
            totals[self._active_page_name] += max(0.0, now - self._active_page_started_at)

        rows = [
            {
                "pagina": page_name,
                "numero_de_visitas": self._page_visit_counts[page_name],
                "tempo_total_segundos": f"{totals[page_name]:.3f}",
            }
            for page_name in self._page_time_totals
        ]
        return measured_at, rows

    def _write_page_times_csv(
        self,
        out_dir: Path,
        measured_at: datetime,
        rows: list[dict[str, object]],
    ) -> Path:
        """Grava, em CSV separado, os tempos acumulados em cada pagina."""
        csv_path = out_dir / "tempos_por_pagina.csv"
        headers = [
            "pagina",
            "numero_de_visitas",
            "tempo_total_segundos",
            "inicio_da_sessao",
            "fim_da_medicao",
        ]
        started_at = self._page_time_started_at or measured_at
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers, delimiter=";")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        **row,
                        "inicio_da_sessao": started_at.isoformat(timespec="milliseconds"),
                        "fim_da_medicao": measured_at.isoformat(timespec="milliseconds"),
                    }
                )
        return csv_path

    def _show_test_selection_page(self) -> None:
        """Volta para a tela inicial de escolha do modo de execucao."""
        if self.ui_busy:
            return
        for page in (self.config_page, self.viewer_page):
            if page is not None and page.winfo_manager():
                page.pack_forget()
        if self.test_selection_page is not None and not self.test_selection_page.winfo_manager():
            self.test_selection_page.pack(fill=tk.BOTH, expand=True)
        self._track_page_transition("selecao_de_teste")
        self.status_var.set(self._t("status.test_modes_opened"))

    def _open_robot_test_shortcut(self, _event=None) -> str:
        """Abre as configuracoes pelo atalho reservado ao teste robotizado."""
        self._robotized_test_active = True
        self._show_config_page()
        return "break"

    def _open_normal_batch_page(self) -> None:
        self._robotized_test_active = False
        self._show_config_page()

    def _focus_robot_field(self, field_name: str) -> str:
        """Foca um campo de configuracao sem depender da acessibilidade do Tk."""
        widget = self.folder_entry if field_name == "images" else self.output_folder_entry
        if widget is not None and widget.winfo_exists():
            widget.focus_force()
            widget.icursor(tk.END)
        return "break"

    def _set_robot_field_from_clipboard(self, field_name: str) -> str:
        """Transfere e confirma um caminho enviado pelo teste robotizado."""
        try:
            value = str(self.root.clipboard_get()).strip()
        except Exception:
            value = ""

        target = Path(value) if value else None
        valid = bool(target is not None and target.is_dir())
        if field_name == "images":
            if valid:
                self.folder_var.set(value)
                self._robot_paths_confirmed.add("images")
            marker = "ROBOT_IMAGES_OK" if valid else "ROBOT_IMAGES_ERROR"
        else:
            if valid:
                self.output_folder_var.set(value)
                self._robot_paths_confirmed.add("output")
            marker = "ROBOT_PATHS_OK" if valid else "ROBOT_OUTPUT_ERROR"

        self.root.title(f"{self._t('app.title')} [{marker}]")
        self.root.update_idletasks()
        return "break"

    def _robot_load_groups_shortcut(self, _event=None) -> str:
        if self._robot_paths_confirmed != {"images", "output"}:
            self.root.title(f"{self._t('app.title')} [ROBOT_PATHS_REQUIRED]")
            return "break"
        self.root.title(self._t("app.title"))
        if not self.ui_busy and self.scan_btn is not None:
            self.scan_btn.invoke()
        return "break"

    def _run_robot_review_action(self, action_name: str) -> str:
        callback = self._robot_review_actions.get(action_name)
        if callback is not None:
            callback()
        return "break"

    def _robot_select_next_save_item(self, _event=None) -> str:
        """Inclui uma imagem sem redesenhar duas vezes o mosaico do grupo."""
        items = [
            (group_key, time_tag)
            for group_key in self.group_order
            for time_tag in TIME_ORDER
            if time_tag in self.processed_by_group.get(group_key, {})
        ]
        index = int(self._robot_save_selection_index)
        if index >= len(items):
            return "break"
        group_key, time_tag = items[index]
        self._robot_save_selection_index = index + 1
        self.pending_reprocess.discard((group_key, time_tag))
        self.reviewed_groups.add(group_key)
        self.reviewed_items.add((group_key, time_tag))

        if self.current_group_key != group_key:
            # PT: A troca de grupo exige um novo mosaico, mas apenas uma vez. | EN: Changing groups requires a new mosaic, but only once.
            self._show_group(group_key, mark_reviewed=True)
        else:
            # PT: No mesmo grupo, a imagem e os overlays ja estao desenhados. Atualizar somente o checkbox e o painel textual evita limpar e reconstruir toda a figura Matplotlib a cada F12.
            # EN: Within the same group, the image and overlays are already drawn. Updating only the checkbox and text panel avoids clearing and rebuilding the entire Matplotlib figure on every F12 press.
            self.refazer_vars[time_tag].set(True)
            self._refresh_metrics(group_key)
        return "break"

    def _robot_next_group_shortcut(self, _event=None) -> str:
        """Avanca um grupo e publica a posicao para confirmacao do Teste 2."""
        total = len(self.group_order)
        if total <= 0:
            self.root.title(f"{self._t('app.title')} [ROBOT_GROUP:0/0]")
            return "break"

        index = self._current_group_index()
        if index < 0:
            self._show_group_by_index(0, mark_reviewed=True)
        elif index < total - 1:
            self._show_group_by_index(index + 1, mark_reviewed=True)

        position = max(1, self._current_group_index() + 1)
        self.root.title(f"{self._t('app.title')} [ROBOT_GROUP:{position}/{total}]")
        self.root.update_idletasks()
        return "break"

    def _robot_edit_masks_shortcut(self, _event=None) -> str:
        """Aciona o editor sem depender da arvore de acessibilidade do Tk."""
        self.root.title(self._t("app.title"))
        if not self.ui_busy and self.redefine_roi_btn is not None:
            requested = len(self.pending_reprocess)
            self.redefine_roi_btn.invoke()
            applied = max(0, requested - len(self.pending_reprocess))
            self.root.title(
                f"{self._t('app.title')} [ROBOT_EDIT_DONE:{applied}/{requested}]"
            )
            self.root.update_idletasks()
        return "break"

    def _robot_save_results_shortcut(self, _event=None) -> str:
        if not self.ui_busy and self.save_btn is not None:
            self.save_btn.invoke()
        return "break"

    def _choose_single_image_test(self) -> None:
        """Seleciona, prepara e processa uma unica imagem no fluxo real do app."""
        self._robotized_test_active = False
        if self.ui_busy:
            return
        selected = filedialog.askopenfilename(
            title=self._t("test.select_image_title"),
            filetypes=(
                ("Imagens suportadas", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"),
                ("Todos os arquivos", "*.*"),
            ),
        )
        if not selected:
            return

        image_path = Path(selected)
        if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_EXTS:
            messagebox.showerror(
                self._t("dialog.error"),
                f"Arquivo de imagem invalido ou formato nao suportado:\n{image_path}",
            )
            return

        # PT: A pasta e atualizada antes de montar o grupo, pois o trace desse campo limpa resultados previamente carregados.
        # EN: The folder is updated before building the group because this field's trace clears previously loaded results.
        self.folder_var.set(str(image_path.parent))
        self._reset_all_state()

        group_key = "teste_imagem_unica"
        label = f"{image_path.stem} | imagem unica"
        self.group_files = {group_key: {"0h": image_path}}
        self.group_order = [group_key]
        self.group_labels = {group_key: label}
        self.label_to_group = {label: group_key}
        self.group_combo["values"] = [label]
        self.group_var.set(label)
        self.current_group_key = None
        self._processing_metrics_by_item = {}
        self._single_test_runs = []
        self._single_test_context = {
            "path": image_path,
            "group_key": group_key,
            "time_tag": "0h",
            "repetitions": SINGLE_IMAGE_TEST_REPETITIONS,
            "started_at": datetime.now().astimezone(),
        }

        self._set_ui_busy(False)
        self._show_group(group_key, mark_reviewed=False)
        self._show_viewer_page()
        self._start_processing_job(
            mode="single_test",
            items=[(group_key, "0h", image_path)] * SINGLE_IMAGE_TEST_REPETITIONS,
            save_after=False,
        )

    def _choose_batch_performance_test(self, batch_size: int) -> None:
        """Seleciona um lote fixo e executa dez iteracoes com as mesmas imagens."""
        self._robotized_test_active = False
        if self.ui_busy:
            return
        batch_size = int(batch_size)
        if batch_size not in BATCH_TEST_SIZES:
            raise ValueError(f"Tamanho de lote nao suportado: {batch_size}")

        selected = filedialog.askdirectory(
            title=self._t("test.select_batch_folder", count=batch_size)
        )
        if not selected:
            return
        folder = Path(selected)
        if not folder.exists() or not folder.is_dir():
            messagebox.showerror(
                self._t("dialog.error"),
                self._t("scan.invalid_folder", folder=folder),
            )
            return

        groups, _duplicates = discover_image_groups(folder)
        available_items: list[tuple[str, str, Path]] = []
        for group_key in sorted(groups):
            for time_tag in TIME_ORDER:
                path = groups[group_key].get(time_tag)
                if path is not None:
                    available_items.append((group_key, time_tag, path))

        if len(available_items) < batch_size:
            messagebox.showwarning(
                self._t("dialog.warning"),
                self._t(
                    "test.batch_not_enough",
                    found=len(available_items),
                    required=batch_size,
                ),
            )
            return

        batch_items = available_items[:batch_size]
        self.folder_var.set(str(folder))
        self._reset_all_state()

        selected_groups: dict[str, dict[str, Path]] = {}
        for group_key, time_tag, path in batch_items:
            selected_groups.setdefault(group_key, {})[time_tag] = path
        self.group_files = selected_groups
        self.group_order = sorted(selected_groups)

        labels: list[str] = []
        for group_key in self.group_order:
            times = [time_tag for time_tag in TIME_ORDER if time_tag in selected_groups[group_key]]
            label = f"{group_key} | {'/'.join(times)}"
            labels.append(label)
            self.group_labels[group_key] = label
            self.label_to_group[label] = group_key
        self.group_combo["values"] = labels
        self.group_var.set(labels[0])
        self.current_group_key = None
        self._batch_test_context = {
            "folder": folder,
            "batch_size": batch_size,
            "iterations": BATCH_TEST_ITERATIONS,
            "started_at": datetime.now().astimezone(),
        }
        self._batch_test_rows = []
        self._processing_metrics_by_item = {}

        self._set_ui_busy(False)
        self._show_group(self.group_order[0], mark_reviewed=False)
        self._show_viewer_page()
        self._start_processing_job(
            mode="batch_test",
            items=batch_items * BATCH_TEST_ITERATIONS,
            save_after=False,
        )

    def _show_config_page(self, set_status: bool = True) -> None:
        """Exibe a tela inicial de configuracoes e oculta a visualizacao das imagens."""
        config_page = self.config_page
        viewer_page = self.viewer_page
        test_selection_page = self.test_selection_page
        if test_selection_page is not None and test_selection_page.winfo_manager():
            test_selection_page.pack_forget()
        if viewer_page is not None and viewer_page.winfo_manager():
            viewer_page.pack_forget()
        if config_page is not None and (not config_page.winfo_manager()):
            config_page.pack(fill=tk.BOTH, expand=True)
        self._track_page_transition("configuracoes")
        if set_status:
            self.status_var.set(self._t("status.config_page_opened"))

    def _show_viewer_page(self) -> None:
        """Exibe a tela de visualizacao das imagens e oculta o painel de configuracoes."""
        config_page = self.config_page
        viewer_page = self.viewer_page
        test_selection_page = self.test_selection_page
        if test_selection_page is not None and test_selection_page.winfo_manager():
            test_selection_page.pack_forget()
        if config_page is not None and config_page.winfo_manager():
            config_page.pack_forget()
        if viewer_page is not None and (not viewer_page.winfo_manager()):
            viewer_page.pack(fill=tk.BOTH, expand=True)
        self._track_page_transition("visualizacao")
        self._refresh_metrics()
        self._refresh_figure()
        self.status_var.set(self._t("status.viewer_page_opened"))

    def _hide_results_panel_on_startup(self) -> None:
        body = self.body_paned
        left = self.left_panel
        if body is None or left is None:
            self.results_panel_visible = False
            if self.toggle_results_btn is not None:
                self.toggle_results_btn.configure(text=self._results_toggle_label())
            return

        left_name = str(left)
        if left_name in set(body.panes()):
            body.forget(left)
        self.results_panel_visible = False
        if self.toggle_results_btn is not None:
            self.toggle_results_btn.configure(text=self._results_toggle_label())
        self._refresh_help_tab()

    def _set_ui_busy(self, busy: bool) -> None:
        """Alterna estado ocupada/livre da interface e atualiza controles relevantes."""
        self.ui_busy = busy
        btn_state = tk.DISABLED if busy else tk.NORMAL

        for btn in (
            self.browse_btn,
            self.browse_output_btn,
            self.scan_btn,
            self.process_all_btn,
            self.contour_settings_btn,
            self.go_viewer_btn,
            self.open_config_btn,
            self.prev_btn,
            self.next_btn,
            self.redefine_roi_btn,
            self.clear_roi_btn,
            self.save_btn,
            self.single_image_test_btn,
            self.batch_mode_btn,
            self.robotized_test_btn,
            self.test_modes_btn_config,
        ):
            if btn is not None:
                btn.configure(state=btn_state)
        for btn in self.batch_test_buttons:
            btn.configure(state=btn_state)

        combo_state = "readonly" if (not busy and self.group_order) else "disabled"
        if self.group_combo is not None:
            self.group_combo.configure(state=combo_state)
        if self.preview_mode_combo is not None:
            self.preview_mode_combo.configure(state=("disabled" if busy else "readonly"))
        if self.preview_source_mode_combo is not None:
            self.preview_source_mode_combo.configure(state=("disabled" if busy else "readonly"))
        self._refresh_contour_color_controls_state()

        if busy:
            for check in self.refazer_checks.values():
                check.configure(state=tk.DISABLED)
            for check in self.no_area_checks.values():
                check.configure(state=tk.DISABLED)
            self.root.configure(cursor="watch")
        else:
            self.root.configure(cursor="")
            self._sync_check_vars_for_group()
            self._update_navigation_buttons()
        self._sync_cancel_controls()
        self._update_inline_progress_visibility()

    def _cancel_enabled(self) -> bool:
        if not self.ui_busy:
            return False
        if self.active_processing_mode not in ("full", "reprocess", "single_test"):
            return False
        return not self.cancel_event.is_set()

    def _sync_cancel_controls(self) -> None:
        state = tk.NORMAL if self._cancel_enabled() else tk.DISABLED
        if self.cancel_btn is not None:
            self.cancel_btn.configure(state=state)

    def _update_inline_progress_visibility(self) -> None:
        frame = self.inline_progress_frame
        if frame is None:
            return
        should_show = bool(
            self.ui_busy
            and (self.active_processing_mode in ("full", "reprocess"))
            and self._progress_popup_dismissed
        )
        mapped = bool(frame.winfo_manager())
        if should_show and (not mapped):
            frame.pack(fill=tk.X, pady=(0, 6), before=self.metrics_text)
            self._show_metrics_tab()
        elif (not should_show) and mapped:
            frame.pack_forget()

    def _on_progress_popup_close_requested(self) -> None:
        popup = self.progress_popup
        if popup is not None and popup.winfo_exists():
            popup.destroy()
        self.progress_popup = None
        self._progress_popup_dismissed = bool(
            self.ui_busy and (self.active_processing_mode in ("full", "reprocess"))
        )
        self._update_inline_progress_visibility()

    def _open_progress_popup(self, title: str) -> None:
        self._progress_popup_dismissed = False
        self._update_inline_progress_visibility()
        popup = self.progress_popup
        if popup is None or not popup.winfo_exists():
            popup = tk.Toplevel(self.root)
            popup.geometry("620x190")
            popup.resizable(False, False)
            popup.transient(self.root)
            popup.protocol("WM_DELETE_WINDOW", self._on_progress_popup_close_requested)

            frame = ttk.Frame(popup, padding=12)
            frame.pack(fill=tk.BOTH, expand=True)

            ttk.Label(frame, text=self._t("label.processing_progress")).pack(anchor=tk.W)
            info_grid = ttk.Frame(frame)
            info_grid.pack(fill=tk.X, pady=(8, 6))
            ttk.Label(info_grid, text=self._t("label.current_image")).grid(row=0, column=0, sticky=tk.W)
            ttk.Label(
                info_grid,
                textvariable=self.progress_current_image_var,
                anchor=tk.W,
                justify=tk.LEFT,
                wraplength=460,
            ).grid(row=0, column=1, sticky=tk.W, padx=(6, 0))
            ttk.Label(info_grid, text=self._t("label.remaining")).grid(row=1, column=0, sticky=tk.W, pady=(4, 0))
            ttk.Label(info_grid, textvariable=self.progress_remaining_var, anchor=tk.W).grid(
                row=1, column=1, sticky=tk.W, padx=(6, 0), pady=(4, 0)
            )
            ttk.Label(info_grid, text=self._t("label.eta")).grid(row=2, column=0, sticky=tk.W, pady=(4, 0))
            ttk.Label(info_grid, textvariable=self.progress_eta_var, anchor=tk.W).grid(
                row=2, column=1, sticky=tk.W, padx=(6, 0), pady=(4, 0)
            )
            info_grid.columnconfigure(1, weight=1)

            row = ttk.Frame(frame)
            row.pack(fill=tk.X)
            ttk.Progressbar(
                row,
                mode="determinate",
                variable=self.progress_var,
                maximum=100.0,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)
            ttk.Label(row, textvariable=self.progress_percent_var, width=8, anchor=tk.E).pack(side=tk.LEFT, padx=(8, 0))
            self.progress_popup = popup
        else:
            popup.deiconify()
            popup.lift()

        popup.title(title)
        self._translate_widget_tree(popup)
        self._sync_cancel_controls()
        self._center_toplevel_on_root(popup, width=620, height=190)

    def _close_progress_popup(self) -> None:
        popup = self.progress_popup
        if popup is None:
            self._progress_popup_dismissed = False
            self._update_inline_progress_visibility()
            return
        if popup.winfo_exists():
            popup.destroy()
        self.progress_popup = None
        self._progress_popup_dismissed = False
        self._update_inline_progress_visibility()

    def _show_progress_tab(self) -> None:
        if self.side_notebook is None:
            return
        if self.progress_tab is not None:
            self.side_notebook.select(self.progress_tab)
            return
        self._show_metrics_tab()

    def _show_metrics_tab(self) -> None:
        if self.side_notebook is None or self.metrics_tab is None:
            return
        self.side_notebook.select(self.metrics_tab)

    def _clear_progress_log(self) -> None:
        self.progress_log_last_status = ""
        if self.progress_log_text is None:
            return
        self.progress_log_text.configure(state=tk.NORMAL)
        self.progress_log_text.delete("1.0", tk.END)
        self.progress_log_text.configure(state=tk.DISABLED)

    def _append_progress_log(self, status: str, pct: float | None = None, force: bool = False) -> None:
        clean = " ".join(str(status).split())
        if not clean:
            return
        if (not force) and clean == self.progress_log_last_status:
            return

        self.progress_log_last_status = clean
        if self.progress_log_text is None:
            return

        stamp = time.strftime("%H:%M:%S")
        pct_text = "" if pct is None else f" [{float(pct):5.1f}%]"
        line = f"[{stamp}]{pct_text} {clean}"
        self.progress_log_text.configure(state=tk.NORMAL)
        self.progress_log_text.insert(tk.END, line + "\n")
        total_lines = int(self.progress_log_text.index("end-1c").split(".")[0])
        if total_lines > 250:
            self.progress_log_text.delete("1.0", f"{total_lines - 249}.0")
        self.progress_log_text.see(tk.END)
        self.progress_log_text.configure(state=tk.DISABLED)

    @staticmethod
    def _format_eta(seconds: float | None) -> str:
        if seconds is None:
            return "--"
        if not math.isfinite(float(seconds)):
            return "--"
        sec = max(0, int(round(float(seconds))))
        hh, rem = divmod(sec, 3600)
        mm, ss = divmod(rem, 60)
        if hh > 0:
            return f"{hh:02d}:{mm:02d}:{ss:02d}"
        return f"{mm:02d}:{ss:02d}"

    def _request_cancel_processing(self) -> None:
        if not self.ui_busy:
            return
        if self.active_processing_mode not in ("full", "reprocess"):
            return
        if self.cancel_event.is_set():
            return
        self.cancel_event.set()
        self.status_var.set(self._t("processing.cancel_requested"))
        self.progress_eta_var.set("--")
        self._sync_cancel_controls()

    def _set_progress_feedback(
        self,
        pct: float,
        status: str,
        current_image: str | None = None,
        remaining: int | None = None,
        eta_seconds: float | None = None,
    ) -> None:
        pct_clamped = min(max(float(pct), 0.0), 100.0)
        self.progress_var.set(pct_clamped)
        self.progress_percent_var.set(f"{pct_clamped:.1f}%")
        self.status_var.set(status)
        self._update_inline_progress_visibility()

        image_text = current_image if current_image is not None else status
        self.progress_current_image_var.set(str(image_text))
        if remaining is None:
            self.progress_remaining_var.set("-")
        else:
            self.progress_remaining_var.set(str(max(0, int(remaining))))

        if eta_seconds is None:
            if pct_clamped >= 100.0:
                self.progress_eta_var.set("00:00")
            else:
                self.progress_eta_var.set("--")
        else:
            self.progress_eta_var.set(self._format_eta(eta_seconds))

        popup = self.progress_popup
        if popup is not None and popup.winfo_exists():
            popup.update_idletasks()
        self.root.update_idletasks()

    @staticmethod
    def _normalize_folder_entry(raw_value: object) -> str:
        text = str(raw_value or "").strip()
        if not text:
            return ""
        expanded = os.path.expanduser(os.path.expandvars(text))
        return os.path.normcase(os.path.normpath(expanded))

    def _on_folder_var_changed(self, *_args) -> None:
        current = self._normalize_folder_entry(self.folder_var.get())
        if current == self._last_folder_entry_value:
            return
        self._last_folder_entry_value = current

        if self.group_combo is None:
            return
        had_loaded_items = bool(self.group_order or self.group_files or self.processed_by_group)
        if not had_loaded_items:
            return

        self._reset_all_state()
        self._refresh_metrics()
        self._refresh_figure()
        self.status_var.set(self._t("folder.changed_reload"))

    def _browse_folder(self) -> None:
        folder = filedialog.askdirectory(title=self._t("folder.select_images_title"))
        if folder:
            self.folder_var.set(folder)
            self._save_user_settings()

    def _browse_output_folder(self) -> None:
        initial_dir = self.output_folder_var.get().strip() or self.folder_var.get().strip()
        kwargs = {"title": self._t("folder.select_output_title")}
        if initial_dir:
            kwargs["initialdir"] = initial_dir
        folder = filedialog.askdirectory(**kwargs)
        if folder:
            self.output_folder_var.set(folder)
            self._save_user_settings()

    @staticmethod
    def _sorted_image_paths_for_scan(folder: Path) -> list[Path]:
        paths = [
            p
            for p in folder.glob("*")
            if p.is_file() and (p.suffix.lower() in SUPPORTED_EXTS)
        ]
        return sorted(paths, key=lambda p: p.name.lower())

    def _collect_grouping_entries(
        self,
        folder: Path,
        groups: dict[str, dict[str, Path]],
        duplicates: list[tuple[str, str, Path, Path]],
    ) -> list[GroupingEntry]:
        all_paths = self._sorted_image_paths_for_scan(folder)
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
                stem_clean = " ".join(_strip_accents(path.stem).split()).strip()
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

    def _clear_group_review_host(self) -> ttk.Frame | None:
        host = getattr(self, "group_review_host", None)
        if host is None:
            return None
        for child in list(host.winfo_children()):
            try:
                child.destroy()
            except Exception:
                pass
        return host

    def _show_group_review_message(self, text: str) -> None:
        host = self._clear_group_review_host()
        if host is None:
            return
        ttk.Label(
            host,
            text=str(text),
            justify=tk.LEFT,
            wraplength=1120,
        ).pack(anchor=tk.W, fill=tk.X)

    def _review_grouping_entries(self, entries: list[GroupingEntry]) -> list[GroupingEntry] | None:
        if not entries:
            return []
        host = self._clear_group_review_host()
        if host is None:
            return None

        source_labels = {
            "auto": self._t("review.source.auto"),
            "duplicata": self._t("review.source.duplicate"),
            "sugestao": self._t("review.source.suggestion"),
        }

        confirmed = {"ok": False}
        done_var = tk.BooleanVar(value=False)
        summary_var = tk.StringVar(value="")
        preview_var = tk.StringVar(value=self._t("review.select_preview"))
        group_edit_var = tk.StringVar(value="")
        time_edit_var = tk.StringVar(value="")

        container = ttk.Frame(host, padding=6)
        container.pack(fill=tk.BOTH, expand=True)
        ttk.Label(container, textvariable=summary_var).pack(anchor=tk.W, pady=(4, 8))

        content = ttk.Panedwindow(container, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(content)
        right_panel = ttk.Frame(content, padding=(10, 0, 0, 0))
        content.add(left_panel, weight=1)
        content.add(right_panel, weight=1)

        def keep_half_split(_event=None) -> None:
            try:
                total_width = int(content.winfo_width())
                if total_width > 40:
                    content.sashpos(0, total_width // 2)
            except Exception:
                pass

        content.bind("<Configure>", keep_half_split, add="+")
        self.root.after(40, keep_half_split)

        tree_frame = ttk.Frame(left_panel)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("usar", "grupo", "tempo", "origem", "arquivo")
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings", selectmode="extended")
        review_tree_style = "CORA.Review.Treeview"
        review_tree_heading_style = "CORA.Review.Treeview.Heading"
        palette = self._theme_palette()
        panel_bg = str(palette.get("panel_bg", "#FFFFFF"))
        fg = str(palette.get("fg", "#111827"))
        header_bg = str(palette.get("header_bg", "#E5EBF4"))
        select_bg = str(palette.get("select_bg", "#2F6FED"))
        select_fg = str(palette.get("select_fg", "#FFFFFF"))
        try:
            style = ttk.Style(self.root)
            base_font = tkfont.nametofont("TkDefaultFont")
            family = str(base_font.actual("family"))
            size = max(11, int(base_font.actual("size")))
            style.configure(
                review_tree_style,
                rowheight=30,
                font=(family, size),
                background=panel_bg,
                fieldbackground=panel_bg,
                foreground=fg,
            )
            style.map(
                review_tree_style,
                background=[("selected", select_bg)],
                foreground=[("selected", select_fg)],
            )
            style.configure(review_tree_heading_style, font=(family, size, "bold"), background=header_bg, foreground=fg)
            tree.configure(style=review_tree_style)
        except Exception:
            try:
                ttk.Style(self.root).configure(review_tree_style, rowheight=30)
                tree.configure(style=review_tree_style)
            except Exception:
                pass
        tree.heading("usar", text=self._t("review.heading.include"))
        tree.heading("grupo", text=self._t("review.group"))
        tree.heading("tempo", text=self._t("review.time"))
        tree.heading("origem", text=self._t("review.heading.source"))
        tree.heading("arquivo", text=self._t("review.heading.file"))
        tree.column("usar", width=104, anchor=tk.CENTER, stretch=False)
        tree.column("grupo", width=210, anchor=tk.W, stretch=False)
        tree.column("tempo", width=80, anchor=tk.CENTER, stretch=False)
        tree.column("origem", width=100, anchor=tk.CENTER, stretch=False)
        tree.column("arquivo", width=640, anchor=tk.W, stretch=True)

        yscroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
        xscroll = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        tree.grid(row=0, column=0, sticky=tk.NSEW)
        yscroll.grid(row=0, column=1, sticky=tk.NS)
        xscroll.grid(row=1, column=0, sticky=tk.EW)
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        is_light_theme = self._ui_theme_mode_key(self.ui_theme_key) == "light"
        off_color = "#9B6574" if is_light_theme else "#95A1B8"
        dup_color = "#8A214E" if is_light_theme else "#F0B97A"
        tree.tag_configure("off", foreground=off_color)
        tree.tag_configure("dup", foreground=dup_color)
        self._register_tooltip(
            tree,
            self._t("review.include_tooltip"),
        )
        toggle_on_glyph = "[x]"
        toggle_off_glyph = "[ ]"

        preview_header = ttk.Frame(right_panel)
        preview_header.pack(fill=tk.X)

        preview_state: dict[str, object] = {"photo": None, "path": None, "mode": "original"}

        def _preview_variant_rgb(path_obj: Path, mode: str) -> np.ndarray | None:
            original_rgb = self._read_original_preview_image(path_obj)
            if original_rgb is None:
                return None
            if str(mode).lower() == "enhanced":
                cache_key = f"review::{self._image_cache_key(path_obj)}"
                return self._best_defined_image(original_rgb, cache_key=cache_key)
            return np.asarray(original_rgb, dtype=np.uint8)

        def open_preview_fullscreen() -> None:
            path_obj = preview_state.get("path")
            if not isinstance(path_obj, Path):
                messagebox.showinfo(self._t("dialog.preview"), self._t("review.select_image_full"), parent=self.root)
                return
            if Image is None or ImageTk is None:
                messagebox.showinfo(self._t("dialog.preview"), self._t("review.pillow_unavailable"), parent=self.root)
                return
            source_rgb = _preview_variant_rgb(path_obj, str(preview_state.get("mode", "original")))
            if source_rgb is None:
                messagebox.showwarning(
                    self._t("dialog.preview"),
                    self._t("review.open_image_failed", name=path_obj.name),
                    parent=self.root,
                )
                return
            source_img = Image.fromarray(np.asarray(source_rgb, dtype=np.uint8), mode="RGB")

            popup = tk.Toplevel(self.root)
            popup.title(f"Preview - {path_obj.name}")
            try:
                popup.attributes("-fullscreen", True)
            except Exception:
                try:
                    popup.state("zoomed")
                except Exception:
                    pass

            root_frame = ttk.Frame(popup, padding=8)
            root_frame.pack(fill=tk.BOTH, expand=True)

            top_bar = ttk.Frame(root_frame)
            top_bar.pack(fill=tk.X, pady=(0, 6))
            ttk.Label(top_bar, text=path_obj.name).pack(side=tk.LEFT, anchor=tk.W)
            ttk.Button(top_bar, text=self._t("button.close_esc"), command=popup.destroy).pack(side=tk.RIGHT)

            image_holder = tk.Label(
                root_frame,
                text=self._t("review.loading"),
                justify=tk.CENTER,
                anchor=tk.CENTER,
                background=str(palette.get("text_bg", "#202733")),
                foreground=str(palette.get("text_fg", "#E7ECF3")),
            )
            image_holder.pack(fill=tk.BOTH, expand=True)
            full_state: dict[str, object] = {"photo": None, "after_id": None}

            try:
                resampling = getattr(Image, "Resampling", Image).LANCZOS
            except Exception:
                resampling = Image.LANCZOS

            def render_fullscreen() -> None:
                width = max(1, int(image_holder.winfo_width()))
                height = max(1, int(image_holder.winfo_height()))
                if width < 20 or height < 20:
                    return
                img = source_img.copy()
                img.thumbnail((width, height), resampling)
                photo = ImageTk.PhotoImage(img)
                image_holder.configure(image=photo, text="")
                image_holder.image = photo
                full_state["photo"] = photo

            def schedule_render(_event=None) -> None:
                after_id = full_state.get("after_id")
                if isinstance(after_id, str):
                    try:
                        popup.after_cancel(after_id)
                    except Exception:
                        pass
                full_state["after_id"] = popup.after(30, render_fullscreen)

            popup.bind("<Configure>", schedule_render, add="+")
            popup.bind("<Escape>", lambda _e: popup.destroy())
            popup.after(50, render_fullscreen)

        preview_full_btn = ttk.Button(
            preview_header,
            text=self._t("button.fullscreen"),
            command=open_preview_fullscreen,
            state=tk.DISABLED,
        )
        preview_full_btn.pack(side=tk.RIGHT)
        preview_toggle_btn = ttk.Button(preview_header, text=self._t("button.view_enhanced"), state=tk.DISABLED)
        preview_toggle_btn.pack(side=tk.RIGHT, padx=(0, 6))
        self._register_tooltip(preview_full_btn, self._t("review.fullscreen_tooltip"))
        self._register_tooltip(
            preview_toggle_btn,
            self._t("review.toggle_preview_tooltip"),
        )
        preview_box = tk.Label(
            right_panel,
            text=self._t("review.select_row_preview"),
            justify=tk.CENTER,
            anchor=tk.CENTER,
            relief=tk.SUNKEN,
            borderwidth=1,
            background=str(palette.get("text_bg", "#FFFFFF")),
            foreground=str(palette.get("text_fg", "#111827")),
        )
        preview_box.pack(fill=tk.BOTH, expand=True)
        preview_info = ttk.Frame(right_panel)
        preview_info.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(preview_info, text=self._t("label.selected_image")).pack(anchor=tk.W)
        ttk.Label(
            preview_info,
            textvariable=preview_var,
            justify=tk.LEFT,
            wraplength=420,
        ).pack(anchor=tk.W, fill=tk.X, pady=(4, 0))

        def update_preview(entry: GroupingEntry) -> None:
            tempo_txt = "-" if entry.time_tag is None else str(entry.time_tag).upper()
            preview_state["path"] = entry.path
            mode = str(preview_state.get("mode", "original")).lower()
            mode_label = self._preview_source_mode_label("original" if mode == "original" else "enhanced")
            preview_var.set(
                f"{self._t('review.file')}: {entry.rel_path}\n"
                f"{self._t('review.group')}: {entry.group_key}\n"
                f"{self._t('review.time')}: {tempo_txt}\n"
                f"{self._t('review.source')}: {source_labels.get(entry.source, entry.source)}\n"
                f"{self._t('review.view')}: {mode_label}"
            )
            if Image is None or ImageTk is None:
                preview_box.configure(
                    image="",
                    text=self._t("review.preview_unavailable"),
                )
                preview_full_btn.configure(state=tk.DISABLED)
                preview_toggle_btn.configure(state=tk.DISABLED)
                preview_state["photo"] = None
                return
            try:
                rgb = _preview_variant_rgb(entry.path, mode)
                if rgb is None:
                    raise ValueError("Nao foi possivel abrir imagem para preview.")
                img = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
                max_w, max_h = 520, 420
                try:
                    resampling = getattr(Image, "Resampling", Image).LANCZOS
                except Exception:
                    resampling = Image.LANCZOS
                img.thumbnail((max_w, max_h), resampling)
                photo = ImageTk.PhotoImage(img)
                preview_box.configure(image=photo, text="")
                preview_state["photo"] = photo
                preview_box.image = photo
                preview_full_btn.configure(state=tk.NORMAL)
                toggle_label = self._t("button.view_original" if mode == "enhanced" else "button.view_enhanced")
                preview_toggle_btn.configure(text=toggle_label, state=tk.NORMAL)
            except Exception:
                preview_box.configure(
                    image="",
                    text=self._t("review.open_image_failed", name=entry.path.name),
                )
                preview_full_btn.configure(state=tk.DISABLED)
                preview_toggle_btn.configure(state=tk.DISABLED)
                preview_state["photo"] = None

        def toggle_preview_variant() -> None:
            current = str(preview_state.get("mode", "original")).lower()
            preview_state["mode"] = "enhanced" if current == "original" else "original"
            selected_ids = list(tree.selection())
            if selected_ids:
                update_preview(entries[int(selected_ids[0])])

        preview_toggle_btn.configure(command=toggle_preview_variant)

        def refresh_row(i: int) -> None:
            entry = entries[i]
            use_text = toggle_on_glyph if entry.selected else toggle_off_glyph
            tempo_text = "-" if (entry.time_tag is None) else entry.time_tag.upper()
            source_text = source_labels.get(entry.source, str(entry.source))
            tags: list[str] = []
            if not entry.selected:
                tags.append("off")
            if entry.source == "duplicata":
                tags.append("dup")
            tree.item(
                str(i),
                values=(use_text, entry.group_key, tempo_text, source_text, entry.rel_path),
                tags=tuple(tags),
            )

        for idx, _entry in enumerate(entries):
            tree.insert("", tk.END, iid=str(idx))
            refresh_row(idx)

        def refresh_summary() -> None:
            selected_entries = [e for e in entries if e.selected]
            selected_count = len(selected_entries)
            group_count = len(
                {
                    " ".join(str(e.group_key).split()).strip().lower()
                    for e in selected_entries
                    if str(e.group_key).strip()
                }
            )
            missing_time = sum(1 for e in selected_entries if e.time_tag not in TIME_ORDER)
            summary_var.set(
                self._t(
                    "review.summary",
                    selected=selected_count,
                    total=len(entries),
                    groups=group_count,
                    missing=missing_time,
                )
            )

        def on_select(_event=None) -> None:
            selected_ids = list(tree.selection())
            if not selected_ids:
                return
            first = entries[int(selected_ids[0])]
            group_edit_var.set(first.group_key)
            time_edit_var.set("" if first.time_tag is None else first.time_tag)
            update_preview(first)

        def _set_selected_flag(iids: list[str], flag: bool) -> None:
            for iid in iids:
                idx = int(iid)
                entries[idx].selected = bool(flag)
                refresh_row(idx)
            refresh_summary()

        def on_tree_click(event) -> str | None:
            row_id = tree.identify_row(event.y)
            col_id = tree.identify_column(event.x)
            if not row_id or col_id != "#1":
                return None

            if row_id not in tree.selection():
                tree.selection_set(row_id)
            tree.focus(row_id)
            tree.see(row_id)

            target_ids = list(tree.selection()) or [row_id]
            base_flag = bool(entries[int(row_id)].selected)
            _set_selected_flag(target_ids, not base_flag)
            on_select()
            return "break"

        tree.bind("<<TreeviewSelect>>", on_select)
        tree.bind("<Button-1>", on_tree_click, add="+")

        controls = ttk.Frame(left_panel)
        controls.pack(fill=tk.X, pady=(8, 0))

        ttk.Label(controls, text=self._t("label.group")).grid(row=0, column=0, sticky=tk.W)
        group_entry = ttk.Entry(controls, textvariable=group_edit_var)
        group_entry.grid(row=0, column=1, sticky=tk.EW, padx=(6, 10))

        ttk.Label(controls, text=self._t("label.time")).grid(row=0, column=2, sticky=tk.W)
        time_combo = ttk.Combobox(
            controls,
            textvariable=time_edit_var,
            values=["", "0h", "24h", "48h"],
            state="readonly",
            width=8,
        )
        time_combo.grid(row=0, column=3, sticky=tk.W, padx=(6, 10))

        def apply_to_selection() -> None:
            selected_ids = list(tree.selection())
            if not selected_ids:
                messagebox.showinfo(self._t("dialog.review"), self._t("review.no_selection_apply"), parent=self.root)
                return

            group_text = " ".join(str(group_edit_var.get()).split()).strip()
            time_raw = str(time_edit_var.get()).strip().lower()
            time_norm = time_raw if time_raw in TIME_ORDER else None

            for iid in selected_ids:
                entry = entries[int(iid)]
                if group_text:
                    entry.group_key = group_text
                entry.time_tag = time_norm
                refresh_row(int(iid))

            refresh_summary()

        apply_group_time_btn = ttk.Button(controls, text=self._t("button.apply_group_time"), command=apply_to_selection)
        apply_group_time_btn.grid(row=0, column=4, sticky=tk.EW, padx=(0, 6))
        self._register_tooltip(
            apply_group_time_btn,
            self._t("review.apply_tooltip"),
        )

        def mark_all(flag: bool) -> None:
            for i, entry in enumerate(entries):
                entry.selected = flag
                refresh_row(i)
            refresh_summary()

        bulk_toggle_state = {"next_mark_all": True}
        bulk_toggle_btn = ttk.Button(controls)

        def _refresh_bulk_toggle_button() -> None:
            if bulk_toggle_state["next_mark_all"]:
                bulk_toggle_btn.configure(text=self._t("button.mark_all"))
                self._register_tooltip(
                    bulk_toggle_btn,
                    self._t("review.mark_all_tooltip"),
                )
            else:
                bulk_toggle_btn.configure(text=self._t("button.unmark_all"))
                self._register_tooltip(
                    bulk_toggle_btn,
                    self._t("review.unmark_all_tooltip"),
                )

        def toggle_mark_all() -> None:
            mark_flag = bool(bulk_toggle_state["next_mark_all"])
            mark_all(mark_flag)
            bulk_toggle_state["next_mark_all"] = not mark_flag
            _refresh_bulk_toggle_button()

        def robot_mark_all() -> None:
            mark_all(True)
            bulk_toggle_state["next_mark_all"] = False
            _refresh_bulk_toggle_button()

        bulk_toggle_btn.configure(command=toggle_mark_all)
        bulk_toggle_btn.grid(row=0, column=5, sticky=tk.EW, padx=(0, 6))
        _refresh_bulk_toggle_button()

        def auto_fill_missing_times() -> None:
            per_group: dict[str, list[GroupingEntry]] = {}
            for entry in entries:
                if not entry.selected:
                    continue
                key = " ".join(str(entry.group_key).split()).strip()
                if not key:
                    continue
                per_group.setdefault(key, []).append(entry)

            for group_key, group_entries in per_group.items():
                slot_used = {e.time_tag for e in group_entries if e.time_tag in TIME_ORDER}
                pending = [e for e in group_entries if e.time_tag not in TIME_ORDER]
                pending_sorted = sorted(pending, key=lambda e: e.rel_path.lower())
                for entry in pending_sorted:
                    next_time = next((t for t in TIME_ORDER if t not in slot_used), None)
                    if next_time is None:
                        break
                    entry.time_tag = next_time
                    slot_used.add(next_time)

            for i, _entry in enumerate(entries):
                refresh_row(i)
            refresh_summary()

        fill_times_btn = ttk.Button(controls, text=self._t("button.auto_fill_times"), command=auto_fill_missing_times)
        fill_times_btn.grid(row=0, column=6, sticky=tk.EW, padx=(10, 0))
        self._register_tooltip(
            fill_times_btn,
            self._t("review.auto_fill_tooltip"),
        )
        controls.columnconfigure(1, weight=1)

        def close_cancel() -> None:
            confirmed["ok"] = False
            done_var.set(True)

        def validate_current_selection() -> bool:
            invalid_group = [e for e in entries if e.selected and (not str(e.group_key).strip())]
            if invalid_group:
                details = "\n".join(x.rel_path for x in invalid_group[:8])
                if len(invalid_group) > 8:
                    details += "\n" + self._t("common.more_items", count=len(invalid_group) - 8)
                messagebox.showwarning(
                    self._t("dialog.review"),
                    self._t("review.empty_group_warning", details=details),
                    parent=self.root,
                )
                return False
            if not any(e.selected for e in entries):
                keep = messagebox.askyesno(
                    self._t("dialog.review"),
                    self._t("review.no_selected_confirm"),
                    parent=self.root,
                )
                if not keep:
                    return False
            return True

        def close_start_processing() -> None:
            if not validate_current_selection():
                return
            confirmed["ok"] = True
            done_var.set(True)

        actions_bar = ttk.Frame(controls)
        actions_bar.grid(row=1, column=0, columnspan=12, sticky=tk.EW, pady=(8, 0))
        cancel_review_btn = ttk.Button(actions_bar, text=self._t("button.cancel"), command=close_cancel)
        cancel_review_btn.pack(side=tk.RIGHT, padx=(8, 0))
        start_processing_btn = ttk.Button(actions_bar, text=self._t("button.start_processing"), command=close_start_processing)
        start_processing_btn.pack(side=tk.RIGHT)
        self._register_tooltip(cancel_review_btn, self._t("review.cancel_tooltip"))
        self._register_tooltip(
            start_processing_btn,
            self._t("review.start_tooltip"),
        )
        self._robot_review_actions = {
            "mark_all": robot_mark_all,
            "auto_fill": auto_fill_missing_times,
            "start": close_start_processing,
        }

        refresh_summary()
        if entries:
            first_id = "0"
            tree.selection_set(first_id)
            tree.focus(first_id)
            tree.see(first_id)
            on_select()
        try:
            group_entry.focus_set()
        except Exception:
            pass

        try:
            self.root.wait_variable(done_var)
        finally:
            self._robot_review_actions = {}
        if not confirmed["ok"]:
            self._show_group_review_message(self._t("review.cancelled"))
            return None
        self._start_processing_after_grouping = True
        self._show_group_review_message(self._t("review.confirmed_loading"))
        return entries

    def _build_groups_from_entries(
        self,
        entries: list[GroupingEntry],
    ) -> tuple[dict[str, dict[str, Path]], list[tuple[str, str, Path, Path]], list[str]]:
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
                warnings.append(self._t("scan.empty_group_adjustment", path=entry.rel_path))
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
                warnings.append(self._t("scan.invalid_time_adjustment", time=time_tag, path=entry.rel_path))
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

    def _append_grouping_log(
        self,
        groups: dict[str, dict[str, Path]],
        duplicates: list[tuple[str, str, Path, Path]],
    ) -> None:
        self._append_progress_log(self._t("scan.grouping_summary_log"), force=True)
        for group_key in sorted(groups.keys()):
            slot = groups[group_key]
            parts: list[str] = []
            for time_tag in TIME_ORDER:
                path = slot.get(time_tag)
                if path is not None:
                    parts.append(f"{time_tag.upper()}={path.name}")
            if not parts:
                continue
            self._append_progress_log(f"{group_key}: " + " | ".join(parts), force=True)

        if duplicates:
            self._append_progress_log(
                self._t("scan.duplicates_log_warning", count=len(duplicates)),
                force=True,
            )

    def _scan_folder(self) -> None:
        # PT: Sempre reinicia a lista visual antes de um novo carregamento. | EN: Always resets the visual list before a new load.
        self._reset_all_state()
        self._show_group_review_message(self._t("scan.preparing"))
        self._refresh_metrics()
        self._refresh_figure()

        folder_text = self.folder_var.get().strip()
        if not folder_text:
            messagebox.showerror(self._t("dialog.error"), self._t("scan.folder_required"))
            return

        folder = Path(folder_text)
        if not folder.exists() or not folder.is_dir():
            messagebox.showerror(self._t("dialog.error"), self._t("scan.invalid_folder", folder=folder))
            return

        self._save_user_settings()
        self._clear_progress_log()
        self._show_progress_tab()
        self._start_processing_after_grouping = False
        self._set_progress_feedback(
            0.0,
            self._t("scan.scanning_folder"),
            current_image=self._t("scan.scanning_files"),
            remaining=None,
            eta_seconds=None,
        )

        def on_scan_progress(done: int, total: int) -> None:
            pct = 100.0 if total <= 0 else (100.0 * float(done) / float(total))
            self._set_progress_feedback(
                pct,
                self._t("scan.grouping_progress", done=done, total=total),
                current_image=self._t("scan.scanning_files"),
                remaining=max(total - done, 0),
                eta_seconds=None,
            )

        groups_auto, duplicates_auto = discover_image_groups(folder, progress_callback=on_scan_progress)
        entries = self._collect_grouping_entries(folder, groups_auto, duplicates_auto)
        self.status_var.set(self._t("scan.review_prompt"))
        reviewed_entries = self._review_grouping_entries(entries)
        if reviewed_entries is None:
            self.status_var.set(self._t("scan.cancelled"))
            return

        groups, duplicates, warnings = self._build_groups_from_entries(reviewed_entries)
        self._reset_all_state()

        if not groups:
            self.status_var.set(self._t("scan.no_group_found_status"))
            self._show_group_review_message(self._t("scan.no_group_found_message"))
            self._refresh_metrics()
            self._refresh_figure()
            messagebox.showwarning(
                self._t("dialog.warning"),
                self._t("scan.no_group_found_warning"),
            )
            return

        self._append_grouping_log(groups, duplicates)

        self.group_files = groups
        self.group_order = sorted(groups.keys())

        labels: list[str] = []
        for group_key in self.group_order:
            times = [t for t in TIME_ORDER if t in groups[group_key]]
            label = f"{group_key} | {'/'.join(times)}"
            labels.append(label)
            self.group_labels[group_key] = label
            self.label_to_group[label] = group_key

        self.group_combo["values"] = labels
        self.group_var.set(labels[0])
        self.current_group_key = None
        # PT: Mantem o mesmo comportamento de navegacao usado apos o processamento: combo habilitado e primeiro grupo exibido no painel.
        # EN: Preserves the navigation behavior used after processing: the combo box is enabled and the first group is displayed in the panel.
        self._set_ui_busy(False)
        self._show_group(
            self.group_order[0],
            mark_reviewed=False,
            status_text=self._t("scan.groups_found", count=len(labels)),
        )

        if duplicates:
            preview = []
            for group_key, time_tag, kept, ignored in duplicates[:12]:
                preview.append(
                    self._t(
                        "scan.duplicate_line",
                        group=group_key,
                        time=time_tag.upper(),
                        kept=kept.name,
                        ignored=ignored.name,
                    )
                )
            detail_text = "\n".join(preview)
            if len(duplicates) > 12:
                detail_text += "\n" + self._t("scan.more_duplicates", count=len(duplicates) - 12)
            messagebox.showwarning(
                self._t("scan.duplicates_title"),
                self._t("scan.duplicates_message", details=detail_text),
            )

        if warnings:
            preview = "\n".join(warnings[:10])
            if len(warnings) > 10:
                preview += "\n" + self._t("common.more_warnings", count=len(warnings) - 10)
            messagebox.showwarning(
                self._t("scan.grouping_warnings_title"),
                self._t("scan.grouping_warnings_message", details=preview),
            )

        self._show_viewer_page()
        if self._start_processing_after_grouping:
            self.status_var.set(self._t("scan.loaded_starting", count=len(labels)))
            self._start_full_processing()
        else:
            self.status_var.set(self._t("scan.loaded_not_started", count=len(labels)))

    def _reset_all_state(self) -> None:
        self.group_files = {}
        self.group_labels = {}
        self.label_to_group = {}
        self.group_order = []
        self.current_group_key = None

        self.roi_by_item = {}
        self.processing_mask_by_item = {}
        self.processed_by_group = {}
        self.processing_errors = {}
        self.preview_original_image_cache = {}
        self.preview_enhanced_image_cache = {}
        self.pending_reprocess = set()
        self.reprocessed_history = set()
        self.reviewed_groups = set()
        self.reviewed_items = set()
        self.no_area_items = set()
        self.no_area_backup = {}
        self._single_test_runs = []
        self._batch_test_context = None
        self._batch_test_rows = []
        self._processing_metrics_by_item = {}

        self.group_combo["values"] = []
        self.group_var.set("")
        self.progress_var.set(0.0)
        self.progress_percent_var.set("0.0%")

    def _selected_group_key(self) -> str | None:
        label = self.group_var.get().strip()
        if not label:
            return None
        return self.label_to_group.get(label)

    def _group_times(self, group_key: str) -> list[str]:
        group_map = self.group_files.get(group_key, {})
        return [t for t in TIME_ORDER if t in group_map]

    def _current_group_index(self) -> int:
        if self.current_group_key is None:
            return -1
        try:
            return self.group_order.index(self.current_group_key)
        except ValueError:
            return -1

    def _update_navigation_buttons(self) -> None:
        if self.ui_busy or not self.group_order:
            self.prev_btn.configure(state=tk.DISABLED)
            self.next_btn.configure(state=tk.DISABLED)
            return

        idx = self._current_group_index()
        if idx <= 0:
            self.prev_btn.configure(state=tk.DISABLED)
        else:
            self.prev_btn.configure(state=tk.NORMAL)

        if idx < 0 or idx >= (len(self.group_order) - 1):
            self.next_btn.configure(state=tk.DISABLED)
        else:
            self.next_btn.configure(state=tk.NORMAL)

    def _show_group_by_index(self, index: int, mark_reviewed: bool = True) -> None:
        if not self.group_order:
            return
        idx = min(max(0, index), len(self.group_order) - 1)
        self._show_group(self.group_order[idx], mark_reviewed=mark_reviewed)

    def _toggle_results_panel(self) -> None:
        body = self.body_paned
        left = self.left_panel
        right = self.right_panel
        if body is None or left is None or right is None:
            return

        pane_names = set(body.panes())
        left_name = str(left)
        right_name = str(right)

        if self.results_panel_visible:
            if left_name in pane_names:
                body.forget(left)
            self.results_panel_visible = False
            if self.toggle_results_btn is not None:
                self.toggle_results_btn.configure(text=self._results_toggle_label())
            self._refresh_help_tab()
            if not self.ui_busy:
                self.status_var.set(self._t("status.results_hidden"))
            return

        if left_name in pane_names:
            body.forget(left)
        if right_name in pane_names:
            body.forget(right)
        body.add(left, weight=2)
        body.add(right, weight=5)
        self.results_panel_visible = True
        if self.toggle_results_btn is not None:
            self.toggle_results_btn.configure(text=self._results_toggle_label())
        self._refresh_help_tab()
        if not self.ui_busy:
            self.status_var.set(self._t("status.results_shown"))

    def _show_prev_group(self) -> None:
        idx = self._current_group_index()
        if idx > 0:
            self._show_group_by_index(idx - 1, mark_reviewed=True)

    def _show_next_group(self) -> None:
        idx = self._current_group_index()
        if 0 <= idx < len(self.group_order) - 1:
            self._show_group_by_index(idx + 1, mark_reviewed=True)

    def _on_group_selected(self, _event=None) -> None:
        if self.ui_busy:
            return
        key = self._selected_group_key()
        if key is not None:
            self._show_group(key, mark_reviewed=True)

    def _show_group(self, group_key: str, mark_reviewed: bool = True, status_text: str | None = None) -> None:
        if group_key not in self.group_files:
            return

        self.current_group_key = group_key
        if mark_reviewed:
            self.reviewed_groups.add(group_key)
        processed = self.processed_by_group.get(group_key, {})
        for time_tag in TIME_ORDER:
            if time_tag in processed:
                self.reviewed_items.add((group_key, time_tag))

        label = self.group_labels.get(group_key, group_key)
        if self.group_var.get() != label:
            self.group_var.set(label)

        self._sync_check_vars_for_group()
        self._refresh_metrics(group_key)
        self._refresh_figure(group_key)
        self._update_navigation_buttons()
        self._show_metrics_tab()

        if status_text is not None:
            self.status_var.set(status_text)
            return

        idx = self._current_group_index() + 1
        total = len(self.group_order)
        reviewed = len(self.reviewed_groups)
        self.status_var.set(self._t("status.review_progress", index=idx, total=total, reviewed=reviewed))

    def _sync_check_vars_for_group(self) -> None:
        group_key = self.current_group_key
        processed = self.processed_by_group.get(group_key or "", {})

        for time_tag in TIME_ORDER:
            key = (group_key or "", time_tag)
            available = bool(group_key and time_tag in self.group_files.get(group_key, {}) and time_tag in processed)
            checked = bool(available and key not in self.pending_reprocess)
            self.refazer_vars[time_tag].set(checked)
            self.no_area_vars[time_tag].set(bool(available and key in self.no_area_items))

            enabled = (not self.ui_busy) and available
            self.refazer_checks[time_tag].configure(state=(tk.NORMAL if enabled else tk.DISABLED))
            self.no_area_checks[time_tag].configure(state=(tk.NORMAL if enabled else tk.DISABLED))

    def _set_no_area_override_for_item(self, group_key: str, time_tag: str, enabled: bool) -> str | None:
        key = (group_key, time_tag)
        if enabled:
            if key in self.no_area_items:
                return None
        else:
            if key not in self.no_area_items:
                return None

        group_processed = self.processed_by_group.setdefault(group_key, {})
        current = group_processed.get(time_tag)
        if enabled:
            if current is None:
                return self._t("roi.no_area.no_processing", group=group_key, time=time_tag.upper())

            self.no_area_backup.setdefault(key, (current, self.processing_errors.get(key)))
            base_rgb_u8 = np.asarray(current.artifacts.base_rgb_u8, dtype=np.uint8)
            shape = base_rgb_u8.shape[:2]
            zero_mask = np.zeros(shape, dtype=bool)
            zero_contour = np.zeros(shape, dtype=bool)

            area_manual_i = int(current.results.area_manual)
            area_auto = 0
            erro_abs = int(abs(area_auto - area_manual_i))
            erro_pct = 100.0 * float(erro_abs) / max(area_manual_i, 1)
            area_ratio = 0.0
            area_diff_norm = float(area_auto - area_manual_i) / max(float(area_manual_i), 1.0)

            results = AreaResults(
                area_manual=area_manual_i,
                area_auto=area_auto,
                erro_abs=erro_abs,
                erro_pct=float(erro_pct),
                area_ratio=float(area_ratio),
                area_diff_norm=float(area_diff_norm),
                processing_time_s=0.0,
            )

            prev_artifacts = current.artifacts
            artifacts = CORAArtifacts(
                base_rgb_u8=base_rgb_u8,
                mask_auto=zero_mask,
                roi_mask=zero_mask,
                contour_mask=zero_contour,
                contour_overlay_rgb_u8=base_rgb_u8.copy(),
                mask_overlay_rgb_u8=base_rgb_u8.copy(),
                effective_gabor_iterations=prev_artifacts.effective_gabor_iterations,
                requested_gabor_iterations=prev_artifacts.requested_gabor_iterations,
                effective_kmeans_iter=prev_artifacts.effective_kmeans_iter,
                requested_kmeans_iter=prev_artifacts.requested_kmeans_iter,
                hat_ratio=prev_artifacts.hat_ratio,
                tex_ratio=prev_artifacts.tex_ratio,
                max_cores=prev_artifacts.max_cores,
                max_processing_time_s=prev_artifacts.max_processing_time_s,
            )
            group_processed[time_tag] = ProcessedTimepoint(path=current.path, results=results, artifacts=artifacts)
            self.no_area_items.add(key)
            self.processing_errors.pop(key, None)
            return None

        backup = self.no_area_backup.pop(key, None)
        if backup is None:
            self.no_area_items.discard(key)
            return None

        prev_proc, prev_err = backup
        if prev_proc is None:
            group_processed.pop(time_tag, None)
        else:
            group_processed[time_tag] = prev_proc

        if prev_err:
            self.processing_errors[key] = prev_err
        else:
            self.processing_errors.pop(key, None)
        self.no_area_items.discard(key)
        return None

    def _on_mark_changed(self) -> None:
        group_key = self.current_group_key
        if group_key is None:
            return

        processed = self.processed_by_group.get(group_key, {})
        no_area_errors: list[str] = []
        for time_tag in TIME_ORDER:
            key = (group_key, time_tag)
            checked = bool(self.refazer_vars[time_tag].get())
            no_area_checked = bool(self.no_area_vars[time_tag].get())

            if time_tag not in processed:
                self.pending_reprocess.discard(key)
                self.no_area_items.discard(key)
                self.no_area_backup.pop(key, None)
                self.no_area_vars[time_tag].set(False)
                continue

            if no_area_checked:
                self.pending_reprocess.discard(key)
                self.refazer_vars[time_tag].set(True)
                err = self._set_no_area_override_for_item(group_key, time_tag, enabled=True)
                if err:
                    no_area_errors.append(err)
                    self.no_area_vars[time_tag].set(False)
            else:
                err = self._set_no_area_override_for_item(group_key, time_tag, enabled=False)
                if err:
                    no_area_errors.append(err)

            checked = bool(self.refazer_vars[time_tag].get())
            if checked:
                self.pending_reprocess.discard(key)
            else:
                self.pending_reprocess.add(key)

        if no_area_errors:
            msg = "\n".join(no_area_errors[:8])
            if len(no_area_errors) > 8:
                msg += "\n" + self._t("common.more_errors", count=len(no_area_errors) - 8)
            messagebox.showwarning(self._t("dialog.warning"), msg)

        self._refresh_metrics(group_key)
        self._refresh_figure(group_key)

    def _all_items(self) -> list[tuple[str, str, Path]]:
        items: list[tuple[str, str, Path]] = []
        for group_key in self.group_order:
            for time_tag in TIME_ORDER:
                path = self.group_files.get(group_key, {}).get(time_tag)
                if path is not None:
                    items.append((group_key, time_tag, path))
        return items

    def _sorted_item_keys(self, keys: set[tuple[str, str]] | list[tuple[str, str]]) -> list[tuple[str, str]]:
        group_rank = {group_key: idx for idx, group_key in enumerate(self.group_order)}
        time_rank = {time_tag: idx for idx, time_tag in enumerate(TIME_ORDER)}
        valid_keys: list[tuple[str, str]] = []
        for group_key, time_tag in list(keys):
            if time_tag not in TIME_ORDER:
                continue
            if time_tag not in self.processed_by_group.get(group_key, {}):
                continue
            valid_keys.append((group_key, time_tag))
        return sorted(
            valid_keys,
            key=lambda item: (
                group_rank.get(item[0], len(group_rank) + 1),
                time_rank.get(item[1], len(time_rank) + 1),
                item[0],
                item[1],
            ),
        )

    def _roi_for_item(self, group_key: str, time_tag: str) -> np.ndarray | None:
        key = (group_key, time_tag)
        roi = self.roi_by_item.get(key)
        if roi is None:
            return None
        return np.asarray(roi).astype(bool)

    def _remember_processing_mask_for_item(self, group_key: str, time_tag: str, mask: object) -> None:
        if mask is None:
            return
        try:
            mask_arr = np.asarray(mask).astype(bool)
        except Exception:
            return
        if mask_arr.ndim != 2:
            return
        self.processing_mask_by_item[(group_key, time_tag)] = mask_arr.copy()

    def _reprocess_item_with_roi(
        self,
        group_key: str,
        time_tag: str,
        roi_mask: np.ndarray,
    ) -> str | None:
        path = self.group_files.get(group_key, {}).get(time_tag)
        if path is None:
            return self._t("roi.reprocess.file_not_found", group=group_key, time=time_tag.upper())

        existing = self.processed_by_group.get(group_key, {}).get(time_tag)
        if existing is None:
            return self._t("roi.reprocess.no_previous_processing", group=group_key, time=time_tag.upper())

        clean_roi = np.asarray(roi_mask).astype(bool)

        base_rgb_u8 = np.asarray(existing.artifacts.base_rgb_u8, dtype=np.uint8)
        expected_shape = base_rgb_u8.shape[:2]
        if clean_roi.shape != expected_shape:
            clean_roi = resize_mask(clean_roi, expected_shape)

        key = (group_key, time_tag)
        if key not in self.processing_mask_by_item:
            self._remember_processing_mask_for_item(group_key, time_tag, existing.artifacts.mask_auto)

        contour_mask = contour_mask_with_thickness(clean_roi, radius=self.config.r_auto)
        contour_overlay = overlay_perimeter(base_rgb_u8, contour_mask, self.config.col_auto)
        mask_overlay = overlay_mask_alpha(base_rgb_u8, clean_roi, self.config.col_auto, alpha=0.35)
        mask_overlay = overlay_perimeter(mask_overlay, contour_mask, self.config.col_auto)

        area_manual_i = int(getattr(existing.results, "area_manual", self._area_reference_px))
        area_auto = int(np.count_nonzero(clean_roi))
        erro_abs = int(abs(area_auto - area_manual_i))
        erro_pct = 100.0 * float(erro_abs) / max(area_manual_i, 1)
        area_ratio = float(area_auto) / max(float(area_manual_i), 1.0)
        area_diff_norm = float(area_auto - area_manual_i) / max(float(area_manual_i), 1.0)
        results = AreaResults(
            area_manual=area_manual_i,
            area_auto=area_auto,
            erro_abs=erro_abs,
            erro_pct=float(erro_pct),
            area_ratio=float(area_ratio),
            area_diff_norm=float(area_diff_norm),
            processing_time_s=0.0,
        )

        prev_artifacts = existing.artifacts
        artifacts = CORAArtifacts(
            base_rgb_u8=base_rgb_u8,
            mask_auto=clean_roi,
            roi_mask=clean_roi,
            contour_mask=contour_mask,
            contour_overlay_rgb_u8=contour_overlay,
            mask_overlay_rgb_u8=mask_overlay,
            effective_gabor_iterations=prev_artifacts.effective_gabor_iterations,
            requested_gabor_iterations=prev_artifacts.requested_gabor_iterations,
            effective_kmeans_iter=prev_artifacts.effective_kmeans_iter,
            requested_kmeans_iter=prev_artifacts.requested_kmeans_iter,
            hat_ratio=prev_artifacts.hat_ratio,
            tex_ratio=prev_artifacts.tex_ratio,
            max_cores=prev_artifacts.max_cores,
            max_processing_time_s=prev_artifacts.max_processing_time_s,
        )

        group_results = self.processed_by_group.setdefault(group_key, {})
        group_results[time_tag] = ProcessedTimepoint(path=path, results=results, artifacts=artifacts)
        self.processing_errors.pop((group_key, time_tag), None)
        return None

    def _reprocess_item_auto(self, group_key: str, time_tag: str) -> str | None:
        path = self.group_files.get(group_key, {}).get(time_tag)
        if path is None:
            return self._t("roi.reprocess.file_not_found", group=group_key, time=time_tag.upper())

        try:
            results, artifacts = process_single_item(
                app=self,
                group_key=group_key,
                time_tag=time_tag,
                path=path,
            )
        except Exception as exc:
            return str(exc)

        group_results = self.processed_by_group.setdefault(group_key, {})
        group_results[time_tag] = ProcessedTimepoint(path=path, results=results, artifacts=artifacts)
        self._remember_processing_mask_for_item(group_key, time_tag, artifacts.mask_auto)
        self.processing_errors.pop((group_key, time_tag), None)
        return None

    # PT: Ciclo de processamento: iniciar thread, receber eventos e consolidar resultados. | EN: Processing cycle: start the thread, receive events, and consolidate results.
    def _start_processing_job(
        self,
        mode: str,
        items: list[tuple[str, str, Path]],
        save_after: bool = False,
    ) -> None:
        """Inicializa processamento em thread, configura progresso e bloqueia controles de edicao."""
        if self.ui_busy:
            return
        if not items:
            messagebox.showwarning(self._t("dialog.warning"), self._t("processing.no_images"))
            return

        self.worker_queue = queue.Queue()
        self._clear_progress_log()
        self.cancel_event.clear()
        self.active_processing_mode = mode
        self.processing_started_at = time.perf_counter()
        self.processing_total_items = len(items)
        self.progress_var.set(0.0)
        self.progress_percent_var.set("0.0%")
        self.progress_current_image_var.set("-")
        self.progress_remaining_var.set(str(len(items)))
        self.progress_eta_var.set("--")
        self._set_ui_busy(True)
        if self._robotized_test_active:
            self._track_page_transition("espera_processamento_imagem")
        if mode == "single_test":
            title = self._t("test.progress_title")
            self._single_test_resource_peaks = ResourcePeaks(None, None)
            # PT: Cada repeticao possui seu proprio monitor no worker para gerar uma linha de RAM/CPU independente no CSV.
            # EN: Each repetition has its own monitor in the worker to generate an independent RAM/CPU row in the CSV.
        elif mode == "batch_test":
            context = self._batch_test_context or {}
            title = self._t(
                "test.batch_progress_title",
                count=int(context.get("batch_size", 0)),
            )
        else:
            title = self._t("title.progress_full") if mode == "full" else self._t("title.progress_reprocess")
        self._open_progress_popup(title)
        self._sync_cancel_controls()

        if mode in ("full", "single_test", "batch_test"):
            self._set_progress_feedback(
                0.0,
                self._t("processing.full_initial", total=len(items)),
                current_image=self._t("processing.waiting_first"),
                remaining=len(items),
                eta_seconds=None,
            )
        else:
            self._set_progress_feedback(
                0.0,
                self._t("processing.reprocess_initial", total=len(items)),
                current_image=self._t("processing.waiting_first"),
                remaining=len(items),
                eta_seconds=None,
            )

        self.worker_thread = threading.Thread(
            target=run_processing_worker,
            args=(self, mode, items, save_after),
            daemon=True,
        )
        self.worker_thread.start()
        self.root.after(90, self._poll_worker_queue)

    def _poll_worker_queue(self) -> None:
        """Consome eventos da thread de processamento e reflete estado na interface."""
        while True:
            try:
                msg = self.worker_queue.get_nowait()
            except queue.Empty:
                break

            tag = msg[0]
            if tag == "item_ok":
                _, group_key, time_tag, path, results, artifacts, *extra = msg
                group_results = self.processed_by_group.setdefault(group_key, {})
                group_results[time_tag] = ProcessedTimepoint(path=path, results=results, artifacts=artifacts)
                self._remember_processing_mask_for_item(group_key, time_tag, artifacts.mask_auto)
                self.processing_errors.pop((group_key, time_tag), None)
                metrics = dict(extra[0]) if extra and isinstance(extra[0], dict) else {}
                if metrics:
                    self._processing_metrics_by_item[(group_key, time_tag)] = metrics
                if self.active_processing_mode == "single_test":
                    try:
                        base = np.asarray(artifacts.base_rgb_u8)
                        height, width = int(base.shape[0]), int(base.shape[1])
                    except Exception:
                        width = height = ""
                    run_number = int(extra[1]) if len(extra) > 1 else len(self._single_test_runs) + 1
                    self._single_test_runs.append(
                        {
                            "repeticao": run_number,
                            "executado_em": datetime.now().astimezone().isoformat(timespec="seconds"),
                            "sucesso": True,
                            "metrics": metrics,
                            "area_detectada_px": int(getattr(results, "area_auto", 0)),
                            "largura_px": width,
                            "altura_px": height,
                            "erro": "",
                        }
                    )
            elif tag == "item_error":
                _, group_key, time_tag, err, *extra = msg
                self.processing_errors[(group_key, time_tag)] = err
                if self.active_processing_mode == "single_test":
                    metrics = dict(extra[0]) if extra and isinstance(extra[0], dict) else {}
                    run_number = int(extra[1]) if len(extra) > 1 else len(self._single_test_runs) + 1
                    self._single_test_runs.append(
                        {
                            "repeticao": run_number,
                            "executado_em": datetime.now().astimezone().isoformat(timespec="seconds"),
                            "sucesso": False,
                            "metrics": metrics,
                            "area_detectada_px": "",
                            "largura_px": "",
                            "altura_px": "",
                            "erro": str(err),
                        }
                    )
            elif tag == "batch_iteration_done":
                row = dict(msg[1]) if len(msg) > 1 and isinstance(msg[1], dict) else {}
                row["executado_em"] = datetime.now().astimezone().isoformat(timespec="seconds")
                self._batch_test_rows.append(row)
            elif tag == "item_stage":
                _, mode, idx, total, group_key, time_tag, file_name, stage, stage_progress = msg
                stage_text = self._translate_processing_stage(stage)
                stage_progress = min(max(float(stage_progress), 0.0), 1.0)
                pct = 100.0 * ((float(idx - 1) + stage_progress) / float(max(total, 1)))
                prefix = (
                    self._t("processing.prefix_full")
                    if mode in ("full", "single_test", "batch_test")
                    else self._t("processing.prefix_reprocess")
                )
                elapsed = (
                    max(0.0, time.perf_counter() - float(self.processing_started_at))
                    if self.processing_started_at is not None
                    else 0.0
                )
                eta_seconds = None
                if pct > 0.1:
                    eta_seconds = elapsed * max(0.0, (100.0 - pct) / pct)
                remaining = int(math.ceil(max(0.0, float(total) - (float(idx - 1) + stage_progress))))
                image_text = f"{file_name} ({group_key} {time_tag.upper()})"
                self._set_progress_feedback(
                    pct,
                    self._t(
                        "processing.stage_status",
                        prefix=prefix,
                        group=group_key,
                        time=time_tag.upper(),
                        stage=stage_text,
                        index=idx,
                        total=total,
                    ),
                    current_image=image_text,
                    remaining=remaining,
                    eta_seconds=eta_seconds,
                )
            elif tag == "progress":
                _, mode, idx, total, group_key, time_tag, file_name = msg
                pct = 100.0 * (float(idx) / float(max(total, 1)))
                prefix = (
                    self._t("processing.prefix_full")
                    if mode in ("full", "single_test")
                    else self._t("processing.prefix_reprocess")
                )
                elapsed = (
                    max(0.0, time.perf_counter() - float(self.processing_started_at))
                    if self.processing_started_at is not None
                    else 0.0
                )
                eta_seconds = None
                if pct > 0.1 and pct < 100.0:
                    eta_seconds = elapsed * max(0.0, (100.0 - pct) / pct)
                remaining = max(total - idx, 0)
                image_text = f"{file_name} ({group_key} {time_tag.upper()})"
                self._set_progress_feedback(
                    pct,
                    self._t(
                        "processing.item_done",
                        prefix=prefix,
                        group=group_key,
                        time=time_tag.upper(),
                        index=idx,
                        total=total,
                    ),
                    current_image=image_text,
                    remaining=remaining,
                    eta_seconds=eta_seconds,
                )
            elif tag == "done":
                if len(msg) >= 6:
                    _, mode, success, failures, save_after, cancelled = msg
                else:
                    _, mode, success, failures, save_after = msg
                    cancelled = False
                self._finish_processing_job(mode, success, failures, save_after, cancelled=bool(cancelled))

        if self.ui_busy:
            self.root.after(90, self._poll_worker_queue)

    @staticmethod
    def _format_test_metric(value: object, suffix: str, decimals: int = 3) -> str:
        if value is None or value == "":
            return "indisponivel"
        try:
            return f"{float(value):.{decimals}f} {suffix}".strip()
        except (TypeError, ValueError):
            return "indisponivel"

    def _finalize_single_image_test(
        self,
        success: list[tuple[str, str]],
        failures: list[tuple[str, str, str]],
        cancelled: bool,
    ) -> None:
        """Grava uma linha por repeticao e apresenta o resumo das 30 execucoes."""
        context = self._single_test_context
        if not context:
            return

        image_path = Path(context["path"])
        expected_runs = int(context.get("repetitions", SINGLE_IMAGE_TEST_REPETITIONS))
        runs = sorted(self._single_test_runs, key=lambda run: int(run.get("repeticao", 0)))
        if not runs and failures:
            runs = [
                {
                    "repeticao": 1,
                    "executado_em": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "sucesso": False,
                    "metrics": {},
                    "area_detectada_px": "",
                    "largura_px": "",
                    "altura_px": "",
                    "erro": "\n".join(error for _group, _time, error in failures),
                }
            ]

        output_root_text = self.output_folder_var.get().strip()
        output_root = Path(output_root_text) if output_root_text else image_path.parent
        csv_path = output_root / "_cora_testes" / "testes_imagem_unica.csv"
        xlsx_path = csv_path.with_suffix(".xlsx")
        report_error = ""
        for run in runs:
            metrics = run.get("metrics", {})
            metrics = metrics if isinstance(metrics, dict) else {}
            try:
                append_test_report(
                    csv_path,
                    {
                        "executado_em": run.get("executado_em", ""),
                        "repeticao": run.get("repeticao", ""),
                        "imagem": image_path.name,
                        "caminho_imagem": str(image_path),
                        "sucesso": "sim" if bool(run.get("sucesso")) else "nao",
                        "cancelado": "sim" if cancelled else "nao",
                        "tempo_carregamento_s": metrics.get("loading_time_s", ""),
                        "tempo_segmentacao_s": metrics.get("segmentation_time_s", ""),
                        "tempo_total_pipeline_s": metrics.get("total_pipeline_time_s", ""),
                        "ram_maxima_mb": metrics.get("peak_ram_mb", ""),
                        "cpu_maxima_percent": metrics.get("peak_cpu_percent", ""),
                        "area_detectada_px": run.get("area_detectada_px", ""),
                        "largura_px": run.get("largura_px", ""),
                        "altura_px": run.get("altura_px", ""),
                        "erro": run.get("erro", ""),
                    },
                )
            except Exception as exc:
                report_error = str(exc)
                break

        if not report_error:
            try:
                export_service.write_test_report_excel(xlsx_path, read_test_report(csv_path))
            except Exception as exc:
                report_error = str(exc)

        def numeric_values(metric_key: str) -> list[float]:
            values: list[float] = []
            for run in runs:
                metrics = run.get("metrics", {})
                raw = metrics.get(metric_key) if isinstance(metrics, dict) else None
                try:
                    if raw is not None and raw != "":
                        values.append(float(raw))
                except (TypeError, ValueError):
                    continue
            return values

        loading_values = numeric_values("loading_time_s")
        segmentation_values = numeric_values("segmentation_time_s")
        total_values = numeric_values("total_pipeline_time_s")
        ram_values = numeric_values("peak_ram_mb")
        cpu_values = numeric_values("peak_cpu_percent")
        successful_runs = [run for run in runs if bool(run.get("sucesso"))]
        last_area = successful_runs[-1].get("area_detectada_px", "") if successful_runs else ""

        average = lambda values: (sum(values) / len(values)) if values else None
        self._single_test_resource_peaks = ResourcePeaks(
            max(ram_values) if ram_values else None,
            max(cpu_values) if cpu_values else None,
        )

        popup = tk.Toplevel(self.root)
        popup.title(self._t("test.report_title"))
        popup.transient(self.root)
        popup.protocol("WM_DELETE_WINDOW", popup.destroy)
        content = ttk.Frame(popup, padding=18)
        content.pack(fill=tk.BOTH, expand=True)

        ttk.Label(content, text=image_path.name, font=("TkDefaultFont", 12, "bold")).grid(
            row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 12)
        )
        values = (
            ("Status", "Cancelado" if cancelled else "Concluido"),
            ("Repeticoes finalizadas", f"{len(runs)}/{expected_runs}"),
            ("Repeticoes com sucesso", f"{len(successful_runs)}/{len(runs)}"),
            ("Carregamento medio", self._format_test_metric(average(loading_values), "s")),
            ("Segmentacao media", self._format_test_metric(average(segmentation_values), "s")),
            ("Tempo total medio", self._format_test_metric(average(total_values), "s")),
            ("Maior pico de RAM", self._format_test_metric(max(ram_values) if ram_values else None, "MB", decimals=1)),
            ("Maior pico de CPU", self._format_test_metric(max(cpu_values) if cpu_values else None, "%", decimals=1)),
            ("Area da ultima repeticao", f"{last_area} px" if last_area != "" else "indisponivel"),
        )
        for row_idx, (label, value) in enumerate(values, start=1):
            ttk.Label(content, text=f"{label}:").grid(row=row_idx, column=0, sticky=tk.W, pady=3)
            ttk.Label(content, text=value).grid(row=row_idx, column=1, sticky=tk.W, padx=(16, 0), pady=3)

        report_text = (
            f"CSV: {csv_path}\nExcel: {xlsx_path}"
            if not report_error
            else f"Falha ao salvar relatorios: {report_error}"
        )
        ttk.Label(content, text=report_text, wraplength=650, justify=tk.LEFT).grid(
            row=len(values) + 1,
            column=0,
            columnspan=2,
            sticky=tk.W,
            pady=(14, 0),
        )
        run_errors = [str(run.get("erro", "")) for run in runs if str(run.get("erro", "")).strip()]
        if run_errors:
            error_preview = "\n".join(run_errors[:3])
            ttk.Label(content, text=error_preview, wraplength=650, justify=tk.LEFT).grid(
                row=len(values) + 2,
                column=0,
                columnspan=2,
                sticky=tk.W,
                pady=(8, 0),
            )
        ttk.Button(content, text=self._t("button.close"), command=popup.destroy).grid(
            row=len(values) + 3,
            column=0,
            columnspan=2,
            sticky=tk.E,
            pady=(16, 0),
        )
        content.columnconfigure(1, weight=1)
        self._center_toplevel_on_root(popup, width=720, height=500)
        popup.lift()
        self._single_test_context = None

    def _finalize_batch_performance_test(self, cancelled: bool) -> None:
        """Persiste uma linha para cada iteracao concluida do teste de batching."""
        context = self._batch_test_context
        if not context:
            return
        batch_size = int(context.get("batch_size", 0))
        expected_iterations = int(context.get("iterations", BATCH_TEST_ITERATIONS))
        rows = sorted(self._batch_test_rows, key=lambda row: int(row.get("iteracao", 0)))

        output_root_text = self.output_folder_var.get().strip()
        folder = Path(context.get("folder", self.folder_var.get()))
        output_root = Path(output_root_text) if output_root_text else folder
        csv_path = output_root / "_cora_testes" / f"teste_batch_{batch_size}.csv"
        xlsx_path = csv_path.with_suffix(".xlsx")
        report_error = ""
        try:
            for row in rows:
                append_batch_test_report(csv_path, row)
            export_service.write_batch_test_excel(xlsx_path, read_batch_test_report(csv_path))
        except Exception as exc:
            report_error = str(exc)

        completed = len(rows)
        summary = (
            f"Lote: {batch_size} imagens\n"
            f"Iteracoes concluidas: {completed}/{expected_iterations}\n"
            f"CSV: {csv_path}\n"
            f"Excel: {xlsx_path}"
        )
        if report_error:
            messagebox.showerror(
                self._t("dialog.error"),
                f"Falha ao salvar relatorios: {report_error}\n\n{summary}",
            )
        elif cancelled or completed != expected_iterations:
            messagebox.showwarning(self._t("dialog.warning"), summary)
        else:
            messagebox.showinfo(self._t("dialog.done"), summary)
        self._batch_test_context = None

    # PT: Encerramento do ciclo de processamento (mensagens finais, estado de botoes e popups). | EN: End of the processing cycle (final messages, button states, and popups).
    def _finish_processing_job(
        self,
        mode: str,
        success: list[tuple[str, str]],
        failures: list[tuple[str, str, str]],
        save_after: bool,
        cancelled: bool = False,
    ) -> None:
        """Consolida resultados do lote, atualiza status final e opcionalmente salva saidas."""
        if self._robotized_test_active:
            self._track_page_transition("visualizacao")
        total_items = int(self.processing_total_items)
        processed_count = len(success) + len(failures)
        remaining_count = max(total_items - processed_count, 0)
        status_now = self.status_var.get()
        if cancelled:
            status_now = self._t("processing.cancelled_status", done=processed_count, total=total_items)

        self._set_ui_busy(False)
        self.active_processing_mode = None
        self.processing_started_at = None
        self.processing_total_items = 0
        self.cancel_event.clear()
        self._sync_cancel_controls()
        self._set_progress_feedback(
            (float(self.progress_var.get()) if cancelled else 100.0),
            status_now,
            current_image=self.progress_current_image_var.get(),
            remaining=(remaining_count if cancelled else 0),
            eta_seconds=(None if cancelled else 0.0),
        )

        if mode == "batch_test":
            self._close_progress_popup()
            self._finalize_batch_performance_test(cancelled=cancelled)
            return

        if mode in ("full", "single_test"):
            self.pending_reprocess = {
                (group_key, time_tag)
                for group_key, processed in self.processed_by_group.items()
                for time_tag in processed
            }
            self.reviewed_groups = set()
            self.reviewed_items = set()
            if self.group_order:
                self._show_group(self.group_order[0], mark_reviewed=True)
            else:
                self._refresh_metrics()
                self._refresh_figure()

            if cancelled:
                msg = self._t("processing.cancelled_message", done=processed_count, total=total_items)
                if failures:
                    details = "\n".join(f"{g} {t}: {e}" for g, t, e in failures[:10])
                    if len(failures) > 10:
                        details += "\n" + self._t("common.more_errors", count=len(failures) - 10)
                    msg += "\n\n" + self._t("processing.errors_registered", details=details)
                    messagebox.showwarning(self._t("processing.cancelled_with_warnings"), msg)
                else:
                    messagebox.showinfo(self._t("processing.cancelled_title"), msg)
                self._close_progress_popup()
                if mode == "single_test":
                    self._finalize_single_image_test(success, failures, cancelled=True)
                return

            if failures:
                self._set_progress_feedback(100.0, self._t("processing.completed_with_errors", count=len(failures)))
                details = "\n".join(f"{g} {t}: {e}" for g, t, e in failures[:12])
                if len(failures) > 12:
                    details += "\n" + self._t("common.more_errors", count=len(failures) - 12)
                messagebox.showwarning(self._t("processing.completed_with_warnings"), details)
            else:
                self._set_progress_feedback(100.0, self._t("processing.completed_review"))
            self._close_progress_popup()
            if mode == "single_test":
                self._finalize_single_image_test(success, failures, cancelled=False)
            return

        for key in success:
            self.pending_reprocess.discard(key)
            self.reprocessed_history.add(key)

        focus_group_key = success[0][0] if success else None
        if self.current_group_key:
            self._sync_check_vars_for_group()
            self._refresh_metrics(self.current_group_key)
            self._refresh_figure(self.current_group_key)

        save_error = ""
        out_dir = None
        if save_after and (not cancelled):
            try:
                out_dir = self._save_results_to_disk()
            except Exception as exc:
                save_error = str(exc)

        if cancelled:
            lines = [self._t("processing.reprocess_cancelled", done=processed_count, total=total_items)]
            if failures:
                lines.extend(f"{g} {t}: {e}" for g, t, e in failures[:10])
                if len(failures) > 10:
                    lines.append(self._t("common.more_errors", count=len(failures) - 10))
                messagebox.showwarning(self._t("processing.reprocess_cancelled_with_warnings"), "\n".join(lines))
            else:
                messagebox.showinfo(self._t("processing.reprocess_cancelled_title"), "\n".join(lines))
            if focus_group_key is not None:
                final_status = self.status_var.get()
                self._show_group(focus_group_key, mark_reviewed=False, status_text=final_status)
            self._close_progress_popup()
            return

        if failures or save_error:
            self._set_progress_feedback(100.0, self._t("processing.reprocess_completed_warnings_status"))
            lines = [f"{g} {t}: {e}" for g, t, e in failures[:12]]
            if len(failures) > 12:
                lines.append(self._t("common.more_errors", count=len(failures) - 12))
            if save_error:
                lines.append(self._t("processing.reprocess_save_failed", error=save_error))
            messagebox.showwarning(self._t("processing.reprocess_completed_warnings"), "\n".join(lines))
        else:
            if out_dir is not None:
                self._set_progress_feedback(100.0, self._t("processing.reprocess_saved_status", path=out_dir))
                messagebox.showinfo(self._t("dialog.done"), self._t("processing.reprocess_saved_message", path=out_dir))
            else:
                self._set_progress_feedback(100.0, self._t("processing.reprocess_completed"))

        if focus_group_key is not None:
            final_status = self.status_var.get()
            self._show_group(focus_group_key, mark_reviewed=False, status_text=final_status)
        self._close_progress_popup()

    def _start_full_processing(self) -> None:
        if not self.group_order:
            messagebox.showinfo(self._t("dialog.warning"), self._t("processing.load_groups_first"))
            return

        items = self._all_items()
        if not items:
            messagebox.showwarning(self._t("dialog.warning"), self._t("processing.no_valid_images"))
            return

        self.processed_by_group = {}
        self.processing_errors = {}
        self.processing_mask_by_item = {}
        self.pending_reprocess = set()
        self.reprocessed_history = set()
        self.reviewed_groups = set()
        self.reviewed_items = set()
        self.no_area_items = set()
        self.no_area_backup = {}
        self._robot_save_selection_index = 0
        self._sync_check_vars_for_group()

        self._start_processing_job(mode="full", items=items, save_after=False)

    def _unselected_times_in_current_group(self) -> list[str]:
        group_key = self.current_group_key
        if group_key is None:
            return []

        processed = self.processed_by_group.get(group_key, {})
        unselected = []
        for time_tag in TIME_ORDER:
            if (not self.refazer_vars[time_tag].get()) and time_tag in processed:
                unselected.append(time_tag)
        return unselected

    def _redefine_roi_for_keys(
        self,
        keys: list[tuple[str, str]],
        title_prefix: str = "",
        require_all: bool = False,
    ) -> tuple[list[tuple[str, str]], bool]:
        updated: list[tuple[str, str]] = []
        staged_rois: dict[tuple[str, str], np.ndarray] = {}
        aborted = False
        valid_keys: list[tuple[str, str]] = []
        for group_key, time_tag in keys:
            proc = self.processed_by_group.get(group_key, {}).get(time_tag)
            if proc is None:
                self.processing_errors[(group_key, time_tag)] = self._t("roi.redefine.no_previous_processing")
                continue
            valid_keys.append((group_key, time_tag))

        total = len(valid_keys)
        next_index = 0

        def editor_payload(index: int) -> dict[str, object]:
            group_key, time_tag = valid_keys[index]
            proc = self.processed_by_group[group_key][time_tag]
            key = (group_key, time_tag)
            seed_roi = self.roi_by_item.get(key)
            if seed_roi is None:
                seed_roi = proc.artifacts.mask_auto
            else:
                seed_roi = resize_mask(seed_roi, proc.artifacts.base_rgb_u8.shape[:2])
            original_rgb = self._read_original_preview_image(proc.path)
            title = str(proc.path.stem)
            if total > 1:
                title = f"{title} [{index + 1}/{total}]"
            return {
                "image": proc.artifacts.base_rgb_u8,
                "original_image": original_rgb,
                "initial_mask": seed_roi,
                "title": title,
                "image_cache_key": f"review::{self._image_cache_key(proc.path)}",
            }

        def accept_and_advance(new_roi: np.ndarray) -> dict[str, object] | None:
            nonlocal next_index
            key = valid_keys[next_index]
            staged_rois[key] = np.asarray(new_roi).astype(bool)
            updated.append(key)
            next_index += 1
            if next_index >= total:
                return None
            return editor_payload(next_index)

        while next_index < total:
            payload = editor_payload(next_index)
            original_payload = payload.get("original_image")
            new_roi = self._open_roi_editor(
                image=np.asarray(payload["image"]),
                original_image=(None if original_payload is None else np.asarray(original_payload)),
                initial_mask=np.asarray(payload["initial_mask"]),
                title=str(payload["title"]),
                image_cache_key=str(payload["image_cache_key"]),
                on_accept_next=accept_and_advance,
            )
            if new_roi is not None:
                continue
            if not require_all:
                break

            group_key, time_tag = valid_keys[next_index]
            retry = messagebox.askyesno(
                self._t("roi.redefine.retry_title"),
                self._t("roi.redefine.retry_message", group=group_key, time=time_tag.upper()),
            )
            if not retry:
                aborted = True
                break

        if aborted:
            return [], True

        if not updated:
            return [], False

        total_recalc = len(updated)
        applied: list[tuple[str, str]] = []
        failures: list[tuple[str, str, str]] = []
        refresh_current_group = False
        title = self._t("roi.apply.title_single" if total_recalc == 1 else "roi.apply.title_plural")
        self._clear_progress_log()
        if self._robotized_test_active:
            self._track_page_transition("espera_processamento_imagem")
        self._open_progress_popup(title)
        self._set_progress_feedback(0.0, self._t("roi.apply.preparing", total=total_recalc))
        self._set_ui_busy(True)
        try:
            for idx, key in enumerate(updated, start=1):
                group_key, time_tag = key
                clean_roi = np.asarray(staged_rois[key]).astype(bool)
                self._set_progress_feedback(
                    100.0 * (float(idx - 1) / float(max(total_recalc, 1))),
                    self._t(
                        "roi.apply.applying",
                        group=group_key,
                        time=time_tag.upper(),
                        index=idx,
                        total=total_recalc,
                    ),
                )

                reprocess_error = self._reprocess_item_with_roi(
                    group_key=group_key,
                    time_tag=time_tag,
                    roi_mask=clean_roi,
                )
                if reprocess_error:
                    failures.append((group_key, time_tag, reprocess_error))
                    self.processing_errors[(group_key, time_tag)] = reprocess_error
                    self._append_progress_log(
                        self._t("roi.apply.failed_log", group=group_key, time=time_tag.upper(), error=reprocess_error),
                        pct=100.0 * (float(idx) / float(max(total_recalc, 1))),
                        force=True,
                    )
                    continue

                self.roi_by_item[key] = clean_roi
                self.no_area_items.discard(key)
                self.no_area_backup.pop(key, None)
                self.pending_reprocess.discard(key)
                self.reprocessed_history.add(key)
                applied.append(key)

                if self.current_group_key == group_key:
                    refresh_current_group = True

                self._set_progress_feedback(
                    100.0 * (float(idx) / float(max(total_recalc, 1))),
                    self._t(
                        "roi.apply.done",
                        group=group_key,
                        time=time_tag.upper(),
                        index=idx,
                        total=total_recalc,
                    ),
                )
        finally:
            self._set_ui_busy(False)
            self._close_progress_popup()
            if self._robotized_test_active:
                self._track_page_transition("visualizacao")

        if refresh_current_group and self.current_group_key is not None:
            # PT: Todas as ROIs do grupo ja foram aplicadas: um unico redesenho. | EN: All ROIs in the group have already been applied, so only one redraw is needed.
            self._sync_check_vars_for_group()
            self._refresh_metrics(self.current_group_key)
            self._refresh_figure(self.current_group_key)

        if failures:
            details = "\n".join(f"{g} {t.upper()}: {e}" for g, t, e in failures[:8])
            if len(failures) > 8:
                details += "\n" + self._t("common.more_errors", count=len(failures) - 8)
            messagebox.showwarning(
                self._t("dialog.warning"),
                self._t("roi.apply.failures_warning", details=details),
            )

        return applied, False

    def _redefine_roi_for_unselected(self) -> None:
        if not self.group_order:
            messagebox.showinfo(self._t("dialog.warning"), self._t("roi.edit.load_groups_first"))
            return
        if not self.processed_by_group:
            messagebox.showinfo(self._t("dialog.warning"), self._t("roi.edit.process_first"))
            return

        keys = self._sorted_item_keys(self.pending_reprocess)
        if not keys:
            messagebox.showinfo(
                self._t("dialog.warning"),
                self._t("roi.edit.no_unselected"),
            )
            return

        missing_views = [key for key in keys if key not in self.reviewed_items]
        if missing_views:
            preview = ", ".join(f"{g} {t.upper()}" for g, t in missing_views[:8])
            if len(missing_views) > 8:
                preview += f", ... +{len(missing_views) - 8}"
            messagebox.showwarning(
                self._t("roi.edit.incomplete_review_title"),
                self._t("roi.edit.incomplete_review_message", items=preview),
            )
            return

        confirm = messagebox.askyesno(
            self._t("roi.edit.queue_title"),
            self._t("roi.edit.queue_message", count=len(keys)),
        )
        if not confirm:
            self.status_var.set(self._t("roi.edit.queue_cancelled_status"))
            return

        updated_keys, aborted = self._redefine_roi_for_keys(
            keys,
            title_prefix=self._t("roi.redefine.retry_title"),
            require_all=True,
        )
        if aborted and not updated_keys:
            self.status_var.set(self._t("roi.edit.cancelled_status"))
            return
        updated = len(updated_keys)

        if updated < len(keys):
            missing = len(keys) - updated
            messagebox.showwarning(
                self._t("dialog.warning"),
                self._t("roi.edit.pending_without_redefinition", count=missing),
            )

        if self.current_group_key:
            self._refresh_metrics(self.current_group_key)
            self._refresh_figure(self.current_group_key)

        if updated > 0:
            self.status_var.set(self._t("roi.edit.updated_status", count=updated))
        else:
            self.status_var.set(self._t("roi.edit.no_confirmed_status"))

    def _clear_roi_for_unselected(self) -> None:
        group_key = self.current_group_key
        if group_key is None:
            messagebox.showinfo(self._t("dialog.warning"), self._t("roi.restore.no_group"))
            return

        unselected_times = self._unselected_times_in_current_group()
        if not unselected_times:
            messagebox.showinfo(
                self._t("dialog.warning"),
                self._t("roi.restore.no_unselected"),
            )
            return

        times_text = ", ".join(t.upper() for t in unselected_times)
        confirm = messagebox.askyesno(
            self._t("roi.restore.confirm_title"),
            self._t("roi.restore.confirm_message", group=group_key, times=times_text),
        )
        if not confirm:
            self.status_var.set(self._t("roi.restore.cancelled_status"))
            return

        keys = [(group_key, time_tag) for time_tag in unselected_times]
        removed_custom = 0
        for key in keys:
            if key in self.roi_by_item:
                del self.roi_by_item[key]
                removed_custom += 1
            self.reprocessed_history.discard(key)
            self.pending_reprocess.add(key)

        total = len(keys)
        restored = 0
        failures: list[tuple[str, str, str]] = []

        self._clear_progress_log()
        self._open_progress_popup(self._t("roi.restore.progress_title"))
        self._set_progress_feedback(0.0, self._t("roi.restore.progress_initial", total=total))
        self._set_ui_busy(True)
        try:
            for idx, (gk, tt) in enumerate(keys, start=1):
                self._set_progress_feedback(
                    100.0 * (float(idx - 1) / float(max(total, 1))),
                    self._t("roi.restore.progress_item", group=gk, time=tt.upper(), index=idx, total=total),
                )
                err = self._reprocess_item_auto(gk, tt)
                if err:
                    failures.append((gk, tt, err))
                    self.processing_errors[(gk, tt)] = err
                    self._append_progress_log(
                        self._t("roi.restore.failed_log", group=gk, time=tt.upper(), error=err),
                        pct=100.0 * (float(idx) / float(max(total, 1))),
                        force=True,
                    )
                    continue
                restored += 1
                self.pending_reprocess.add((gk, tt))
                self._set_progress_feedback(
                    100.0 * (float(idx) / float(max(total, 1))),
                    self._t("roi.restore.progress_done", group=gk, time=tt.upper(), index=idx, total=total),
                )
        finally:
            self._set_ui_busy(False)
            self._close_progress_popup()

        self._sync_check_vars_for_group()
        self._refresh_metrics(group_key)
        self._refresh_figure(group_key)

        if failures:
            details = "\n".join(f"{g} {t.upper()}: {e}" for g, t, e in failures[:8])
            if len(failures) > 8:
                details += "\n" + self._t("common.more_errors", count=len(failures) - 8)
            self.status_var.set(self._t("roi.restore.failures_status", removed=removed_custom, restored=restored, total=total))
            messagebox.showwarning(
                self._t("roi.restore.failures_title"),
                self._t("roi.restore.failures_message", details=details),
            )
            return

        self.status_var.set(self._t("roi.restore.done_status", removed=removed_custom, restored=restored))

    def _save_results_clicked(self) -> None:
        if not self.processed_by_group:
            messagebox.showinfo(self._t("dialog.warning"), self._t("save.no_results"))
            return
        measured_at, page_time_rows = self._page_time_snapshot()
        try:
            out_dir = self._save_results_to_disk()
            self._write_page_times_csv(out_dir, measured_at, page_time_rows)
        except Exception as exc:
            messagebox.showerror(self._t("dialog.error"), self._t("save.failed", error=exc))
            return
        self.status_var.set(self._t("save.saved_status", path=out_dir))
        messagebox.showinfo(self._t("dialog.saved"), self._t("save.saved_message", path=out_dir))

    def _button_help_specs(self) -> list[tuple[str, str, str]]:
        return [
            ("browse_btn", self._t("button.browse_folder"), self._t("help.browse_folder")),
            ("browse_output_btn", self._t("button.choose_output"), self._t("help.browse_output")),
            (
                "scan_btn",
                self._t("button.load_groups"),
                self._t("help.load_groups"),
            ),
            (
                "open_config_btn",
                self._t("menu.configurations_button"),
                self._t("help.settings"),
            ),
            (
                "theme_toggle_btn",
                self._ui_theme_toggle_label(),
                self._t("help.theme"),
            ),
            (
                "language_toggle_btn_config",
                self._ui_language_toggle_label(),
                self._t("help.language"),
            ),
            ("viewer_guide_btn", self._t("button.guide"), self._t("help.guide")),
            (
                "contour_settings_btn",
                self._t("button.configure_contours"),
                self._t("help.contours"),
            ),
            (
                "compare_area_overlay_check",
                self._t("button.compare_areas"),
                self._t("help.compare_areas"),
            ),
            ("toggle_results_btn", self._results_toggle_label(), self._t("help.toggle_results")),
            ("prev_btn", self._t("button.prev_group"), self._t("help.prev_group")),
            ("next_btn", self._t("button.next_group"), self._t("help.next_group")),
            (
                "redefine_roi_btn",
                self._t("button.edit_masks"),
                self._t("help.edit_masks"),
            ),
            (
                "clear_roi_btn",
                self._t("button.restore_auto"),
                self._t("help.restore_auto"),
            ),
            ("save_btn", self._t("button.save_result"), self._t("help.save_result")),
            ("cancel_btn", self._t("button.cancel_processing"), self._t("help.cancel_processing")),
        ]

    def _button_help_lines(self) -> list[str]:
        def btn_text(attr_name: str, fallback: str) -> str:
            btn = getattr(self, attr_name, None)
            if btn is None:
                return fallback
            try:
                value = str(btn.cget("text")).strip()
            except Exception:
                return fallback
            return value or fallback

        lines = [
            self._t("guide.title"),
            self._t("guide.hover"),
            "",
            self._t("guide.flow"),
            self._t("guide.step1"),
            self._t("guide.step2"),
            self._t("guide.step3"),
            self._t("guide.step4"),
            "",
            self._t("guide.buttons"),
        ]
        for attr_name, fallback, desc in self._button_help_specs():
            lines.append(f"{btn_text(attr_name, fallback)}: {desc}")
        lines.extend(
            [
                "",
                self._t("guide.group_review"),
                self._t("guide.review1"),
                self._t("guide.review2"),
                self._t("guide.review3"),
            ]
        )
        return lines

    def _refresh_button_tooltips(self) -> None:
        for attr_name, fallback, desc in self._button_help_specs():
            widget = getattr(self, attr_name, None)
            if widget is None:
                continue
            try:
                label = str(widget.cget("text")).strip() or fallback
            except Exception:
                label = fallback
            self._register_tooltip(widget, f"{label}: {desc}")
        self._register_tooltip(self.config_guide_btn, self._t("tooltip.guide"))
        self._register_tooltip(
            self.theme_toggle_btn_config,
            self._t("tooltip.theme"),
        )
        self._register_tooltip(self.language_toggle_btn_config, self._t("tooltip.language"))

    def _refresh_help_tab(self) -> None:
        lines = self._button_help_lines()
        self._guide_text_cache = "\n".join(lines).rstrip()
        self._refresh_button_tooltips()
        text = self.guide_popup_text
        if text is None or (not text.winfo_exists()):
            return
        text.configure(state=tk.NORMAL)
        text.delete("1.0", tk.END)
        text.insert(tk.END, self._guide_text_cache)
        text.configure(state=tk.DISABLED)

    @staticmethod
    def _safe_filename(name: str, max_len: int = 120) -> str:
        return export_service.safe_filename(name, max_len=max_len)

    @staticmethod
    def _write_image_file(path: Path, image: np.ndarray) -> bool:
        return export_service.write_image_file(path, image)

    def _write_results_excel(self, xlsx_path: Path, rows: list[dict[str, object]]) -> None:
        export_service.write_results_excel(xlsx_path=xlsx_path, rows=rows, group_order=self.group_order)

    def _area_auto_from_processed(self, group_key: str, time_tag: str) -> float | None:
        proc = self.processed_by_group.get(group_key, {}).get(time_tag)
        if proc is None:
            return None
        try:
            return float(proc.results.area_auto)
        except Exception:
            return None

    def _closure_vs_0h(self, group_key: str, time_tag: str) -> tuple[float | None, float | None]:
        area_base = self._area_auto_from_processed(group_key, "0h")
        area_ref = self._area_auto_from_processed(group_key, time_tag)
        if area_base is None or area_ref is None:
            return None, None
        fechamento_px = float(area_base - area_ref)
        if area_base <= 0.0:
            return fechamento_px, None
        fechamento_pct = 100.0 * fechamento_px / area_base
        return fechamento_px, fechamento_pct

    def _save_results_to_disk(self) -> Path:
        output_folder_text = self.output_folder_var.get().strip()
        if output_folder_text:
            out_dir = Path(output_folder_text).expanduser()
            if out_dir.exists() and not out_dir.is_dir():
                raise RuntimeError(f"Pasta de saida invalida (nao e diretorio): {out_dir}")
        else:
            folder_text = self.folder_var.get().strip()
            if not folder_text:
                raise RuntimeError("Pasta base nao definida para salvar resultados.")
            out_dir = Path(folder_text) / OUTPUT_DIR_NAME

        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "resumo_resultados.csv"
        xlsx_path = out_dir / "resumo_resultados.xlsx"

        headers = [
            "grupo",
            "tempo",
            "arquivo_nome",
            "arquivo_origem",
            "status_revisao",
            "sem_area_marcada",
            "area_auto_px",
            "fechamento_area_px_vs_0h",
            "fechamento_area_pct_vs_0h",
            "tempo_processamento_s",
            "overlay_png",
            "erro",
        ]

        rows: list[dict[str, object]] = []
        for group_key in self.group_order:
            group_processed = self.processed_by_group.get(group_key, {})
            for time_tag in self._group_times(group_key):
                key = (group_key, time_tag)
                src_path = self.group_files[group_key][time_tag]
                proc = group_processed.get(time_tag)
                err = self.processing_errors.get(key, "")

                overlay_name = ""
                area_auto: object = ""
                fechamento_px: object = ""
                fechamento_pct: object = ""
                proc_time: object = ""

                if proc is not None:
                    token = self._safe_filename(f"{group_key}_{time_tag}_{src_path.stem}")
                    overlay_name = f"{token}_overlay.png"

                    overlay_rgb = self._build_filled_overlay_with_selected_color(proc, time_tag)
                    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

                    overlay_path = out_dir / overlay_name

                    if not self._write_image_file(overlay_path, overlay_bgr):
                        raise RuntimeError(self._t("save.write_failed", name=overlay_name, path=overlay_path))

                    area_auto = int(round(float(proc.results.area_auto)))
                    proc_time = float(proc.results.processing_time_s)
                    clos_px, clos_pct = self._closure_vs_0h(group_key, time_tag)
                    if clos_px is not None:
                        fechamento_px = float(clos_px)
                    if clos_pct is not None:
                        fechamento_pct = float(clos_pct)

                if key in self.no_area_items:
                    status_revisao = "sem_area_zerada"
                elif key in self.pending_reprocess:
                    status_revisao = "nao_selecionada_para_salvar"
                elif key in self.reprocessed_history:
                    status_revisao = "roi_substituida"
                else:
                    status_revisao = "selecionada_para_salvar"

                rows.append(
                    {
                        "grupo": group_key,
                        "tempo": time_tag,
                        "arquivo_nome": src_path.name,
                        "arquivo_origem": str(src_path),
                        "status_revisao": status_revisao,
                        "sem_area_marcada": ("SIM" if key in self.no_area_items else "NAO"),
                        "area_auto_px": area_auto,
                        "fechamento_area_px_vs_0h": fechamento_px,
                        "fechamento_area_pct_vs_0h": fechamento_pct,
                        "tempo_processamento_s": proc_time,
                        "overlay_png": overlay_name,
                        "erro": err,
                    }
                )

        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers, delimiter=";")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        self._write_results_excel(xlsx_path=xlsx_path, rows=rows)

        return out_dir

    def _refresh_metrics(self, group_key: str | None = None) -> None:
        """Atualiza painel textual de resultados com o estado do grupo corrente."""
        self.metrics_text.configure(state=tk.NORMAL)
        self.metrics_text.delete("1.0", tk.END)

        if group_key is None:
            group_key = self.current_group_key
        if group_key is None:
            self.metrics_text.insert(tk.END, self._t("metrics.no_group"))
            self.metrics_text.configure(state=tk.DISABLED)
            return

        lines = [
            self._t("metrics.header"),
            f"{self._t('review.group')}: {group_key}",
            f"{self._t('metrics.reviewed_groups')}: {len(self.reviewed_groups)}/{len(self.group_order)}",
            "",
        ]

        group_processed = self.processed_by_group.get(group_key, {})
        group_times = self._group_times(group_key)
        if not group_times:
            lines.append(self._t("metrics.no_images_group"))
        else:
            for time_tag in group_times:
                key = (group_key, time_tag)
                proc = group_processed.get(time_tag)
                selected_for_save = self._t("yes") if key not in self.pending_reprocess else self._t("no")
                pending_roi = self._t("yes") if key in self.pending_reprocess else self._t("no")
                custom_roi = self._t("yes") if key in self.roi_by_item else self._t("no")
                no_area = self._t("yes") if key in self.no_area_items else self._t("no")

                lines.append(f"[{time_tag.upper()}] {self.group_files[group_key][time_tag].name}")
                lines.append(f"{self._t('metrics.selected_save')}: {selected_for_save}")
                lines.append(f"{self._t('metrics.pending_roi')}: {pending_roi}")
                lines.append(f"{self._t('metrics.custom_roi')}: {custom_roi}")
                lines.append(f"{self._t('metrics.no_area_marked')}: {no_area}")

                if proc is None:
                    err = self.processing_errors.get(key, self._t("metrics.not_processed"))
                    lines.append(f"{self._t('metrics.status')}: {self._t('metrics.no_result')} ({err})")
                    lines.append("")
                    continue

                lines.append(f"{self._t('metrics.auto_area')}: {proc.results.area_auto:.0f} px")
                fechamento_px, fechamento_pct = self._closure_vs_0h(group_key, time_tag)
                if fechamento_px is None:
                    lines.append(f"{self._t('metrics.closure_vs_0h')}: n/d")
                elif fechamento_pct is None:
                    lines.append(f"{self._t('metrics.closure_vs_0h')}: {fechamento_px:+.0f} px")
                else:
                    lines.append(f"{self._t('metrics.closure_vs_0h')}: {fechamento_px:+.0f} px | {fechamento_pct:+.2f}%")
                lines.append(f"{self._t('metrics.processing_time')}: {proc.results.processing_time_s:.3f} s")
                lines.append("")

        self.metrics_text.insert(tk.END, "\n".join(lines).rstrip())
        self.metrics_text.configure(state=tk.DISABLED)

    def _refresh_figure(self, group_key: str | None = None) -> None:
        """Redesenha o mosaico de imagens/overlays do grupo selecionado."""
        self.figure.clear()
        if group_key is None:
            group_key = self.current_group_key

        if group_key is None or group_key not in self.group_files:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, self._t("figure.no_group"), ha="center", va="center")
            ax.axis("off")
            self.canvas.draw_idle()
            return

        group_times = self._group_times(group_key)
        if not group_times:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, self._t("figure.no_images_group"), ha="center", va="center")
            ax.axis("off")
            self.canvas.draw_idle()
            return

        group_processed = self.processed_by_group.get(group_key, {})
        ncols = len(group_times)
        for idx, time_tag in enumerate(group_times, start=1):
            ax = self.figure.add_subplot(1, ncols, idx)
            key = (group_key, time_tag)
            proc = group_processed.get(time_tag)

            if proc is not None:
                preview_rgb = self._build_preview_image(proc, group_key, time_tag)
                ax.imshow(preview_rgb)
                if self._compare_area_masks(proc, group_key, time_tag, tuple(np.asarray(preview_rgb).shape[:2])) is not None:
                    self._draw_compare_area_legend(ax)
                title = f"{time_tag.upper()} | {self._t('figure.area')}={proc.results.area_auto:.0f} px"
                fechamento_px, fechamento_pct = self._closure_vs_0h(group_key, time_tag)
                if fechamento_pct is not None and time_tag != "0h":
                    title += f" | {self._t('figure.closure')}={fechamento_pct:+.1f}%"
                if key in self.pending_reprocess:
                    title += f" | {self._t('figure.pending_roi')}"
                if key in self.no_area_items:
                    title += f" | {self._t('figure.no_area')}"
                    ax.text(
                        0.5,
                        0.07,
                        self._t("figure.no_area_zeroed"),
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white",
                        bbox={"facecolor": "black", "alpha": 0.65, "pad": 3},
                    )
                ax.set_title(title, fontsize=10)
            else:
                err = self.processing_errors.get(key, "")
                src_path = self.group_files.get(group_key, {}).get(time_tag)
                if (not err) and (src_path is not None):
                    try:
                        preview_base = self._preview_base_for_path(src_path)
                        if preview_base is None:
                            raise ValueError(self._t("figure.no_preview"))
                        ax.imshow(preview_base)
                        image_code = src_path.stem.strip() or src_path.name
                        ax.set_title(f"{time_tag.upper()} | {image_code}", fontsize=10)
                    except Exception:
                        ax.text(0.5, 0.5, f"{time_tag.upper()}\n{self._t('figure.no_result')}", ha="center", va="center", wrap=True)
                        ax.set_title(time_tag.upper(), fontsize=10)
                else:
                    msg = err or self._t("figure.no_result")
                    ax.text(0.5, 0.5, f"{time_tag.upper()}\n{msg}", ha="center", va="center", wrap=True)
                    ax.set_title(time_tag.upper(), fontsize=10)

            ax.axis("off")

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _build_roi_editor_enhanced_image(self, image_rgb_u8: np.ndarray) -> np.ndarray:
        return build_roi_editor_enhanced_image_page(self, image_rgb_u8)

    def _roi_editor_window_from_figure(self, fig: object) -> object | None:
        canvas = getattr(fig, "canvas", None)
        manager = getattr(canvas, "manager", None)
        return getattr(manager, "window", None)

    def _capture_roi_editor_window_state(self, fig: object) -> None:
        window = self._roi_editor_window_from_figure(fig)
        state = {"fullscreen": False, "zoomed": False}
        if window is None:
            self._roi_editor_window_state = state
            return

        for method_name in ("isFullScreen", "is_fullscreen"):
            method = getattr(window, method_name, None)
            if callable(method):
                try:
                    state["fullscreen"] = bool(method())
                    break
                except Exception:
                    pass

        if not state["fullscreen"]:
            for method_name in ("wm_attributes", "attributes"):
                method = getattr(window, method_name, None)
                if callable(method):
                    try:
                        state["fullscreen"] = bool(method("-fullscreen"))
                        break
                    except Exception:
                        pass

        for method_name in ("isMaximized", "is_maximized"):
            method = getattr(window, method_name, None)
            if callable(method):
                try:
                    state["zoomed"] = bool(method())
                    break
                except Exception:
                    pass

        if not state["zoomed"]:
            for method_name in ("state", "wm_state"):
                method = getattr(window, method_name, None)
                if callable(method):
                    try:
                        state["zoomed"] = str(method()).lower() == "zoomed"
                        break
                    except Exception:
                        pass

        self._roi_editor_window_state = state

    def _apply_roi_editor_window_state(self, fig: object) -> None:
        state = getattr(self, "_roi_editor_window_state", {"fullscreen": False, "zoomed": False})
        if not (bool(state.get("fullscreen")) or bool(state.get("zoomed"))):
            return

        window = self._roi_editor_window_from_figure(fig)
        if window is None:
            return

        if bool(state.get("fullscreen")):
            for method_name in ("showFullScreen", "show_full_screen"):
                method = getattr(window, method_name, None)
                if callable(method):
                    try:
                        method()
                        return
                    except Exception:
                        pass
            for method_name in ("wm_attributes", "attributes"):
                method = getattr(window, method_name, None)
                if callable(method):
                    try:
                        method("-fullscreen", True)
                        return
                    except Exception:
                        pass
            return

        if bool(state.get("zoomed")):
            for method_name in ("showMaximized", "show_maximized"):
                method = getattr(window, method_name, None)
                if callable(method):
                    try:
                        method()
                        return
                    except Exception:
                        pass
            for method_name in ("state", "wm_state"):
                method = getattr(window, method_name, None)
                if callable(method):
                    try:
                        method("zoomed")
                        return
                    except Exception:
                        pass

    def _open_roi_editor(
        self,
        image: np.ndarray,
        initial_mask: np.ndarray,
        title: str,
        original_image: np.ndarray | None = None,
        image_cache_key: str | None = None,
        on_accept_next=None,
    ) -> np.ndarray | None:
        """Edita uma ou mais ROIs na mesma janela e retorna a ultima mascara aceita."""
        try:
            plt = _get_pyplot()
        except Exception as exc:
            messagebox.showerror(
                self._t("roi.editor.unavailable_title"),
                self._t("roi.editor.unavailable_message", error=exc),
            )
            return None

        fig, ax = plt.subplots(figsize=(12, 9))
        manager = getattr(fig.canvas, "manager", None)
        if manager is not None:
            # PT: Desativa atalhos padrao do Matplotlib (ex.: "p" para pan), para manter os atalhos do editor sob controle exclusivo.
            # EN: Disables default Matplotlib shortcuts (for example, "p" for pan) so the editor retains exclusive control of its shortcuts.
            default_key_handler_cid = getattr(manager, "key_press_handler_id", None)
            if isinstance(default_key_handler_cid, int):
                try:
                    fig.canvas.mpl_disconnect(default_key_handler_cid)
                except Exception:
                    pass
        image_rgb_u8 = np.asarray(image, dtype=np.uint8)
        if image_rgb_u8.ndim == 2:
            image_rgb_u8 = cv2.cvtColor(image_rgb_u8, cv2.COLOR_GRAY2RGB)
        elif image_rgb_u8.ndim == 3 and image_rgb_u8.shape[2] >= 3:
            image_rgb_u8 = image_rgb_u8[:, :, :3]
        else:
            raise ValueError(self._t("roi.editor.invalid_image"))

        # PT: Editor alterna entre imagem original de entrada e versao processada de maior definicao. | EN: The editor switches between the original input image and the higher-definition processed version.
        try:
            if original_image is None:
                original_rgb_u8 = np.asarray(image_rgb_u8, dtype=np.uint8)
            else:
                original_rgb_u8 = self._to_rgb_u8(np.asarray(original_image))
        except Exception:
            original_rgb_u8 = np.asarray(image_rgb_u8, dtype=np.uint8)
        if tuple(original_rgb_u8.shape[:2]) != tuple(image_rgb_u8.shape[:2]):
            target_h, target_w = image_rgb_u8.shape[:2]
            src_h, src_w = original_rgb_u8.shape[:2]
            interpolation = cv2.INTER_AREA if (src_h >= target_h and src_w >= target_w) else cv2.INTER_LINEAR
            original_rgb_u8 = cv2.resize(
                np.asarray(original_rgb_u8, dtype=np.uint8),
                (target_w, target_h),
                interpolation=interpolation,
            )

        orig_shape = (int(image_rgb_u8.shape[0]), int(image_rgb_u8.shape[1]))
        active_profile_key = ROI_EDITOR_PROFILE_DEFAULT
        profile_cfg = ROI_EDITOR_PROFILE_PRESETS[ROI_EDITOR_PROFILE_DEFAULT]
        editor_scale = float(np.clip(float(profile_cfg.get("work_scale", ROI_EDITOR_WORK_SCALE)), 0.1, 1.0))
        profile_label = self._roi_editor_profile_label(active_profile_key)
        render_tuning: dict[str, object] = {
            "work_scale": editor_scale,
            "editor_max_fps": max(1.0, float(profile_cfg.get("editor_max_fps", ROI_EDITOR_MAX_FPS))),
            "cursor_max_fps": max(1.0, float(profile_cfg.get("cursor_max_fps", ROI_CURSOR_MAX_FPS))),
        }
        if editor_scale < 0.999:
            work_h = max(1, int(round(orig_shape[0] * editor_scale)))
            work_w = max(1, int(round(orig_shape[1] * editor_scale)))
            image_work_u8 = cv2.resize(image_rgb_u8, (work_w, work_h), interpolation=cv2.INTER_AREA)
            image_original_work_u8 = cv2.resize(original_rgb_u8, (work_w, work_h), interpolation=cv2.INTER_AREA)
        else:
            image_work_u8 = image_rgb_u8
            image_original_work_u8 = original_rgb_u8

        image = image_work_u8
        try:
            enhanced_full_u8 = self._best_defined_image(
                original_rgb_u8,
                cache_key=(str(image_cache_key or "").strip() or None),
            )
        except Exception:
            enhanced_full_u8 = np.asarray(original_rgb_u8, dtype=np.uint8)

        if editor_scale < 0.999:
            image_enhanced_u8 = cv2.resize(
                np.asarray(enhanced_full_u8, dtype=np.uint8),
                (work_w, work_h),
                interpolation=cv2.INTER_AREA,
            )
        else:
            image_enhanced_u8 = np.asarray(enhanced_full_u8, dtype=np.uint8)
        view_state: dict[str, str] = {"mode": "original"}
        view_label_map = {
            "original": self._preview_source_mode_label("original"),
            "enhanced": self._preview_source_mode_label("enhanced"),
        }

        base_image_artist = ax.imshow(image_original_work_u8)
        ax.set_title(str(title))
        ax.axis("off")

        selected: dict[str, object] = {"verts": None, "accepted": False, "result_mask": None}

        def _remember_window_state() -> None:
            self._capture_roi_editor_window_state(fig)

        current_mask = resize_mask(np.asarray(initial_mask).astype(bool), image.shape[:2])
        roi_state: dict[str, object] = {
            "mask_u8": (current_mask.astype(np.uint8) * 255),
            "mask_cache": current_mask,
        }
        selector_state: dict[str, object | None] = {"obj": None}
        local_drag_state: dict[str, object] = {
            "active": False,
            "anchor_idx": -1,
            "start_xy": None,
            "base_verts": None,
            "last_apply_ts": 0.0,
        }
        brush_state: dict[str, object] = {
            "radius": int(ROI_BRUSH_RADIUS_DEFAULT),
            "show_cursor": True,
            "cursor_xy": (0.0, 0.0),
            "cursor_in_axes": False,
        }
        brush_drag_state: dict[str, object] = {
            "active": False,
            "last_xy": None,
            "mode": "brush",
            "mask_before": None,
        }
        history_state: dict[str, list[np.ndarray]] = {"undo": [current_mask.copy()], "redo": []}
        tool_state: dict[str, str] = {"mode": "brush"}
        render_state: dict[str, float] = {
            "last_contour": 0.0,
            "last_cursor": 0.0,
        }
        render_batch_state: dict[str, object] = {
            "scheduled": False,
            "timer": None,
        }
        mask_overlay_state: dict[str, str] = {
            "mode": self._normalize_preview_mode_key(self.preview_mode_key),
        }

        def _request_canvas_draw(immediate: bool = False) -> None:
            timer = render_batch_state.get("timer")
            if immediate:
                if timer is not None:
                    try:
                        timer.stop()
                    except Exception:
                        pass
                    render_batch_state["timer"] = None
                render_batch_state["scheduled"] = False
                try:
                    fig.canvas.draw_idle()
                except Exception:
                    pass
                return

            if bool(render_batch_state.get("scheduled")):
                return

            try:
                interval_ms = max(8, int(round(1000.0 / max(float(render_tuning["editor_max_fps"]), 1.0))))
                timer = fig.canvas.new_timer(interval=interval_ms)
            except Exception:
                try:
                    fig.canvas.draw_idle()
                except Exception:
                    pass
                return

            def _flush_canvas_draw() -> bool:
                render_batch_state["scheduled"] = False
                render_batch_state["timer"] = None
                try:
                    fig.canvas.draw_idle()
                except Exception:
                    pass
                return False

            if hasattr(timer, "single_shot"):
                try:
                    timer.single_shot = True
                except Exception:
                    pass

            render_batch_state["scheduled"] = True
            render_batch_state["timer"] = timer
            timer.add_callback(_flush_canvas_draw)
            try:
                timer.start()
            except Exception:
                render_batch_state["scheduled"] = False
                render_batch_state["timer"] = None
                try:
                    fig.canvas.draw_idle()
                except Exception:
                    pass

        contour_rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.float32)
        contour_overlay_artist = ax.imshow(
            contour_rgba,
            interpolation="nearest",
        )
        brush_cursor_artist = Circle(
            (0.0, 0.0),
            radius=float(ROI_BRUSH_RADIUS_DEFAULT),
            fill=False,
            linewidth=1.2,
            edgecolor="cyan",
            alpha=0.95,
            visible=False,
        )
        ax.add_patch(brush_cursor_artist)

        def _apply_background_view(force: bool = False) -> None:
            mode = str(view_state.get("mode", "original")).lower()
            if mode == "enhanced":
                base_image_artist.set_data(image_enhanced_u8)
            else:
                base_image_artist.set_data(image_original_work_u8)
                mode = "original"
                view_state["mode"] = mode
            ax.set_title(str(title))
            if force:
                _request_canvas_draw(immediate=True)

        def _mask_u8() -> np.ndarray:
            return np.asarray(roi_state["mask_u8"], dtype=np.uint8)

        def _mask_bool() -> np.ndarray:
            cached = roi_state.get("mask_cache")
            if cached is None:
                cached = _mask_u8() > 0
                roi_state["mask_cache"] = cached
            return np.asarray(cached, dtype=bool)

        def _set_mask(mask: np.ndarray) -> None:
            bool_mask = np.asarray(mask, dtype=bool)
            roi_state["mask_cache"] = bool_mask
            roi_state["mask_u8"] = bool_mask.astype(np.uint8) * 255

        def _mask_has_pixels() -> bool:
            return bool(np.any(_mask_u8()))

        def _history_commit_current() -> None:
            snapshot = _mask_bool().copy()
            undo_stack = history_state["undo"]
            if undo_stack and np.array_equal(undo_stack[-1], snapshot):
                return
            undo_stack.append(snapshot)
            history_state["redo"].clear()

        def _undo_last_action(force_refresh: bool = True) -> bool:
            undo_stack = history_state["undo"]
            redo_stack = history_state["redo"]
            if len(undo_stack) <= 1:
                return False
            redo_stack.append(_mask_bool().copy())
            undo_stack.pop()
            restored = np.asarray(undo_stack[-1]).astype(bool)
            _set_mask(restored)
            selected["verts"] = None
            _refresh_contour(_mask_u8(), force=force_refresh)
            return True

        def _redo_last_action(force_refresh: bool = True) -> bool:
            undo_stack = history_state["undo"]
            redo_stack = history_state["redo"]
            if not redo_stack:
                return False
            restored = np.asarray(redo_stack.pop()).astype(bool)
            _set_mask(restored)
            if (not undo_stack) or (not np.array_equal(undo_stack[-1], restored)):
                undo_stack.append(restored.copy())
            selected["verts"] = None
            _refresh_contour(_mask_u8(), force=force_refresh)
            return True

        perimeter_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        contour_thickness = max(1, int(ROI_CONTOUR_THICKNESS_PX))
        contour_thickness_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * contour_thickness - 1, 2 * contour_thickness - 1),
        )

        def _mask_perimeter(mask_u8: np.ndarray) -> np.ndarray:
            er = cv2.erode(mask_u8, perimeter_kernel)
            perim = cv2.subtract(mask_u8, er)
            if contour_thickness > 1:
                perim = cv2.dilate(perim, contour_thickness_kernel, iterations=1)
            return perim > 0

        def _refresh_contour(mask_u8: np.ndarray, force: bool = False) -> None:
            now = time.perf_counter()
            if not force:
                if (now - float(render_state["last_contour"])) < (1.0 / max(float(render_tuning["editor_max_fps"]), 1.0)):
                    return
            render_state["last_contour"] = now

            perim = _mask_perimeter(mask_u8)
            mask_bool = np.asarray(mask_u8, dtype=np.uint8) > 0
            contour_rgba.fill(0.0)
            overlay_mode = str(mask_overlay_state.get("mode", "filled")).lower()
            if overlay_mode == "filled" and np.any(mask_bool):
                contour_rgba[mask_bool, 0] = 1.0
                contour_rgba[mask_bool, 1] = 1.0
                contour_rgba[mask_bool, 3] = 0.22
            if np.any(perim):
                contour_rgba[perim, 0] = 1.0
                contour_rgba[perim, 1] = 1.0
                contour_rgba[perim, 3] = 0.95
            contour_overlay_artist.set_data(contour_rgba)
            _request_canvas_draw()

        def _update_selector_verts(arr: np.ndarray) -> None:
            selected["verts"] = np.asarray(arr, dtype=np.float32)

        def _current_verts() -> np.ndarray | None:
            verts = selected["verts"]
            selector = selector_state["obj"]
            if verts is None and selector is not None and hasattr(selector, "verts"):
                verts = selector.verts
            if verts is None:
                return None
            arr = np.asarray(verts, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] != 2:
                return None
            return arr

        def _set_polygon_selector_active(active: bool) -> None:
            selector = selector_state["obj"]
            if selector is None or not hasattr(selector, "set_active"):
                return
            try:
                selector.set_active(bool(active))
            except Exception:
                pass

        tools_help_text = fig.text(
            0.015,
            0.50,
            self._t("roi.tools_help"),
            fontsize=10,
            va="center",
            ha="left",
            linespacing=1.25,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85, "pad": 4},
        )
        tool_info_text = fig.text(
            0.80,
            0.60,
            "",
            fontsize=12,
            va="center",
            ha="left",
            linespacing=1.25,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85, "pad": 4},
        )
        tool_desc_title_text = fig.text(
            0.80,
            0.49,
            self._t("roi.tool_desc_title"),
            fontsize=12,
            va="center",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85, "pad": 4},
        )
        tool_desc_text = fig.text(
            0.80,
            0.38,
            "",
            fontsize=12,
            va="center",
            ha="left",
            linespacing=1.25,
            wrap=True,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85, "pad": 4},
        )
        tool_panel_state: dict[str, bool] = {"desc_minimized": True}
        desc_toggle_ax = fig.add_axes([0.80, 0.20, 0.18, 0.05])
        desc_toggle_btn = Button(desc_toggle_ax, self._t("roi.minimize_description"))
        mask_overlay_toggle_ax = fig.add_axes([0.80, 0.14, 0.18, 0.05])
        mask_overlay_toggle_btn = Button(mask_overlay_toggle_ax, self._preview_mode_label("contour"))
        view_toggle_ax = fig.add_axes([0.80, 0.08, 0.18, 0.05])
        view_toggle_btn = Button(view_toggle_ax, self._t("roi.switch_image"))

        def _tool_mode_label() -> str:
            mode = str(tool_state["mode"]).lower()
            if mode == "eraser":
                return self._t("roi.tool.eraser")
            if mode == "component_remove":
                return self._t("roi.tool.component_remove")
            return self._t("roi.tool.brush")

        def _tool_mode_description() -> str:
            mode = str(tool_state["mode"]).lower()
            if mode == "eraser":
                return self._t("roi.desc.eraser")
            if mode == "component_remove":
                return self._t("roi.desc.component_remove")
            return self._t("roi.desc.brush")

        def _view_mode_label() -> str:
            mode = str(view_state.get("mode", "original")).lower()
            return view_label_map.get(mode, "Original")

        def _mask_overlay_mode() -> str:
            mode = str(mask_overlay_state.get("mode", "filled")).lower()
            return "contour" if mode == "contour" else "filled"

        def _mask_overlay_mode_label() -> str:
            return self._preview_mode_label(_mask_overlay_mode())

        def _sync_mask_overlay_toggle_button() -> None:
            next_mode = "filled" if _mask_overlay_mode() == "contour" else "contour"
            mask_overlay_toggle_btn.label.set_text(self._preview_mode_label(next_mode))

        def _sync_tool_description_panel() -> None:
            minimized = bool(tool_panel_state["desc_minimized"])
            show_desc = not minimized
            tool_desc_title_text.set_visible(show_desc)
            tool_desc_text.set_visible(show_desc)
            desc_toggle_btn.label.set_text(self._t("roi.show_description" if minimized else "roi.minimize_description"))

        def _toggle_tool_description(_event) -> None:
            tool_panel_state["desc_minimized"] = not bool(tool_panel_state["desc_minimized"])
            _sync_tool_description_panel()
            _request_canvas_draw()

        def _toggle_background_view(_event=None) -> None:
            view_state["mode"] = "enhanced" if str(view_state.get("mode", "original")) == "original" else "original"
            _apply_background_view(force=True)
            _update_tool_info()

        def _toggle_mask_overlay_mode(_event=None) -> None:
            mask_overlay_state["mode"] = "filled" if _mask_overlay_mode() == "contour" else "contour"
            _refresh_contour(_mask_u8(), force=True)
            _update_tool_info()

        desc_toggle_cid = desc_toggle_btn.on_clicked(_toggle_tool_description)
        mask_overlay_toggle_cid = mask_overlay_toggle_btn.on_clicked(_toggle_mask_overlay_mode)
        view_toggle_cid: int | None = view_toggle_btn.on_clicked(_toggle_background_view)

        def _update_tool_info() -> None:
            radius = int(brush_state["radius"])
            cursor = "ON" if bool(brush_state["show_cursor"]) else "OFF"
            applied_scale_pct = int(round(100.0 * float(editor_scale)))
            selected_scale_pct = int(round(100.0 * float(render_tuning["work_scale"])))
            if selected_scale_pct != applied_scale_pct:
                scale_line = f"{self._t('roi.scale')}: {applied_scale_pct}% ({self._t('roi.next')}: {selected_scale_pct}%)"
            else:
                scale_line = f"{self._t('roi.scale')}: {applied_scale_pct}%"
            tool_info_text.set_text(
                "\n".join(
                    [
                        self._t("roi.status_current"),
                        f"{self._t('roi.profile')}: {profile_label}",
                        scale_line,
                        f"{self._t('roi.tool')}: {_tool_mode_label()}",
                        f"{self._t('roi.radius')}: {radius}px",
                        f"{self._t('roi.cursor')}: {cursor}",
                        f"{self._t('roi.view')}: {_view_mode_label()}",
                        f"{self._t('roi.edge')}: {_mask_overlay_mode_label()}",
                    ]
                )
            )
            tool_desc_text.set_text(_tool_mode_description())
            _sync_tool_description_panel()
            _sync_mask_overlay_toggle_button()
            _request_canvas_draw()

        def _refresh_brush_cursor(force: bool = False) -> None:
            now = time.perf_counter()
            if not force:
                if (now - float(render_state["last_cursor"])) < (
                    1.0 / max(float(render_tuning["cursor_max_fps"]), 1.0)
                ):
                    return
            render_state["last_cursor"] = now

            show = bool(brush_state["show_cursor"]) and bool(brush_state["cursor_in_axes"])
            if not show:
                brush_cursor_artist.set_visible(False)
                _request_canvas_draw()
                return

            cx, cy = brush_state["cursor_xy"]
            brush_cursor_artist.set_center((float(cx), float(cy)))
            mode = str(tool_state["mode"]).lower()
            if mode == "component_remove":
                brush_cursor_artist.set_radius(7.0)
                brush_cursor_artist.set_edgecolor("red")
            else:
                brush_cursor_artist.set_radius(float(_brush_radius()))
                brush_cursor_artist.set_edgecolor("orange" if mode == "eraser" else "cyan")
            brush_cursor_artist.set_visible(True)
            _request_canvas_draw()

        def _set_tool_mode(mode: str) -> None:
            norm = str(mode).lower().strip()
            if norm not in ("brush", "eraser", "component_remove"):
                norm = "brush"
            tool_state["mode"] = norm
            _set_polygon_selector_active(False)
            _update_tool_info()
            _refresh_brush_cursor(force=True)

        def _ensure_current_verts() -> np.ndarray | None:
            verts = _current_verts()
            if verts is not None:
                return verts
            poly = mask_to_polygon(_mask_bool())
            if poly.shape[0] < 3:
                return None
            _update_selector_verts(poly)
            return poly

        # PT: Mantem a edicao focada em mascara (pincel/borracha), sem iniciar exibicao por poligono. | EN: Keeps editing focused on the mask (brush/eraser) without starting in polygon display mode.
        selected["verts"] = None
        _refresh_contour(_mask_u8(), force=True)

        def _set_selected_verts(
            verts: object,
            force_refresh: bool = False,
            record_history: bool = True,
        ) -> None:
            if verts is None:
                return
            arr = np.asarray(verts, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] != 2:
                return
            mask = polygon_to_mask(image.shape[:2], arr)
            if not np.any(mask):
                return
            _update_selector_verts(arr)
            _set_mask(mask)
            if record_history:
                _history_commit_current()
            _refresh_contour(_mask_u8(), force=force_refresh)

        def _sync_selector_from_mask(mask: np.ndarray) -> None:
            poly = mask_to_polygon(mask)
            if poly.shape[0] < 3:
                return
            _update_selector_verts(poly)

        def _brush_radius() -> int:
            return int(np.clip(int(brush_state["radius"]), ROI_BRUSH_RADIUS_MIN, ROI_BRUSH_RADIUS_MAX))

        def _apply_brush_stamp(
            x: float,
            y: float,
            erase: bool,
            connect_from: tuple[float, float] | None = None,
            force_refresh: bool = False,
        ) -> None:
            xi = int(np.clip(round(float(x)), 0, image.shape[1] - 1))
            yi = int(np.clip(round(float(y)), 0, image.shape[0] - 1))
            radius = _brush_radius()
            work = _mask_u8()
            color = 0 if bool(erase) else 255
            if connect_from is not None:
                px, py = connect_from
                pxi = int(np.clip(round(float(px)), 0, image.shape[1] - 1))
                pyi = int(np.clip(round(float(py)), 0, image.shape[0] - 1))
                if (pxi != xi) or (pyi != yi):
                    cv2.line(
                        work,
                        (pxi, pyi),
                        (xi, yi),
                        int(color),
                        thickness=max(1, int(2 * radius)),
                        lineType=cv2.LINE_8,
                    )
            cv2.circle(work, (xi, yi), int(radius), int(color), thickness=-1, lineType=cv2.LINE_8)
            roi_state["mask_u8"] = work
            roi_state["mask_cache"] = None
            selected["verts"] = None
            _refresh_contour(work, force=force_refresh)

        def _holes_from_mask(mask: np.ndarray) -> np.ndarray:
            src = np.asarray(mask).astype(bool)
            if src.ndim != 2 or src.size == 0:
                return np.zeros_like(src, dtype=bool)
            u8 = (src.astype(np.uint8) * 255)
            padded = cv2.copyMakeBorder(u8, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
            flood = padded.copy()
            ff_mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), dtype=np.uint8)
            cv2.floodFill(flood, ff_mask, (0, 0), 255)
            holes_padded = flood == 0
            return holes_padded[1:-1, 1:-1]

        def _fill_enclosed_regions_from_stroke(mask_before: np.ndarray, force_refresh: bool = True) -> bool:
            before = np.asarray(mask_before).astype(bool)
            after = _mask_bool()
            if before.shape != after.shape:
                return False

            changed = after & (~before)
            if not np.any(changed):
                return False

            holes_before = _holes_from_mask(before)
            holes_after = _holes_from_mask(after)
            new_holes = holes_after & (~holes_before)
            if not np.any(new_holes):
                return False

            # PT: Preenche somente buracos novos criados pelo traco atual. | EN: Fills only new holes created by the current stroke.
            updated = after | new_holes
            if np.array_equal(updated, after):
                return False

            _set_mask(updated)
            selected["verts"] = None
            _refresh_contour(_mask_u8(), force=force_refresh)
            return True

        def _remove_component_at(x: float, y: float, force_refresh: bool = True) -> bool:
            mask_now = _mask_bool()
            if not np.any(mask_now):
                return False

            xi = int(np.clip(round(float(x)), 0, image.shape[1] - 1))
            yi = int(np.clip(round(float(y)), 0, image.shape[0] - 1))
            if not bool(mask_now[yi, xi]):
                return False

            _num_labels, labels = cv2.connectedComponents(mask_now.astype(np.uint8), connectivity=8)
            label = int(labels[yi, xi])
            if label <= 0:
                return False

            updated = mask_now.copy()
            updated[labels == label] = False
            if np.array_equal(updated, mask_now):
                return False

            _set_mask(updated)
            selected["verts"] = None
            _history_commit_current()
            _refresh_contour(_mask_u8(), force=force_refresh)
            return True

        def _apply_offset(offset_px: float) -> None:
            verts = _current_verts()
            if verts is not None:
                base_mask = polygon_to_mask(image.shape[:2], verts)
            else:
                base_mask = _mask_bool()
            if not np.any(base_mask):
                return

            shifted = offset_mask_sdf(base_mask, offset_px=float(offset_px))
            if not np.any(shifted):
                return
            _set_mask(shifted)
            selected["verts"] = None
            _history_commit_current()
            _refresh_contour(_mask_u8(), force=True)

        def _clear_entire_mask(force_refresh: bool = True) -> bool:
            if not _mask_has_pixels():
                return False
            _set_mask(np.zeros(image.shape[:2], dtype=bool))
            selected["verts"] = None
            _history_commit_current()
            _refresh_contour(_mask_u8(), force=force_refresh)
            return True

        def _start_brush_drag(event) -> None:
            if event.inaxes != ax or event.button != 1:
                return
            mode = tool_state.get("mode")
            if event.xdata is None or event.ydata is None:
                return
            if mode == "component_remove":
                _remove_component_at(float(event.xdata), float(event.ydata), force_refresh=True)
                return
            if mode not in ("brush", "eraser"):
                return
            brush_drag_state["active"] = True
            brush_drag_state["mode"] = str(mode)
            brush_drag_state["mask_before"] = _mask_bool().copy()
            x = float(event.xdata)
            y = float(event.ydata)
            brush_drag_state["last_xy"] = (x, y)
            _apply_brush_stamp(x, y, erase=(mode == "eraser"), force_refresh=True)

        def _move_brush_drag(event) -> None:
            if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
                brush_state["cursor_in_axes"] = True
                brush_state["cursor_xy"] = (float(event.xdata), float(event.ydata))
            else:
                brush_state["cursor_in_axes"] = False
            _refresh_brush_cursor()

            if not bool(brush_drag_state["active"]):
                return
            if event.inaxes != ax or event.xdata is None or event.ydata is None:
                return
            mode = tool_state.get("mode")
            if mode not in ("brush", "eraser"):
                return
            x = float(event.xdata)
            y = float(event.ydata)
            last_xy = brush_drag_state.get("last_xy")
            connect_from = last_xy if isinstance(last_xy, tuple) else None
            _apply_brush_stamp(x, y, erase=(mode == "eraser"), connect_from=connect_from)
            brush_drag_state["last_xy"] = (x, y)

        def _end_brush_drag(event) -> None:
            if event.button == 1 and bool(brush_drag_state["active"]):
                brush_drag_state["active"] = False
                brush_drag_state["last_xy"] = None
                mode = str(brush_drag_state.get("mode", "")).lower()
                mask_before = brush_drag_state.get("mask_before")
                if mode == "brush" and isinstance(mask_before, np.ndarray):
                    _fill_enclosed_regions_from_stroke(mask_before, force_refresh=False)
                brush_drag_state["mask_before"] = None
                selected["verts"] = None
                _history_commit_current()
                _refresh_contour(_mask_u8(), force=True)

        def _warp_vertices_local(base: np.ndarray, anchor_idx: int, dx: float, dy: float) -> np.ndarray:
            n = int(base.shape[0])
            if n < 3:
                return base
            sigma = max(float(ROI_LOCAL_DRAG_MIN_SIGMA), float(n) * float(ROI_LOCAL_DRAG_SIGMA_FRAC))
            radius = int(max(8.0, np.ceil(3.0 * sigma)))
            idx = np.arange(n, dtype=np.int32)
            d = np.abs(idx - int(anchor_idx))
            d = np.minimum(d, n - d).astype(np.float32)
            w = np.exp(-0.5 * ((d / float(sigma)) ** 2))
            w[d > float(radius)] = 0.0
            out = base.copy()
            out[:, 0] += w * float(dx)
            out[:, 1] += w * float(dy)
            out[:, 0] = np.clip(out[:, 0], 0.0, float(image.shape[1] - 1))
            out[:, 1] = np.clip(out[:, 1], 0.0, float(image.shape[0] - 1))
            return out

        def _start_local_drag(event) -> None:
            if event.inaxes != ax or event.button != 3:
                return
            if event.xdata is None or event.ydata is None:
                return
            verts = _ensure_current_verts()
            if verts is None:
                return
            x = float(event.xdata)
            y = float(event.ydata)
            d2 = np.square(verts[:, 0] - x) + np.square(verts[:, 1] - y)
            local_drag_state["active"] = True
            local_drag_state["anchor_idx"] = int(np.argmin(d2))
            local_drag_state["start_xy"] = (x, y)
            local_drag_state["base_verts"] = verts.copy()
            local_drag_state["last_apply_ts"] = 0.0

        def _move_local_drag(event) -> None:
            if not bool(local_drag_state["active"]):
                return
            if event.inaxes != ax or event.xdata is None or event.ydata is None:
                return
            now = time.perf_counter()
            if (now - float(local_drag_state.get("last_apply_ts", 0.0))) < (
                1.0 / max(float(render_tuning["editor_max_fps"]), 1.0)
            ):
                return
            local_drag_state["last_apply_ts"] = now
            base = local_drag_state.get("base_verts")
            start_xy = local_drag_state.get("start_xy")
            anchor_idx = int(local_drag_state.get("anchor_idx", -1))
            if base is None or start_xy is None or anchor_idx < 0:
                return
            base_arr = np.asarray(base, dtype=np.float32)
            sx, sy = start_xy
            dx = float(event.xdata) - float(sx)
            dy = float(event.ydata) - float(sy)
            moved = _warp_vertices_local(base_arr, anchor_idx=anchor_idx, dx=dx, dy=dy)
            _set_selected_verts(moved, record_history=False)

        def _end_local_drag(event) -> None:
            if event.button == 3 and bool(local_drag_state["active"]):
                local_drag_state["active"] = False
                local_drag_state["anchor_idx"] = -1
                local_drag_state["start_xy"] = None
                local_drag_state["base_verts"] = None
                local_drag_state["last_apply_ts"] = 0.0
                if _mask_has_pixels():
                    _history_commit_current()
                    _refresh_contour(_mask_u8(), force=True)

        selector_state["obj"] = None
        _apply_background_view(force=False)
        _set_tool_mode("brush")

        def _edited_mask_at_original_size() -> np.ndarray:
            edited_mask = _mask_bool().astype(bool)
            if edited_mask.shape != orig_shape:
                edited_mask = resize_mask(edited_mask, orig_shape)
            return edited_mask

        def _load_next_editor_item(payload: dict[str, object]) -> None:
            """Substitui a imagem e reinicia os estados editaveis sem fechar a janela."""
            nonlocal image_rgb_u8
            nonlocal original_rgb_u8
            nonlocal orig_shape
            nonlocal image_work_u8
            nonlocal image_original_work_u8
            nonlocal image_enhanced_u8
            nonlocal image
            nonlocal title
            nonlocal image_cache_key
            nonlocal contour_rgba

            next_title = str(payload.get("title", ""))
            ax.set_title(f"{next_title} - carregando...")
            _request_canvas_draw(immediate=True)
            try:
                fig.canvas.flush_events()
            except Exception:
                pass

            next_image = np.asarray(payload["image"], dtype=np.uint8)
            if next_image.ndim == 2:
                next_image = cv2.cvtColor(next_image, cv2.COLOR_GRAY2RGB)
            elif next_image.ndim == 3 and next_image.shape[2] >= 3:
                next_image = next_image[:, :, :3]
            else:
                raise ValueError(self._t("roi.editor.invalid_image"))

            next_original_raw = payload.get("original_image")
            try:
                if next_original_raw is None:
                    next_original = next_image.copy()
                else:
                    next_original = self._to_rgb_u8(np.asarray(next_original_raw))
            except Exception:
                next_original = next_image.copy()

            next_orig_shape = (int(next_image.shape[0]), int(next_image.shape[1]))
            if tuple(next_original.shape[:2]) != next_orig_shape:
                target_h, target_w = next_orig_shape
                src_h, src_w = next_original.shape[:2]
                interpolation = cv2.INTER_AREA if (src_h >= target_h and src_w >= target_w) else cv2.INTER_LINEAR
                next_original = cv2.resize(
                    np.asarray(next_original, dtype=np.uint8),
                    (target_w, target_h),
                    interpolation=interpolation,
                )

            if editor_scale < 0.999:
                work_h = max(1, int(round(next_orig_shape[0] * editor_scale)))
                work_w = max(1, int(round(next_orig_shape[1] * editor_scale)))
                next_work = cv2.resize(next_image, (work_w, work_h), interpolation=cv2.INTER_AREA)
                next_original_work = cv2.resize(next_original, (work_w, work_h), interpolation=cv2.INTER_AREA)
            else:
                next_work = next_image
                next_original_work = next_original

            next_cache_key = str(payload.get("image_cache_key", "")).strip() or None
            try:
                next_enhanced_full = self._best_defined_image(next_original, cache_key=next_cache_key)
            except Exception:
                next_enhanced_full = next_original.copy()
            if editor_scale < 0.999:
                next_enhanced = cv2.resize(
                    np.asarray(next_enhanced_full, dtype=np.uint8),
                    (work_w, work_h),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                next_enhanced = np.asarray(next_enhanced_full, dtype=np.uint8)

            next_mask = resize_mask(
                np.asarray(payload["initial_mask"]).astype(bool),
                next_work.shape[:2],
            )

            draw_timer = render_batch_state.get("timer")
            if draw_timer is not None:
                try:
                    draw_timer.stop()
                except Exception:
                    pass
            render_batch_state["scheduled"] = False
            render_batch_state["timer"] = None

            image_rgb_u8 = next_image
            original_rgb_u8 = next_original
            orig_shape = next_orig_shape
            image_work_u8 = next_work
            image_original_work_u8 = next_original_work
            image_enhanced_u8 = next_enhanced
            image = image_work_u8
            title = next_title
            image_cache_key = next_cache_key

            height, width = image.shape[:2]
            extent = (-0.5, float(width) - 0.5, float(height) - 0.5, -0.5)
            base_image_artist.set_data(image_original_work_u8)
            base_image_artist.set_extent(extent)
            contour_rgba = np.zeros((height, width, 4), dtype=np.float32)
            contour_overlay_artist.set_data(contour_rgba)
            contour_overlay_artist.set_extent(extent)
            ax.set_xlim(-0.5, float(width) - 0.5)
            ax.set_ylim(float(height) - 0.5, -0.5)

            _set_mask(next_mask)
            history_state["undo"] = [next_mask.copy()]
            history_state["redo"] = []
            selected["verts"] = None
            selected["accepted"] = False
            selected["result_mask"] = None
            local_drag_state.update(
                {
                    "active": False,
                    "anchor_idx": -1,
                    "start_xy": None,
                    "base_verts": None,
                    "last_apply_ts": 0.0,
                }
            )
            brush_drag_state.update(
                {
                    "active": False,
                    "last_xy": None,
                    "mode": "brush",
                    "mask_before": None,
                }
            )
            brush_state["cursor_in_axes"] = False
            brush_cursor_artist.set_visible(False)
            render_state["last_contour"] = 0.0
            render_state["last_cursor"] = 0.0
            view_state["mode"] = "original"
            _apply_background_view(force=False)
            _set_tool_mode("brush")
            _refresh_contour(_mask_u8(), force=True)
            _request_canvas_draw(immediate=True)

        def on_key(event):
            key = str(event.key or "").lower()
            if key in ("ctrl+z", "cmd+z"):
                _undo_last_action(force_refresh=True)
                return
            if key in ("ctrl+y", "cmd+y", "ctrl+shift+z", "cmd+shift+z"):
                _redo_last_action(force_refresh=True)
                return
            if key in ("delete", "backspace"):
                _clear_entire_mask(force_refresh=True)
                return

            if key in ("enter", "return"):
                verts = _current_verts()
                if verts is not None:
                    selected["verts"] = np.asarray(verts, dtype=np.float32)
                    _set_mask(polygon_to_mask(image.shape[:2], verts))
                else:
                    selected["verts"] = None
                edited_mask = _edited_mask_at_original_size()
                if callable(on_accept_next):
                    next_payload = on_accept_next(edited_mask)
                    if next_payload is not None:
                        _load_next_editor_item(next_payload)
                        return
                selected["accepted"] = True
                selected["result_mask"] = edited_mask
                _remember_window_state()
                plt.close(fig)
            elif key == "escape":
                selected["accepted"] = False
                _remember_window_state()
                plt.close(fig)
            elif key in ("+", "="):
                _apply_offset(ROI_OFFSET_STEP_PX)
            elif key in ("-", "_"):
                _apply_offset(-ROI_OFFSET_STEP_PX)
            elif key in ("b",):
                _set_tool_mode("brush")
            elif key in ("e",):
                _set_tool_mode("eraser")
            elif key in ("x",):
                _set_tool_mode("component_remove")
            elif key in ("v",):
                brush_state["show_cursor"] = not bool(brush_state["show_cursor"])
                _update_tool_info()
                _refresh_brush_cursor(force=True)
            elif key in ("t",):
                _toggle_background_view()
            elif key in ("c",):
                _toggle_mask_overlay_mode()
            elif key in ("[", "bracketleft"):
                brush_state["radius"] = max(ROI_BRUSH_RADIUS_MIN, _brush_radius() - 1)
                _update_tool_info()
                _refresh_brush_cursor(force=True)
            elif key in ("]", "bracketright"):
                brush_state["radius"] = min(ROI_BRUSH_RADIUS_MAX, _brush_radius() + 1)
                _update_tool_info()
                _refresh_brush_cursor(force=True)

        cid_key = fig.canvas.mpl_connect("key_press_event", on_key)
        cid_press_brush = fig.canvas.mpl_connect("button_press_event", _start_brush_drag)
        cid_move_brush = fig.canvas.mpl_connect("motion_notify_event", _move_brush_drag)
        cid_release_brush = fig.canvas.mpl_connect("button_release_event", _end_brush_drag)
        cid_press = fig.canvas.mpl_connect("button_press_event", _start_local_drag)
        cid_move = fig.canvas.mpl_connect("motion_notify_event", _move_local_drag)
        cid_release = fig.canvas.mpl_connect("button_release_event", _end_local_drag)
        cid_close = fig.canvas.mpl_connect("close_event", lambda _event: _remember_window_state())
        fig.subplots_adjust(left=0.24, right=0.76, top=0.95, bottom=0.06)
        restore_window_timer = None
        self._apply_roi_editor_window_state(fig)
        if bool(self._roi_editor_window_state.get("fullscreen")) or bool(self._roi_editor_window_state.get("zoomed")):
            try:
                restore_window_timer = fig.canvas.new_timer(interval=80)
                if hasattr(restore_window_timer, "single_shot"):
                    try:
                        restore_window_timer.single_shot = True
                    except Exception:
                        pass

                def _restore_window_state() -> bool:
                    self._apply_roi_editor_window_state(fig)
                    return False

                restore_window_timer.add_callback(_restore_window_state)
                restore_window_timer.start()
            except Exception:
                restore_window_timer = None
        previous_page_name = self._active_page_name
        self._track_page_transition("edicao_de_mascara")
        try:
            plt.show(block=True)
        finally:
            if previous_page_name is not None:
                self._track_page_transition(previous_page_name)
        if restore_window_timer is not None:
            try:
                restore_window_timer.stop()
            except Exception:
                pass
        draw_timer = render_batch_state.get("timer")
        if draw_timer is not None:
            try:
                draw_timer.stop()
            except Exception:
                pass
        render_batch_state["scheduled"] = False
        render_batch_state["timer"] = None
        try:
            desc_toggle_btn.disconnect(desc_toggle_cid)
        except Exception:
            pass
        try:
            mask_overlay_toggle_btn.disconnect(mask_overlay_toggle_cid)
        except Exception:
            pass
        if view_toggle_cid is not None:
            try:
                view_toggle_btn.disconnect(view_toggle_cid)
            except Exception:
                pass
        for cid in (cid_key, cid_press_brush, cid_move_brush, cid_release_brush, cid_press, cid_move, cid_release, cid_close):
            try:
                fig.canvas.mpl_disconnect(cid)
            except Exception:
                pass

        if not bool(selected["accepted"]):
            return None

        result_mask = selected.get("result_mask")
        if isinstance(result_mask, np.ndarray):
            return np.asarray(result_mask).astype(bool)
        return _edited_mask_at_original_size()

    def run(self) -> None:
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.cancel_event.set()
            try:
                self.root.quit()
            except Exception:
                pass
            try:
                self.root.destroy()
            except Exception:
                pass
        finally:
            self.cancel_event.set()
            _close_all_pyplot_figures()


def parse_args() -> argparse.Namespace:
    """Define argumentos de linha de comando para iniciar a GUI com valores predefinidos."""
    parser = argparse.ArgumentParser(description="Interface para processar e revisar grupos 0h/24h/48h em lote.")
    parser.add_argument("--folder", type=str, default=None, help="Pasta inicial com imagens.")
    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="Pasta para salvar resultados. Se vazio, usa <pasta de imagens>/_cora_resultados.",
    )
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada principal para inicializacao da aplicacao de desktop."""
    args = parse_args()
    folder_arg = "" if args.folder is None else str(args.folder).strip()
    output_arg = "" if args.output_folder is None else str(args.output_folder).strip()
    app = CORAApp(
        folder_path=folder_arg,
        output_folder_path=output_arg,
    )
    app.run()


if __name__ == "__main__":
    main()
