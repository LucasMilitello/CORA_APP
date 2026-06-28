"""Pipeline de avaliacao de area com pre-processamento, segmentacao e artefatos de suporte."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

# PT: Configure aqui um caminho padrao para sua imagem base. | EN: Configure a default path to your base image here.
BASE_IMAGE_PATH = r""

# PT: Limiares de brilho usados pelo preprocessamento adaptativo. | EN: Brightness thresholds used by adaptive preprocessing.
BRIGHT_IMAGE_THRESHOLD = 165.0
BRIGHT_0H_IMAGE_THRESHOLD = 155.0
MID_BRIGHT_IMAGE_THRESHOLD = 140.0
DARK_IMAGE_THRESHOLD = 105.0


def binary_fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Preenche buracos de uma mascara binaria com fallback puro em OpenCV/NumPy.
    Mantemos essa implementacao local para evitar dependencia de startup no SciPy.
    """
    src = np.asarray(mask).astype(bool)
    if src.ndim != 2:
        raise ValueError("mask deve ser 2D.")

    # PT: Fallback: identifica fundo conectado a borda e preenche apenas buracos internos. | EN: Fallback: identifies background connected to the border and fills only internal holes.
    padded = np.pad(src.astype(np.uint8), 1, mode="constant", constant_values=0)
    inv = (1 - padded).astype(np.uint8)
    flood = inv.copy()
    flood_mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, seedPoint=(0, 0), newVal=2)
    holes = flood == 1
    filled = (padded > 0) | holes
    return filled[1:-1, 1:-1]


def _normalize_multidimensional_image_shape(img: np.ndarray) -> np.ndarray:
    """Normaliza imagens com eixos extras para formato 2D/3D compativel com o pipeline."""
    arr = np.squeeze(np.asarray(img))
    if arr.size == 0:
        raise ValueError("Imagem vazia.")
    if arr.ndim < 2:
        raise ValueError(f"Formato de imagem nao suportado: shape={arr.shape}")

    # PT: Remove eixos extras (tempo/z/lotes), preservando possiveis canais. | EN: Removes extra axes (time/z/batches) while preserving possible channels.
    while arr.ndim > 3:
        if arr.shape[-1] in (3, 4):
            arr = arr[0, ...]
        elif arr.shape[0] in (3, 4):
            arr = arr[:, 0, ...]
        else:
            arr = arr[0, ...]

    if arr.ndim == 2:
        return arr
    if arr.shape[-1] in (3, 4):
        return arr
    if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        return np.moveaxis(arr, 0, -1)

    # PT: Caso ambiguo (ex.: pilha de paginas em tons de cinza), usa a primeira pagina. | EN: In ambiguous cases (for example, a stack of grayscale pages), uses the first page.
    return arr[0, ...]


def _normalize_tiff_shape_with_axes(img: np.ndarray, axes: str = "") -> np.ndarray:
    """Usa metadados de eixos TIFF para encontrar plano/canais corretos."""
    arr = np.asarray(img)
    axes_norm = (axes or "").upper()

    if arr.size == 0:
        raise ValueError("TIFF vazio.")

    if axes_norm and len(axes_norm) == arr.ndim:
        singleton_axes = tuple(i for i, size in enumerate(arr.shape) if size == 1)
        if singleton_axes:
            arr = np.squeeze(arr, axis=singleton_axes)
            axes_norm = "".join(tag for idx, tag in enumerate(axes_norm) if idx not in singleton_axes)

        if arr.ndim >= 2 and ("Y" in axes_norm) and ("X" in axes_norm):
            channel_tag = None
            for tag in ("S", "C"):
                if tag in axes_norm:
                    idx = axes_norm.index(tag)
                    if arr.shape[idx] in (3, 4):
                        channel_tag = tag
                        break

            keep_tags = {"Y", "X"}
            if channel_tag is not None:
                keep_tags.add(channel_tag)

            for axis_idx in range(arr.ndim - 1, -1, -1):
                if axes_norm[axis_idx] in keep_tags:
                    continue
                arr = np.take(arr, indices=0, axis=axis_idx)
                axes_norm = axes_norm[:axis_idx] + axes_norm[axis_idx + 1 :]

            perm = [axes_norm.index("Y"), axes_norm.index("X")]
            if (channel_tag is not None) and (channel_tag in axes_norm):
                perm.append(axes_norm.index(channel_tag))

            arr = np.transpose(arr, perm)
            if arr.ndim in (2, 3):
                if arr.ndim == 3 and arr.shape[2] not in (3, 4):
                    return arr[:, :, 0]
                return arr

    return _normalize_multidimensional_image_shape(arr)


def _read_tiff_image(path: Path) -> np.ndarray:
    """Le leitor TIFF com fallback entre TiffFile(series) e tiff.imread."""
    last_error = None

    try:
        with tiff.TiffFile(str(path)) as tif:
            if tif.series:
                series = tif.series[0]
                arr = series.asarray()
                axes = getattr(series, "axes", "") or ""
                return _normalize_tiff_shape_with_axes(arr, axes=axes)
    except Exception as exc:
        last_error = exc

    try:
        arr = tiff.imread(str(path))
        return _normalize_tiff_shape_with_axes(arr)
    except Exception as exc:
        last_error = exc
        raise RuntimeError(f"Nao consegui ler TIFF: {path}") from last_error


def _cv2_imread_unicode(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    """
    Leitura robusta para Windows/Unicode.
    Evita falhas do cv2.imread com caminhos contendo acentos/simbolos.
    """
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
        if raw.size > 0:
            decoded = cv2.imdecode(raw, int(flags))
            if decoded is not None:
                return decoded
    except Exception:
        pass
    # PT: Fallback padrao (mantem compatibilidade em cenarios onde fromfile falhar). | EN: Standard fallback that preserves compatibility when fromfile fails.
    return cv2.imread(str(path), int(flags))


def read_image(path: str) -> np.ndarray:
    """Le imagem de forma robusta para formatos comuns e caminhos Unicode."""
    resolved = Path(os.path.expandvars(path)).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Arquivo de imagem nao encontrado: {resolved}\n"
            "Informe um caminho valido com --base."
        )

    ext = resolved.suffix.lower()
    if ext in (".tif", ".tiff"):
        try:
            return _read_tiff_image(resolved)
        except Exception as tiff_error:
            # PT: Fallback para builds onde determinado codec TIFF nao esteja disponivel. | EN: Fallback for builds where a specific TIFF codec is unavailable.
            img = _cv2_imread_unicode(resolved, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"Nao consegui ler TIFF: {resolved}") from tiff_error
            if img.ndim == 3 and img.shape[2] in (3, 4):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return _normalize_multidimensional_image_shape(img)

    img = _cv2_imread_unicode(resolved, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Nao consegui ler: {resolved}")

    if img.ndim == 3 and img.shape[2] in (3, 4):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return _normalize_multidimensional_image_shape(img)


def mat2gray(img: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = img.astype(np.float32, copy=False)
    mn = float(np.min(x))
    mx = float(np.max(x))
    return (x - mn) / (mx - mn + eps)


def as_odd(value: int, min_value: int = 1) -> int:
    value = max(min_value, int(value))
    return value if (value % 2) == 1 else value + 1


def to_uint8_01(img01: np.ndarray) -> np.ndarray:
    return np.clip(mat2gray(img01) * 255.0, 0, 255).astype(np.uint8)


def ensure_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] >= 3:
        return cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
    raise ValueError("Formato de imagem nao suportado para conversao em cinza.")


def ensure_rgb_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        rgb = cv2.cvtColor(to_uint8_01(img), cv2.COLOR_GRAY2RGB)
    else:
        rgb = img[:, :, :3].copy()
        if rgb.dtype != np.uint8:
            rgb = to_uint8_01(rgb)
    return rgb


def _is_24h_time_tag(time_tag: Optional[str]) -> bool:
    return (time_tag or "").strip().lower() == "24h"


def _is_0h_time_tag(time_tag: Optional[str]) -> bool:
    return (time_tag or "").strip().lower() == "0h"


def _uses_0h_brightness_buckets(time_tag: Optional[str]) -> bool:
    # PT: "img" costuma representar a imagem inicial quando nao ha tag de tempo explicita. | EN: "img" usually represents the initial image when there is no explicit time tag.
    return (time_tag or "").strip().lower() in {"0h", "img"}


def _to_bgr_u8_for_adaptive_pipeline(img: np.ndarray) -> np.ndarray:
    src = np.asarray(img)
    if src.ndim == 2:
        gray_u8 = src if src.dtype == np.uint8 else to_uint8_01(src)
        return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    if src.ndim == 3 and src.shape[2] >= 3:
        rgb_u8 = src[:, :, :3]
        if rgb_u8.dtype != np.uint8:
            rgb_u8 = to_uint8_01(rgb_u8)
        return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    raise ValueError("Formato de imagem nao suportado para preprocessamento adaptativo.")


def _mean_gray_intensity(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def _adjust_gamma_bgr(image_bgr: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(float(gamma), 1e-6)
    table = np.array(
        [np.clip(((i / 255.0) ** gamma) * 255.0, 0, 255) for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image_bgr, table)


def _clahe_lab_bgr(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l_ch)
    merged = cv2.merge((l_eq, a_ch, b_ch))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _homomorphic_gray_u8(gray_u8: np.ndarray, sigma: float = 18.0, ksize: int = 81) -> np.ndarray:
    src = gray_u8.astype(np.float32) / 255.0
    src = np.clip(src, 1e-6, 1.0)
    img_log = np.log1p(src)

    k = as_odd(max(3, int(ksize)), min_value=3)
    illum = cv2.GaussianBlur(img_log, (k, k), sigmaX=float(sigma), sigmaY=float(sigma))
    reflect = img_log - illum
    out = np.expm1(reflect)
    out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
    return np.clip(out, 0, 255).astype(np.uint8)


def _local_clahe_gray(gray_u8: np.ndarray, clip_limit: float = 2.4, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    return clahe.apply(gray_u8)


def _edge_texture_confidence_gray(gray_u8: np.ndarray, local_win: int = 13) -> tuple[np.ndarray, np.ndarray]:
    src = gray_u8.astype(np.float32) / 255.0

    gx = cv2.Scharr(src, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(src, cv2.CV_32F, 0, 1)
    grad = (1.20 * np.abs(gx)) + (0.35 * np.abs(gy))
    grad = cv2.normalize(grad, None, 0.0, 1.0, cv2.NORM_MINMAX)

    win = as_odd(max(3, int(local_win)), min_value=3)
    mean = cv2.blur(src, (win, win))
    mean2 = cv2.blur(src * src, (win, win))
    texture = np.sqrt(np.maximum(mean2 - (mean * mean), 0.0))
    texture = cv2.normalize(texture, None, 0.0, 1.0, cv2.NORM_MINMAX)

    low_texture_gate = np.clip(1.0 - (0.72 * texture), 0.0, 1.0)
    edge_support = grad * (0.42 + (0.58 * low_texture_gate))
    edge_support = cv2.normalize(edge_support, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    texture_noise = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return edge_support, texture_noise


def _bright_image_homogeneous_border_enhance(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # PT: Corrige iluminacao sem gerar borda artificial. | EN: Corrects illumination without creating an artificial border.
    homo = _homomorphic_gray_u8(gray, sigma=24.0, ksize=101)
    smooth = cv2.bilateralFilter(homo, d=9, sigmaColor=28, sigmaSpace=32)
    smooth = cv2.medianBlur(smooth, 5)

    # PT: CLAHE leve: recupera bordas sem realcar demais textura celular fina. | EN: Light CLAHE recovers edges without overemphasizing fine cellular texture.
    eq = _local_clahe_gray(smooth, clip_limit=1.25, tile_grid_size=(8, 8))

    base = cv2.GaussianBlur(eq, (5, 5), 0)
    se_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    base = cv2.morphologyEx(base, cv2.MORPH_OPEN, se_bg)
    base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, se_bg)

    se_long = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    se_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    top_long = cv2.morphologyEx(base, cv2.MORPH_TOPHAT, se_long)
    bot_long = cv2.morphologyEx(base, cv2.MORPH_BLACKHAT, se_long)
    grad = cv2.morphologyEx(base, cv2.MORPH_GRADIENT, se_edge)

    border_support = cv2.addWeighted(top_long, 0.52, bot_long, 0.42, 0.0)
    border_support = cv2.addWeighted(border_support, 1.0, grad, 0.58, 0.0)

    refined = cv2.addWeighted(base, 0.92, border_support, 0.44, 0.0)
    refined = cv2.bilateralFilter(refined, d=5, sigmaColor=16, sigmaSpace=18)
    return cv2.cvtColor(refined, cv2.COLOR_GRAY2BGR)


def _mid_bright_image_enhance(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    homo = _homomorphic_gray_u8(gray, sigma=12.0, ksize=55)
    eq = _local_clahe_gray(homo, clip_limit=2.2, tile_grid_size=(8, 8))
    eq = cv2.bilateralFilter(eq, d=7, sigmaColor=22, sigmaSpace=22)

    # PT: Suaviza granulado mantendo estrutura principal das frentes. | EN: Smooths grain while preserving the main structure of the fronts.
    base = cv2.GaussianBlur(eq, (5, 5), 0)

    se_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    se_mid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    se_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

    top_small = cv2.morphologyEx(base, cv2.MORPH_TOPHAT, se_small)
    bot_small = cv2.morphologyEx(base, cv2.MORPH_BLACKHAT, se_small)
    top_mid = cv2.morphologyEx(base, cv2.MORPH_TOPHAT, se_mid)
    bot_mid = cv2.morphologyEx(base, cv2.MORPH_BLACKHAT, se_mid)
    top_large = cv2.morphologyEx(base, cv2.MORPH_TOPHAT, se_large)
    grad = cv2.morphologyEx(base, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    border_support = cv2.addWeighted(top_large, 0.72, top_mid, 0.55, 0.0)
    border_support = cv2.addWeighted(border_support, 1.0, grad, 0.62, 0.0)
    border_support = cv2.addWeighted(border_support, 1.0, bot_mid, 0.22, 0.0)
    central_noise = cv2.addWeighted(top_small, 0.72, bot_small, 0.78, 0.0)

    refined = cv2.addWeighted(base, 1.0, border_support, 0.68, 0.0)
    refined = cv2.addWeighted(refined, 1.0, central_noise, -0.40, 0.0)
    refined = cv2.GaussianBlur(refined, (3, 3), 0)
    return cv2.cvtColor(refined, cv2.COLOR_GRAY2BGR)


def _medium_0h_image_enhance(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # PT: 0h medio: mais correcao de fundo com suavizacao para nao virar textura. | EN: Medium 0h: stronger background correction with smoothing to prevent it from becoming texture.
    homo = _homomorphic_gray_u8(gray, sigma=18.0, ksize=83)
    eq = _local_clahe_gray(homo, clip_limit=2.9, tile_grid_size=(8, 8))
    eq = cv2.bilateralFilter(eq, d=11, sigmaColor=34, sigmaSpace=34)
    eq = cv2.medianBlur(eq, 5)

    base = cv2.GaussianBlur(eq, (5, 5), 0)
    se_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    base = cv2.morphologyEx(base, cv2.MORPH_OPEN, se_bg)
    base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, se_bg)
    shape_base = cv2.bilateralFilter(base, d=7, sigmaColor=22, sigmaSpace=26)
    shape_base = cv2.morphologyEx(
        shape_base,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )

    se_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    se_mid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    se_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    grad = cv2.morphologyEx(shape_base, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    top_small = cv2.morphologyEx(shape_base, cv2.MORPH_TOPHAT, se_small)
    bot_small = cv2.morphologyEx(shape_base, cv2.MORPH_BLACKHAT, se_small)
    top_mid = cv2.morphologyEx(shape_base, cv2.MORPH_TOPHAT, se_mid)
    bot_mid = cv2.morphologyEx(shape_base, cv2.MORPH_BLACKHAT, se_mid)
    top_large = cv2.morphologyEx(shape_base, cv2.MORPH_TOPHAT, se_large)
    edge_support, texture_noise = _edge_texture_confidence_gray(base, local_win=13)

    border_support = cv2.addWeighted(top_large, 0.42, top_mid, 0.38, 0.0)
    border_support = cv2.addWeighted(border_support, 1.0, bot_mid, 0.20, 0.0)
    border_support = cv2.addWeighted(border_support, 1.0, grad, 0.58, 0.0)
    border_support = cv2.addWeighted(border_support, 1.0, edge_support, 0.72, 0.0)
    small_texture = cv2.addWeighted(top_small, 0.32, bot_small, 0.42, 0.0)
    small_texture = cv2.addWeighted(small_texture, 1.0, texture_noise, 0.34, 0.0)

    refined = cv2.addWeighted(base, 1.0, border_support, 0.66, 0.0)
    refined = cv2.addWeighted(refined, 1.0, small_texture, -0.42, 0.0)
    refined = cv2.bilateralFilter(refined, d=5, sigmaColor=16, sigmaSpace=18)
    return cv2.cvtColor(refined, cv2.COLOR_GRAY2BGR)


def _dark_0h_image_enhance(image_bgr: np.ndarray) -> np.ndarray:
    processed = _adjust_gamma_bgr(image_bgr, gamma=0.88)
    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    homo = _homomorphic_gray_u8(gray, sigma=14.0, ksize=65)
    eq = _local_clahe_gray(homo, clip_limit=2.4, tile_grid_size=(8, 8))
    eq = cv2.bilateralFilter(eq, d=11, sigmaColor=30, sigmaSpace=30)
    eq = cv2.medianBlur(eq, 5)

    base = cv2.GaussianBlur(eq, (5, 5), 0)
    base = cv2.morphologyEx(base, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    edge_support, texture_noise = _edge_texture_confidence_gray(base, local_win=15)

    se_mid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    se_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29))
    top_mid = cv2.morphologyEx(base, cv2.MORPH_TOPHAT, se_mid)
    bot_mid = cv2.morphologyEx(base, cv2.MORPH_BLACKHAT, se_mid)
    top_large = cv2.morphologyEx(base, cv2.MORPH_TOPHAT, se_large)

    border_support = cv2.addWeighted(top_large, 0.40, top_mid, 0.38, 0.0)
    border_support = cv2.addWeighted(border_support, 1.0, bot_mid, 0.18, 0.0)
    border_support = cv2.addWeighted(border_support, 1.0, edge_support, 0.68, 0.0)

    refined = cv2.addWeighted(base, 1.0, border_support, 0.60, 0.0)
    refined = cv2.addWeighted(refined, 1.0, texture_noise, -0.34, 0.0)
    refined = cv2.bilateralFilter(refined, d=5, sigmaColor=16, sigmaSpace=18)
    return cv2.cvtColor(refined, cv2.COLOR_GRAY2BGR)


def _dark_24h_image_enhance(image_bgr: np.ndarray) -> np.ndarray:
    processed = _adjust_gamma_bgr(image_bgr, gamma=0.78)
    processed = _clahe_lab_bgr(processed, clip_limit=2.8, tile_grid_size=(8, 8))

    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    homo = _homomorphic_gray_u8(gray, sigma=16.0, ksize=71)
    eq = _local_clahe_gray(homo, clip_limit=3.0, tile_grid_size=(8, 8))
    eq = cv2.bilateralFilter(eq, d=9, sigmaColor=28, sigmaSpace=28)
    eq = cv2.medianBlur(eq, 5)

    base = cv2.GaussianBlur(eq, (5, 5), 0)
    se_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    base = cv2.morphologyEx(base, cv2.MORPH_OPEN, se_bg)
    base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, se_bg)

    edge_support, texture_noise = _edge_texture_confidence_gray(base, local_win=17)
    grad = cv2.morphologyEx(base, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    se_mid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    se_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))
    top_mid = cv2.morphologyEx(base, cv2.MORPH_TOPHAT, se_mid)
    bot_mid = cv2.morphologyEx(base, cv2.MORPH_BLACKHAT, se_mid)
    top_large = cv2.morphologyEx(base, cv2.MORPH_TOPHAT, se_large)

    border_support = cv2.addWeighted(top_large, 0.48, top_mid, 0.36, 0.0)
    border_support = cv2.addWeighted(border_support, 1.0, bot_mid, 0.24, 0.0)
    border_support = cv2.addWeighted(border_support, 1.0, grad, 0.64, 0.0)
    border_support = cv2.addWeighted(border_support, 1.0, edge_support, 0.82, 0.0)

    refined = cv2.addWeighted(base, 1.0, border_support, 0.72, 0.0)
    refined = cv2.addWeighted(refined, 1.0, texture_noise, -0.38, 0.0)
    refined = cv2.bilateralFilter(refined, d=5, sigmaColor=16, sigmaSpace=18)
    refined = cv2.GaussianBlur(refined, (3, 3), 0)
    return cv2.cvtColor(refined, cv2.COLOR_GRAY2BGR)


def _adaptive_preprocess_by_mean_intensity(
    image_bgr: np.ndarray,
    time_tag: Optional[str] = None,
) -> tuple[np.ndarray, dict[str, float | str]]:
    mean_gray = _mean_gray_intensity(image_bgr)

    if _is_24h_time_tag(time_tag):
        mode = "24h_escura"
        processed = _dark_24h_image_enhance(image_bgr)
        return processed, {"mode": mode, "mean_gray": mean_gray}

    if _uses_0h_brightness_buckets(time_tag):
        if mean_gray >= BRIGHT_0H_IMAGE_THRESHOLD:
            mode = "clara"
            processed = _bright_image_homogeneous_border_enhance(image_bgr)
        elif mean_gray <= DARK_IMAGE_THRESHOLD:
            mode = "escura"
            processed = _dark_0h_image_enhance(image_bgr)
        else:
            mode = "media"
            processed = _medium_0h_image_enhance(image_bgr)
        return processed, {"mode": mode, "mean_gray": mean_gray}

    if mean_gray >= BRIGHT_IMAGE_THRESHOLD:
        mode = "clara"
        processed = _bright_image_homogeneous_border_enhance(image_bgr)
    elif mean_gray <= DARK_IMAGE_THRESHOLD:
        mode = "escura"
        processed = image_bgr.copy()
    elif mean_gray >= MID_BRIGHT_IMAGE_THRESHOLD:
        mode = "intermediaria_clara"
        processed = _adjust_gamma_bgr(image_bgr, gamma=1.25)
        processed = _clahe_lab_bgr(processed, clip_limit=1.9, tile_grid_size=(8, 8))
        processed = _mid_bright_image_enhance(processed)
    else:
        mode = "normal"
        processed = _clahe_lab_bgr(image_bgr, clip_limit=1.6, tile_grid_size=(8, 8))
    return processed, {"mode": mode, "mean_gray": mean_gray}


def _prepare_adaptive_input_for_pipeline(
    img_base: np.ndarray,
    time_tag: Optional[str] = None,
    enabled: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str]]:
    if not bool(enabled):
        img_gray = ensure_gray(img_base)
        return img_gray, ensure_rgb_uint8(img_base), {"mode": "desativado", "mean_gray": float(np.mean(img_gray))}

    img_bgr = _to_bgr_u8_for_adaptive_pipeline(img_base)
    processed_bgr, info = _adaptive_preprocess_by_mean_intensity(img_bgr, time_tag=time_tag)
    gray = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
    return gray, rgb, info


def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    source = np.asarray(mask).astype(bool)
    if not np.any(source):
        return source

    flood = source.astype(np.uint8) * 255
    padded = cv2.copyMakeBorder(flood, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    fill_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(padded, fill_mask, (0, 0), 255)
    background = padded[1:-1, 1:-1] > 0
    holes = ~background & ~source
    return source | holes


def _keep_largest_mask_component(mask: np.ndarray) -> np.ndarray:
    source = np.asarray(mask).astype(bool)
    if not np.any(source):
        return source
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        source.astype(np.uint8),
        connectivity=8,
        ltype=cv2.CV_32S,
    )
    if n_labels <= 1:
        return source
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def _suppress_0h_bright_lateral_leakage(mask: np.ndarray, roi: np.ndarray) -> np.ndarray:
    source = np.asarray(mask).astype(bool)
    roi_bool = np.asarray(roi).astype(bool)
    if source.shape != roi_bool.shape or not np.any(source):
        return source

    _, w = source.shape
    if w < 80:
        return source

    guard = max(6, int(round(0.12 * w)))
    border = max(2, int(round(0.015 * w)))
    side_pixels = int(np.count_nonzero(source[:, :guard]) + np.count_nonzero(source[:, w - guard:]))
    border_touch = bool(np.any(source[:, :border]) or np.any(source[:, w - border:]))
    total_pixels = int(np.count_nonzero(source))
    side_fraction = float(side_pixels) / max(float(total_pixels), 1.0)
    area_fraction = float(total_pixels) / max(float(source.size), 1.0)

    if (not border_touch) and side_fraction < 0.035:
        return source
    if area_fraction < 0.12 and side_fraction < 0.08:
        return source

    corridor = np.zeros_like(source, dtype=bool)
    corridor[:, guard:w - guard] = True
    candidate = source & roi_bool & corridor

    candidate = cv2.morphologyEx(
        candidate.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    ) > 0
    candidate = _fill_mask_holes(candidate)
    candidate = _keep_largest_mask_component(candidate & roi_bool & corridor)
    candidate_pixels = int(np.count_nonzero(candidate))
    if candidate_pixels == 0:
        return source

    preserved_fraction = float(candidate_pixels) / max(float(total_pixels), 1.0)
    if preserved_fraction < 0.45:
        return source
    return candidate


def _cleanup_24h_mask(mask: np.ndarray, roi: np.ndarray) -> np.ndarray:
    source = np.asarray(mask).astype(bool)
    roi_bool = np.asarray(roi).astype(bool)
    if source.shape != roi_bool.shape or not np.any(source):
        return source

    cleaned = _fill_mask_holes(source & roi_bool)
    cleaned = cv2.morphologyEx(
        cleaned.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    ) > 0
    cleaned = cv2.morphologyEx(
        cleaned.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ) > 0
    cleaned = _fill_mask_holes(cleaned & roi_bool)
    cleaned = _keep_largest_mask_component(cleaned & roi_bool)
    return cleaned if np.any(cleaned) else source


def _cleanup_bright_mode_mask(mask: np.ndarray, roi: np.ndarray, time_tag: Optional[str] = None) -> np.ndarray:
    source = np.asarray(mask).astype(bool)
    roi_bool = np.asarray(roi).astype(bool)
    if source.shape != roi_bool.shape or not np.any(source):
        return source

    cleaned = _fill_mask_holes(source)
    cleaned = cv2.morphologyEx(
        cleaned.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
    ) > 0
    cleaned = _fill_mask_holes(cleaned)
    cleaned = _keep_largest_mask_component(cleaned & roi_bool)
    if _is_0h_time_tag(time_tag):
        cleaned = _suppress_0h_bright_lateral_leakage(cleaned, roi_bool)
    return cleaned if np.any(cleaned) else source


def homomorphic_filter(img01: np.ndarray, sigma: float = 20.0, ksize: int = 101) -> np.ndarray:
    x = mat2gray(img01).astype(np.float32, copy=False)
    img_log = np.log1p(x)

    ksize = as_odd(ksize, min_value=3)

    illumination = cv2.GaussianBlur(
        img_log,
        (ksize, ksize),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REPLICATE,
    )
    reflectance = img_log - illumination
    img_homo = np.exp(reflectance)
    return mat2gray(img_homo)


def clahe_matlab_like(img01: np.ndarray, num_tiles: Tuple[int, int] = (24, 24), clip_limit: float = 0.08) -> np.ndarray:
    img_u8 = to_uint8_01(img01)
    tiles_y = max(2, int(num_tiles[0]))
    tiles_x = max(2, int(num_tiles[1]))
    cv_clip = max(0.01, float(clip_limit) * 25.0)
    clahe = cv2.createCLAHE(clipLimit=cv_clip, tileGridSize=(tiles_x, tiles_y))
    out_u8 = clahe.apply(img_u8)
    return out_u8.astype(np.float32) / 255.0


@contextmanager
def limit_native_threads(max_cores: int):
    prev_cv_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        if threadpool_limits is None:
            yield
        else:
            with threadpool_limits(limits=int(max_cores)):
                yield
    finally:
        cv2.setNumThreads(prev_cv_threads)


def gabor_response(
    src01: np.ndarray,
    lam: float,
    ang_deg: float,
    ksize: int,
    smooth_sigma_factor: float,
    gamma: float,
    phase_offset: float = 0.0,
) -> np.ndarray:
    lam = float(lam)
    sigma = max(1.0, float(smooth_sigma_factor) * lam)
    theta = np.deg2rad(float(ang_deg))
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lam, float(gamma), float(phase_offset), ktype=cv2.CV_32F)
    resp = cv2.filter2D(src01, cv2.CV_32F, kernel, borderType=cv2.BORDER_REPLICATE)
    mag = np.abs(resp)
    mag = cv2.GaussianBlur(
        mag,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    return mag.astype(np.float32, copy=False)


def morph_matlab_like(
    img01: np.ndarray,
    radius: int = 24,
    a_top: float = 8.0,
    a_both: float = 0.65,
    return_hat_energy: bool = False,
):
    src_u8 = to_uint8_01(img01)
    r = max(1, int(radius))
    k = 2 * r + 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    toph = cv2.morphologyEx(src_u8, cv2.MORPH_TOPHAT, se)
    both = cv2.morphologyEx(src_u8, cv2.MORPH_BLACKHAT, se)
    src01 = src_u8.astype(np.float32) / 255.0
    toph01 = toph.astype(np.float32) / 255.0
    both01 = both.astype(np.float32) / 255.0
    out = mat2gray(src01 + (float(a_top) * toph01) - (float(a_both) * both01))
    hat_energy = mat2gray(toph01 + both01)
    if return_hat_energy:
        return out, hat_energy
    return out


def gabor_feature_stack(
    img01: np.ndarray,
    wavelengths: Sequence[float],
    orientations_deg: Sequence[float],
    ksize: int = 31,
    smooth_sigma_factor: float = 0.5,
    gamma: float = 0.5,
    phase_offset: float = 0.0,
    parallel_workers: int = 2,
) -> np.ndarray:
    src = mat2gray(img01).astype(np.float32, copy=False)
    jobs = [(float(lam), float(ang)) for lam in wavelengths for ang in orientations_deg]
    if len(jobs) == 0:
        raise ValueError("Banco de Gabor vazio: informe ao menos 1 wavelength e 1 orientacao.")

    ksize = as_odd(ksize, min_value=3)

    workers = max(1, int(parallel_workers))
    if workers == 1 or len(jobs) == 1:
        feats = [gabor_response(src, lam, ang, ksize, smooth_sigma_factor, gamma, phase_offset=phase_offset) for lam, ang in jobs]
    else:
        workers = min(workers, len(jobs))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(gabor_response, src, lam, ang, ksize, smooth_sigma_factor, gamma, phase_offset)
                for lam, ang in jobs
            ]
            feats = [f.result() for f in futures]

    return np.stack(feats, axis=2)


def texture_score_map(img01: np.ndarray, local_win: int = 9) -> np.ndarray:
    src = mat2gray(img01).astype(np.float32, copy=False)
    gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad = mat2gray(grad)

    win = as_odd(local_win, min_value=3)
    mean = cv2.blur(src, (win, win))
    mean2 = cv2.blur(src * src, (win, win))
    var = np.maximum(mean2 - (mean * mean), 0.0)
    local_std = mat2gray(np.sqrt(var).astype(np.float32))

    score = 0.65 * grad + 0.35 * local_std
    return mat2gray(score.astype(np.float32))


def pick_wound_label(
    labels: np.ndarray,
    dist_center: np.ndarray,
    texture_score: np.ndarray,
    center_window_frac: float = 0.14,
) -> int:
    h, w = labels.shape
    r = max(3, int(round(min(h, w) * float(center_window_frac) * 0.5)))
    cy, cx = h // 2, w // 2
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    center_patch = labels[y0:y1, x0:x1]

    unique_labels = np.unique(labels)
    best_label = int(unique_labels[0])
    best_score = float("inf")

    for lb in unique_labels:
        m = labels == lb
        if not np.any(m):
            continue

        mean_dist = float(np.mean(dist_center[m]))
        mean_tex = float(np.mean(texture_score[m]))
        center_ratio = float(np.mean(center_patch == lb))
        score = (1.35 * mean_dist) + (2.10 * mean_tex) - (0.40 * center_ratio)

        if score < best_score:
            best_score = score
            best_label = int(lb)

    return best_label


def smooth_mask(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    r = max(0, int(radius))
    if r == 0:
        return mask.astype(bool)

    k = 2 * r + 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = (mask.astype(np.uint8) * 255)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, se)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, se)
    return m > 0


def imsegkmeans_like(
    feature_set: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 0,
    batch_size: int = 8192,
    max_iter: int = 200,
) -> np.ndarray:
    h, w, c = feature_set.shape
    x = feature_set.reshape(-1, c).astype(np.float32, copy=False)

    mu = x.mean(axis=0, keepdims=True, dtype=np.float32)
    sd = x.std(axis=0, keepdims=True, dtype=np.float32)
    sd = np.maximum(sd, 1e-6)
    xn = (x - mu) / sd

    try:
        # Import lazily so the UI can open even when scikit-learn is absent or slow to import.
        from sklearn.cluster import MiniBatchKMeans

        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            n_init=1,
            random_state=random_state,
            batch_size=min(int(batch_size), xn.shape[0]),
            max_iter=int(max_iter),
        )
        labels = km.fit_predict(xn)
    except Exception:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, int(max_iter), 1e-4)
        cv2.setRNGSeed(int(random_state))
        _, labels, _ = cv2.kmeans(
            xn,
            int(n_clusters),
            None,
            criteria,
            1,
            cv2.KMEANS_PP_CENTERS,
        )
        labels = labels.reshape(-1)
    return labels.reshape(h, w)


def keep_largest_cc(mask: np.ndarray) -> np.ndarray:
    bin255 = mask.astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin255, connectivity=8, ltype=cv2.CV_32S)
    if n <= 1:
        return mask.astype(bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def remove_small_objects_compat(mask: np.ndarray, min_area: int) -> np.ndarray:
    bin255 = mask.astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin255, connectivity=8, ltype=cv2.CV_32S)
    if n <= 1:
        return np.zeros_like(mask, dtype=bool)

    keep_labels = np.where(stats[1:, cv2.CC_STAT_AREA] >= int(min_area))[0] + 1
    if keep_labels.size == 0:
        return np.zeros_like(mask, dtype=bool)

    lut = np.zeros(n, dtype=np.uint8)
    lut[keep_labels] = 1
    return lut[labels].astype(bool)


def postprocess_mask(mask: np.ndarray, min_area: int) -> np.ndarray:
    out = binary_fill_holes(mask.astype(bool))
    out = remove_small_objects_compat(out, min_area=int(min_area))
    out = keep_largest_cc(out)
    return out.astype(bool)


def overlay_perimeter(img_rgb_u8: np.ndarray, perim_mask: np.ndarray, color_rgb_255: Tuple[int, int, int]) -> np.ndarray:
    out = img_rgb_u8.copy()
    out[perim_mask] = np.array(color_rgb_255, dtype=np.uint8)
    return out


def overlay_mask_alpha(
    img_rgb_u8: np.ndarray,
    mask: np.ndarray,
    color_rgb_255: Tuple[int, int, int],
    alpha: float = 0.35,
) -> np.ndarray:
    out = img_rgb_u8.astype(np.float32, copy=True)
    color = np.asarray(color_rgb_255, dtype=np.float32)
    alpha_clamped = float(np.clip(alpha, 0.0, 1.0))
    mask_bool = np.asarray(mask).astype(bool)
    if np.any(mask_bool):
        out[mask_bool] = ((1.0 - alpha_clamped) * out[mask_bool]) + (alpha_clamped * color)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def perimeter_mask(mask: np.ndarray) -> np.ndarray:
    bin255 = mask.astype(np.uint8) * 255
    er = cv2.erode(bin255, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    per = cv2.subtract(bin255, er)
    return per > 0


def labeloverlay_like(gray01: np.ndarray, mask: np.ndarray, color_rgb01=(1.0, 0.0, 0.0), alpha: float = 0.55) -> np.ndarray:
    base = np.dstack([gray01, gray01, gray01]).astype(np.float32)
    out = base.copy()
    color = np.array(color_rgb01, dtype=np.float32).reshape(1, 1, 3)
    m = mask.astype(bool)
    out[m] = (1.0 - alpha) * base[m] + alpha * color
    return np.clip(out, 0.0, 1.0)


@dataclass(frozen=True)
class CORAConfig:
    r_auto: int = 2
    col_auto: Tuple[int, int, int] = (255, 0, 0)
    frac_min_area: float = 0.0005
    min_area_floor: int = 200
    min_area_fixed: int = 5000
    proc_scale: float = 0.75
    gabor_wavelengths: Tuple[float, ...] = (2, 4, 8, 16, 32)
    gabor_orientations_deg: Tuple[float, ...] = (0, 45, 90, 135, 180)
    center_dist_weight: float = 2.6
    texture_weight: float = 3.0
    texture_reject_quantile: float = 0.70
    hat_texture_mix: float = 0.50
    texture_window: int = 9
    boundary_smooth_radius: int = 3
    max_cores: int = 4
    gabor_iterations: int = 10
    kmeans_max_iter: int = 150
    max_processing_time_s: float = 30.0
    homomorphic_sigma: float = 20.0
    homomorphic_ksize: int = 101
    clahe_num_tiles: Tuple[int, int] = (31, 31)
    clahe_clip_limit: float = 0.08
    morph_radius: int = 24
    morph_a_top: float = 8.0
    morph_a_both: float = 0.65
    gabor_ksize: int = 31
    gabor_smooth_sigma_factor: float = 0.5
    gabor_gamma: float = 0.5
    center_window_frac: float = 0.14
    kmeans_random_state: int = 0
    kmeans_batch_size: int = 8192
    kmeans_probe_batch_size: int = 4096
    kmeans_probe_max_samples: int = 100000
    adaptive_preprocess_enabled: bool = True
    adaptive_cleanup_enabled: bool = True


@dataclass
class AreaResults:
    area_manual: int
    area_auto: int
    erro_abs: int
    erro_pct: float
    area_ratio: float
    area_diff_norm: float
    processing_time_s: float


@dataclass
class CORAArtifacts:
    base_rgb_u8: np.ndarray
    mask_auto: np.ndarray
    roi_mask: np.ndarray
    contour_mask: np.ndarray
    contour_overlay_rgb_u8: np.ndarray
    mask_overlay_rgb_u8: np.ndarray
    effective_gabor_iterations: int
    requested_gabor_iterations: int
    effective_kmeans_iter: int
    requested_kmeans_iter: int
    hat_ratio: float
    tex_ratio: float
    max_cores: int
    max_processing_time_s: float


def _apply_config_overrides(config: CORAConfig, overrides: Mapping[str, Any]) -> CORAConfig:
    """Aplica overrides dinamicos em CORAConfig com validacao de chaves."""
    if not overrides:
        return config

    valid_fields = {f.name for f in fields(CORAConfig)}
    unknown = sorted(set(overrides) - valid_fields)
    if unknown:
        names = ", ".join(unknown)
        raise TypeError(f"Parametros desconhecidos para CORAConfig: {names}")
    return replace(config, **overrides)


def _normalize_config(config: CORAConfig) -> CORAConfig:
    """Normaliza limites e tipos antes da execucao do pipeline."""
    if not (0.1 <= float(config.proc_scale) <= 1.0):
        raise ValueError("proc_scale deve estar no intervalo [0.1, 1.0].")

    max_cores = int(config.max_cores)
    if not (1 <= max_cores <= 4):
        raise ValueError("max_cores deve estar no intervalo [1, 4].")

    gabor_wavelengths = tuple(float(v) for v in config.gabor_wavelengths)
    gabor_orientations = tuple(float(v) for v in config.gabor_orientations_deg)
    if not gabor_wavelengths or not gabor_orientations:
        raise ValueError("Informe ao menos 1 wavelength e 1 orientacao de Gabor.")

    col_auto = tuple(int(np.clip(c, 0, 255)) for c in config.col_auto)

    return replace(
        config,
        r_auto=max(0, int(config.r_auto)),
        col_auto=col_auto,
        proc_scale=float(config.proc_scale),
        gabor_wavelengths=gabor_wavelengths,
        gabor_orientations_deg=gabor_orientations,
        texture_window=max(3, int(config.texture_window)),
        boundary_smooth_radius=max(0, int(config.boundary_smooth_radius)),
        max_cores=max_cores,
        gabor_iterations=max(1, int(config.gabor_iterations)),
        kmeans_max_iter=max(50, int(config.kmeans_max_iter)),
        max_processing_time_s=max(5.0, float(config.max_processing_time_s)),
        homomorphic_ksize=as_odd(config.homomorphic_ksize, min_value=3),
        clahe_num_tiles=(max(2, int(config.clahe_num_tiles[0])), max(2, int(config.clahe_num_tiles[1]))),
        morph_radius=max(1, int(config.morph_radius)),
        gabor_ksize=as_odd(config.gabor_ksize, min_value=3),
        center_window_frac=max(0.02, float(config.center_window_frac)),
        kmeans_batch_size=max(256, int(config.kmeans_batch_size)),
        kmeans_probe_batch_size=max(256, int(config.kmeans_probe_batch_size)),
        kmeans_probe_max_samples=max(1000, int(config.kmeans_probe_max_samples)),
        adaptive_preprocess_enabled=bool(config.adaptive_preprocess_enabled),
        adaptive_cleanup_enabled=bool(config.adaptive_cleanup_enabled),
    )


def _resize_gray_for_processing(img_gray: np.ndarray, proc_scale: float) -> np.ndarray:
    """Reduz a imagem conforme proc_scale para equilibrar custo e qualidade."""
    if float(proc_scale) >= 0.999:
        return img_gray
    orig_h, orig_w = img_gray.shape
    work_w = max(8, int(round(orig_w * float(proc_scale))))
    work_h = max(8, int(round(orig_h * float(proc_scale))))
    return cv2.resize(img_gray, (work_w, work_h), interpolation=cv2.INTER_AREA)


def _preprocess_maps(img_gray_work: np.ndarray, config: CORAConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Executa pre-processamento e retorna imagem morfologica + energia de hat."""
    mat1 = mat2gray(img_gray_work)
    mat2 = homomorphic_filter(mat1, sigma=config.homomorphic_sigma, ksize=config.homomorphic_ksize)
    img_eq = clahe_matlab_like(mat2, num_tiles=config.clahe_num_tiles, clip_limit=config.clahe_clip_limit)
    return morph_matlab_like(
        img_eq,
        radius=config.morph_radius,
        a_top=config.morph_a_top,
        a_both=config.morph_a_both,
        return_hat_energy=True,
    )


def _estimate_gabor_iterations(img_morph: np.ndarray, config: CORAConfig, t0: float) -> int:
    """Estima numero de iteracoes de Gabor dentro do orcamento de tempo."""
    gabor_jobs_per_iter = len(config.gabor_wavelengths) * len(config.gabor_orientations_deg)
    probe_t0 = time.perf_counter()
    _ = gabor_response(
        img_morph,
        config.gabor_wavelengths[0],
        config.gabor_orientations_deg[0],
        config.gabor_ksize,
        config.gabor_smooth_sigma_factor,
        config.gabor_gamma,
        phase_offset=0.0,
    )
    probe_s = max(time.perf_counter() - probe_t0, 1e-4)
    remaining_after_probe = config.max_processing_time_s - (time.perf_counter() - t0)
    reserve_for_kmeans = min(12.0, max(4.0, 0.38 * config.max_processing_time_s))
    gabor_budget = max(1.0, remaining_after_probe - reserve_for_kmeans)
    est_gabor_iter_s = 1.20 * probe_s * (gabor_jobs_per_iter / max(1.0, float(config.max_cores)))
    return min(
        config.gabor_iterations,
        max(1, int(np.floor(gabor_budget / max(est_gabor_iter_s, 1e-4)))),
    )


def _aggregate_gabor_features(
    img_morph: np.ndarray,
    config: CORAConfig,
    gabor_iterations: int,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    phase_offsets = np.linspace(0.0, np.pi / 2.0, num=gabor_iterations, endpoint=True, dtype=np.float32)
    gabor_acc = None
    for idx, phase in enumerate(phase_offsets, start=1):
        feats_i = gabor_feature_stack(
            img_morph,
            wavelengths=config.gabor_wavelengths,
            orientations_deg=config.gabor_orientations_deg,
            ksize=config.gabor_ksize,
            smooth_sigma_factor=config.gabor_smooth_sigma_factor,
            gamma=config.gabor_gamma,
            phase_offset=float(phase),
            parallel_workers=config.max_cores,
        )
        gabor_acc = feats_i if gabor_acc is None else (gabor_acc + feats_i)
        if progress_callback is not None:
            try:
                progress_callback(idx, gabor_iterations)
            except Exception:
                pass
    return (gabor_acc / float(gabor_iterations)).astype(np.float32, copy=False)


def _build_auxiliary_maps(img_morph: np.ndarray, hat_energy: np.ndarray, config: CORAConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Constroi mapas auxiliares de distancia e textura para o classificador."""
    h, w = img_morph.shape
    yy, xx = np.ogrid[:h, :w]
    dist_center = mat2gray(np.hypot(xx - (w / 2.0), yy - (h / 2.0)).astype(np.float32))
    texture_base = texture_score_map(img_morph, local_win=config.texture_window)
    hat_mix = float(np.clip(config.hat_texture_mix, 0.0, 1.0))
    texture_score = mat2gray(((1.0 - hat_mix) * texture_base) + (hat_mix * hat_energy))
    return dist_center, texture_score


# PT: Empacota as features finais (Gabor + distancia + textura) para o k-means. | EN: Packs the final features (Gabor + distance + texture) for k-means.
def _build_feature_set(
    gabor_feats: np.ndarray,
    dist_center: np.ndarray,
    texture_score: np.ndarray,
    config: CORAConfig,
) -> np.ndarray:
    return np.concatenate(
        [
            gabor_feats,
            (float(config.center_dist_weight) * dist_center)[:, :, None],
            (float(config.texture_weight) * texture_score)[:, :, None],
        ],
        axis=2,
    ).astype(np.float32, copy=False)


def _estimate_kmeans_iterations(feature_set: np.ndarray, config: CORAConfig, t0: float) -> int:
    """Define teto de iteracoes do k-means em funcao do tamanho dos dados."""
    remaining_for_kmeans = config.max_processing_time_s - (time.perf_counter() - t0)
    effective_kmeans_iter = config.kmeans_max_iter
    if remaining_for_kmeans <= 1.0:
        return min(effective_kmeans_iter, 30)
    if config.kmeans_max_iter <= 40:
        return effective_kmeans_iter

    probe_iters = min(20, max(8, config.kmeans_max_iter // 10))
    flat_feats = feature_set.reshape(-1, feature_set.shape[2])
    if flat_feats.shape[0] > config.kmeans_probe_max_samples:
        step = int(np.ceil(float(flat_feats.shape[0]) / float(config.kmeans_probe_max_samples)))
        probe_flat = flat_feats[::step]
    else:
        probe_flat = flat_feats

    probe_set = probe_flat.reshape(probe_flat.shape[0], 1, feature_set.shape[2])
    probe_t0 = time.perf_counter()
    _ = imsegkmeans_like(
        probe_set,
        n_clusters=2,
        random_state=config.kmeans_random_state,
        batch_size=config.kmeans_probe_batch_size,
        max_iter=probe_iters,
    )
    probe_s = max(time.perf_counter() - probe_t0, 1e-4)
    est_kmeans_s = probe_s * (
        float(flat_feats.shape[0]) / max(1.0, float(probe_flat.shape[0]))
    ) * (
        float(config.kmeans_max_iter) / max(1.0, float(probe_iters))
    )
    available_for_kmeans = max(1.0, config.max_processing_time_s - (time.perf_counter() - t0) - 1.0)
    if est_kmeans_s <= available_for_kmeans:
        return effective_kmeans_iter

    scale = available_for_kmeans / est_kmeans_s
    return max(20, int(np.floor(config.kmeans_max_iter * scale)))


def contour_mask_with_thickness(mask: np.ndarray, radius: int) -> np.ndarray:
    contour = perimeter_mask(mask)
    if int(radius) <= 0:
        return contour

    rk = as_odd((2 * int(radius)) + 1, min_value=3)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rk, rk))
    return cv2.dilate((contour.astype(np.uint8) * 255), se) > 0


def _show_results_plot(
    img_gray: np.ndarray,
    base_rgb_u8: np.ndarray,
    mask_auto: np.ndarray,
    area_auto: int,
    erro_pct: float,
    config: CORAConfig,
) -> None:
    auto_perim = contour_mask_with_thickness(mask_auto, radius=config.r_auto)
    img_contours = overlay_perimeter(base_rgb_u8, auto_perim, config.col_auto)
    filled_overlay = overlay_mask_alpha(base_rgb_u8, mask_auto, config.col_auto, alpha=0.35)
    filled_overlay = overlay_perimeter(filled_overlay, auto_perim, config.col_auto)

    fig = plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(img_contours)
    ax1.set_title("Segmentacao automatica (contorno)")
    ax1.axis("off")

    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(filled_overlay)
    ax2.set_title(f"Area auto = {area_auto:.0f} px | Erro = {erro_pct:.2f}%")
    ax2.axis("off")

    fig.suptitle("Avaliacao baseada exclusivamente em area (pixels)")
    plt.tight_layout()
    plt.show()


# PT: Pipeline principal executado pela GUI e pelo modo CLI. | EN: Main pipeline executed by the GUI and CLI modes.
def run_cora(
    base_path: str,
    area_manual: int,
    config: Optional[CORAConfig] = None,
    roi_mask: Optional[np.ndarray] = None,
    show: bool = True,
    verbose: bool = True,
    return_artifacts: bool = False,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    time_tag: Optional[str] = None,
    **overrides: Any,
) -> Union[AreaResults, Tuple[AreaResults, CORAArtifacts]]:
    """Pipeline completo: leitura, segmentacao, pos-processamento e artefatos de saida."""
    t0 = time.perf_counter()

    def emit_progress(stage: str, value01: float) -> None:
        if progress_callback is None:
            return
        try:
            clamped = float(np.clip(value01, 0.0, 1.0))
            progress_callback(stage, clamped)
        except Exception:
            pass

    emit_progress("Carregando imagem", 0.02)
    if isinstance(config, dict):
        merged_overrides = {**config, **overrides}
        base_config = CORAConfig()
    elif isinstance(config, CORAConfig) or config is None:
        merged_overrides = overrides
        base_config = config or CORAConfig()
    else:
        # PT: Compatibilidade: terceiro argumento posicional antigo era r_auto. | EN: Compatibility: the former third positional argument was r_auto.
        merged_overrides = {"r_auto": config, **overrides}
        base_config = CORAConfig()

    cfg = _apply_config_overrides(base_config, merged_overrides)
    cfg = _normalize_config(cfg)
    emit_progress("Configurando parametros", 0.06)

    gabor_iterations_requested = cfg.gabor_iterations
    kmeans_max_iter_requested = cfg.kmeans_max_iter

    img_base = read_image(base_path)
    emit_progress("Analisando brilho medio", 0.08)
    img_gray, base_rgb_u8, adaptive_info = _prepare_adaptive_input_for_pipeline(
        img_base,
        time_tag=time_tag,
        enabled=cfg.adaptive_preprocess_enabled,
    )
    emit_progress(
        f"Modo adaptativo: {adaptive_info['mode']} (media={float(adaptive_info['mean_gray']):.1f})",
        0.12,
    )
    emit_progress("Preparando ROI", 0.14)
    orig_h, orig_w = img_gray.shape
    if roi_mask is None:
        roi_full = np.ones((orig_h, orig_w), dtype=bool)
    else:
        roi_arr = np.asarray(roi_mask).astype(bool)
        if roi_arr.shape != (orig_h, orig_w):
            raise ValueError(
                "roi_mask deve ter o mesmo tamanho da imagem base: "
                f"esperado {(orig_h, orig_w)}, recebido {roi_arr.shape}."
            )
        if not np.any(roi_arr):
            raise ValueError("roi_mask nao pode ser vazio.")
        roi_full = roi_arr

    img_gray_work = _resize_gray_for_processing(img_gray, cfg.proc_scale)
    if img_gray_work.shape != img_gray.shape:
        roi_work = cv2.resize(
            roi_full.astype(np.uint8),
            (img_gray_work.shape[1], img_gray_work.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
    else:
        roi_work = roi_full.copy()

    with limit_native_threads(cfg.max_cores):
        emit_progress("Pre-processando imagem", 0.22)
        img_morph, hat_energy = _preprocess_maps(img_gray_work, cfg)
        emit_progress("Estimando iteracoes de Gabor", 0.30)
        effective_gabor_iterations = _estimate_gabor_iterations(img_morph, cfg, t0)
        emit_progress("Executando filtros de Gabor", 0.36)
        gabor_feats = _aggregate_gabor_features(
            img_morph,
            cfg,
            effective_gabor_iterations,
            progress_callback=lambda idx, total: emit_progress(
                f"Executando filtros de Gabor ({idx}/{total})",
                0.36 + (0.36 * (float(idx) / float(max(total, 1)))),
            ),
        )
        emit_progress("Montando mapa de textura", 0.75)
        dist_center, texture_score = _build_auxiliary_maps(img_morph, hat_energy, cfg)
        feature_set = _build_feature_set(gabor_feats, dist_center, texture_score, cfg)
        emit_progress("Estimando iteracoes do k-means", 0.82)
        effective_kmeans_iter = _estimate_kmeans_iterations(feature_set, cfg, t0)
        emit_progress("Separando regioes (k-means)", 0.88)
        labels = imsegkmeans_like(
            feature_set,
            n_clusters=2,
            random_state=cfg.kmeans_random_state,
            batch_size=cfg.kmeans_batch_size,
            max_iter=effective_kmeans_iter,
        )

    emit_progress("Pos-processando mascara", 0.93)
    wound_label = pick_wound_label(
        labels,
        dist_center=dist_center,
        texture_score=texture_score,
        center_window_frac=cfg.center_window_frac,
    )
    mask_auto = (labels == wound_label) & roi_work

    if 0.0 < float(cfg.texture_reject_quantile) < 1.0 and np.any(mask_auto):
        tex_vals = texture_score[mask_auto]
        if tex_vals.size > 32:
            thr = float(np.quantile(tex_vals, float(cfg.texture_reject_quantile)))
            mask_auto = mask_auto & (texture_score <= thr)

    mask_auto = smooth_mask(mask_auto, radius=cfg.boundary_smooth_radius) & roi_work

    area_scale = float(cfg.proc_scale) * float(cfg.proc_scale)
    min_area_floor_work = max(1, int(round(int(cfg.min_area_floor) * area_scale)))
    min_area_fixed_work = max(1, int(round(int(cfg.min_area_fixed) * area_scale)))
    roi_work_area = int(np.count_nonzero(roi_work))
    min_area_auto = max(min_area_floor_work, int(round(cfg.frac_min_area * max(roi_work_area, 1))), min_area_fixed_work)
    mask_auto = postprocess_mask(mask_auto, min_area=min_area_auto) & roi_work

    if img_gray_work.shape != img_gray.shape:
        mask_auto = cv2.resize(mask_auto.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST) > 0
        resize_smooth_radius = max(1, cfg.boundary_smooth_radius - 1)
        mask_auto = smooth_mask(mask_auto, radius=resize_smooth_radius)
        mask_auto = keep_largest_cc(mask_auto)

    mask_auto = mask_auto & roi_full
    if np.any(mask_auto):
        mask_auto = keep_largest_cc(mask_auto)

    if cfg.adaptive_cleanup_enabled and np.any(mask_auto):
        emit_progress("Refinando mascara por modo adaptativo", 0.96)
        if _is_24h_time_tag(time_tag):
            mask_auto = _cleanup_24h_mask(mask_auto, roi_full)
        elif str(adaptive_info.get("mode", "")).strip().lower() == "clara":
            mask_auto = _cleanup_bright_mode_mask(mask_auto, roi_full, time_tag=time_tag)
        mask_auto = mask_auto & roi_full
        if np.any(mask_auto):
            mask_auto = keep_largest_cc(mask_auto)

    if np.count_nonzero(mask_auto) == 0:
        raise RuntimeError("Mascara automatica ficou vazia apos o pos-processamento.")

    if mask_auto.shape != hat_energy.shape:
        mask_sep = cv2.resize(
            mask_auto.astype(np.uint8),
            (hat_energy.shape[1], hat_energy.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
    else:
        mask_sep = mask_auto

    if np.any(mask_sep) and np.any(~mask_sep):
        hat_w = float(np.mean(hat_energy[mask_sep]))
        hat_c = float(np.mean(hat_energy[~mask_sep]))
        tex_w = float(np.mean(texture_score[mask_sep]))
        tex_c = float(np.mean(texture_score[~mask_sep]))
        hat_ratio = hat_c / max(hat_w, 1e-8)
        tex_ratio = tex_c / max(tex_w, 1e-8)
    else:
        hat_ratio = float("nan")
        tex_ratio = float("nan")

    area_auto = int(np.count_nonzero(mask_auto))
    erro_abs = int(abs(area_auto - int(area_manual)))
    erro_pct = 100.0 * float(erro_abs) / max(int(area_manual), 1)
    area_ratio = float(area_auto) / max(float(area_manual), 1.0)
    area_diff_norm = float(area_auto - int(area_manual)) / max(float(area_manual), 1.0)
    processing_time_s = time.perf_counter() - t0
    emit_progress("Calculando metricas", 0.98)
    results = AreaResults(
        area_manual=int(area_manual),
        area_auto=area_auto,
        erro_abs=erro_abs,
        erro_pct=float(erro_pct),
        area_ratio=float(area_ratio),
        area_diff_norm=float(area_diff_norm),
        processing_time_s=float(processing_time_s),
    )

    contour_mask = contour_mask_with_thickness(mask_auto, radius=cfg.r_auto)
    contour_overlay = overlay_perimeter(base_rgb_u8, contour_mask, cfg.col_auto)
    mask_overlay = overlay_mask_alpha(base_rgb_u8, mask_auto, cfg.col_auto, alpha=0.35)
    mask_overlay = overlay_perimeter(mask_overlay, contour_mask, cfg.col_auto)
    artifacts = CORAArtifacts(
        base_rgb_u8=base_rgb_u8,
        mask_auto=mask_auto,
        roi_mask=roi_full,
        contour_mask=contour_mask,
        contour_overlay_rgb_u8=contour_overlay,
        mask_overlay_rgb_u8=mask_overlay,
        effective_gabor_iterations=effective_gabor_iterations,
        requested_gabor_iterations=gabor_iterations_requested,
        effective_kmeans_iter=effective_kmeans_iter,
        requested_kmeans_iter=kmeans_max_iter_requested,
        hat_ratio=hat_ratio,
        tex_ratio=tex_ratio,
        max_cores=cfg.max_cores,
        max_processing_time_s=cfg.max_processing_time_s,
    )

    if verbose:
        print("\n=========== RESULTADOS (AREA) ===========")
        print(f"Area manual (ref) : {results.area_manual:.0f} px")
        print(f"Area automatica   : {results.area_auto:.0f} px")
        print(f"Erro absoluto     : {results.erro_abs:.0f} px")
        print(f"Erro relativo     : {results.erro_pct:.2f} %")
        print(f"Razao de area     : {results.area_ratio:.3f}")
        print(f"Modo adaptativo   : {adaptive_info['mode']} (media={float(adaptive_info['mean_gray']):.1f})")
        print(f"Limite de nucleos : {cfg.max_cores}")
        print(f"Gabor iteracoes   : {effective_gabor_iterations}/{gabor_iterations_requested}")
        print(f"K-means max_iter  : {effective_kmeans_iter}/{kmeans_max_iter_requested}")
        print(f"Orcamento tempo   : {cfg.max_processing_time_s:.1f} s")
        print(f"Tempo processamento: {results.processing_time_s:.3f} s")
        if results.processing_time_s > cfg.max_processing_time_s:
            print("Aviso             : Tempo excedeu o limite configurado; reduza iters ou escala.")
        print(f"Separacao hat (cel/cic): {hat_ratio:.3f} | Separacao textura (cel/cic): {tex_ratio:.3f}")
        print("Tendencia         : Supersegmentacao" if results.area_ratio > 1 else "Tendencia         : Subsegmentacao")

    if show:
        _show_results_plot(img_gray, base_rgb_u8, mask_auto, area_auto, erro_pct, cfg)

    emit_progress("Concluido", 1.0)
    if return_artifacts:
        return results, artifacts
    return results


def build_parser() -> argparse.ArgumentParser:
    """Monta parser CLI para execucao standalone do algoritmo."""
    defaults = CORAConfig()
    parser = argparse.ArgumentParser(description="Conversao MATLAB -> Python para avaliacao de area de scratch.")
    parser.add_argument(
        "--base",
        type=str,
        default=BASE_IMAGE_PATH,
        help="Caminho da imagem base. Se vazio, usa BASE_IMAGE_PATH definido no codigo.",
    )
    parser.add_argument("--area_manual", type=int, default=403281, help="Area manual em pixels (medicao externa).")
    parser.add_argument(
        "--gabor_wavelengths",
        type=float,
        nargs="+",
        default=list(defaults.gabor_wavelengths),
        help="Comprimentos de onda do banco de Gabor.",
    )
    parser.add_argument(
        "--gabor_orientations",
        type=float,
        nargs="+",
        default=list(defaults.gabor_orientations_deg),
        help="Orientacoes (graus) do banco de Gabor.",
    )
    parser.add_argument(
        "--clahe_num_tiles",
        type=int,
        nargs=2,
        metavar=("TILES_Y", "TILES_X"),
        default=list(defaults.clahe_num_tiles),
        help="Numero de tiles CLAHE em Y e X.",
    )

    scalar_specs = [
        ("--r_auto", int, defaults.r_auto, "Espessura do contorno automatico."),
        ("--min_area_fixed", int, defaults.min_area_fixed, "Equivalente ao bwareaopen(..., 5000)."),
        ("--frac_min_area", float, defaults.frac_min_area, "Fator minimo proporcional de area."),
        ("--min_area_floor", int, defaults.min_area_floor, "Piso absoluto de area minima."),
        ("--proc_scale", float, defaults.proc_scale, "Escala de processamento [0.1, 1.0]. Menor = mais rapido."),
        ("--gabor_iterations", int, defaults.gabor_iterations, "Numero de iteracoes/fases do banco de Gabor."),
        ("--kmeans_max_iter", int, defaults.kmeans_max_iter, "Numero maximo de iteracoes do MiniBatchKMeans."),
        ("--max_processing_time_s", float, defaults.max_processing_time_s, "Orcamento maximo de tempo em segundos."),
        ("--center_dist_weight", float, defaults.center_dist_weight, "Peso da distancia ao centro nas features."),
        ("--texture_weight", float, defaults.texture_weight, "Peso da textura nas features."),
        ("--texture_reject_quantile", float, defaults.texture_reject_quantile, "Rejeita topo de textura na mascara inicial."),
        ("--hat_texture_mix", float, defaults.hat_texture_mix, "Mistura da energia top+bottom-hat na textura [0..1]."),
        ("--texture_window", int, defaults.texture_window, "Janela para variancia local da textura."),
        ("--boundary_smooth_radius", int, defaults.boundary_smooth_radius, "Suavizacao morfologica da borda."),
        ("--max_cores", int, defaults.max_cores, "Limite de nucleos para processamento paralelo [1..4]."),
        ("--homomorphic_sigma", float, defaults.homomorphic_sigma, "Sigma do filtro homomorfico."),
        ("--homomorphic_ksize", int, defaults.homomorphic_ksize, "Kernel size do filtro homomorfico."),
        ("--clahe_clip_limit", float, defaults.clahe_clip_limit, "Clip limit do CLAHE."),
        ("--morph_radius", int, defaults.morph_radius, "Raio da morfologia top/black-hat."),
        ("--morph_a_top", float, defaults.morph_a_top, "Peso do top-hat."),
        ("--morph_a_both", float, defaults.morph_a_both, "Peso do black-hat."),
        ("--gabor_ksize", int, defaults.gabor_ksize, "Kernel size dos filtros de Gabor."),
        ("--gabor_smooth_sigma_factor", float, defaults.gabor_smooth_sigma_factor, "Fator de sigma da suavizacao Gabor."),
        ("--gabor_gamma", float, defaults.gabor_gamma, "Gamma (aspect ratio) dos filtros de Gabor."),
        ("--center_window_frac", float, defaults.center_window_frac, "Fracao da janela central para escolher o cluster."),
    ]
    for flag, value_type, default_value, help_text in scalar_specs:
        parser.add_argument(flag, type=value_type, default=default_value, help=help_text)

    parser.add_argument("--no_show", action="store_true", help="Nao exibe figura final.")
    return parser


def config_from_args(args: argparse.Namespace) -> CORAConfig:
    """Converte argumentos parseados em um objeto de configuracao validado."""
    return CORAConfig(
        r_auto=args.r_auto,
        frac_min_area=args.frac_min_area,
        min_area_floor=args.min_area_floor,
        min_area_fixed=args.min_area_fixed,
        proc_scale=args.proc_scale,
        gabor_wavelengths=tuple(args.gabor_wavelengths),
        gabor_orientations_deg=tuple(args.gabor_orientations),
        center_dist_weight=args.center_dist_weight,
        texture_weight=args.texture_weight,
        texture_reject_quantile=args.texture_reject_quantile,
        hat_texture_mix=args.hat_texture_mix,
        texture_window=args.texture_window,
        boundary_smooth_radius=args.boundary_smooth_radius,
        max_cores=args.max_cores,
        gabor_iterations=args.gabor_iterations,
        kmeans_max_iter=args.kmeans_max_iter,
        max_processing_time_s=args.max_processing_time_s,
        homomorphic_sigma=args.homomorphic_sigma,
        homomorphic_ksize=args.homomorphic_ksize,
        clahe_num_tiles=tuple(args.clahe_num_tiles),
        clahe_clip_limit=args.clahe_clip_limit,
        morph_radius=args.morph_radius,
        morph_a_top=args.morph_a_top,
        morph_a_both=args.morph_a_both,
        gabor_ksize=args.gabor_ksize,
        gabor_smooth_sigma_factor=args.gabor_smooth_sigma_factor,
        gabor_gamma=args.gabor_gamma,
        center_window_frac=args.center_window_frac,
    )


def main() -> None:
    """Executa o pipeline via linha de comando para uma imagem informada."""
    parser = build_parser()
    args = parser.parse_args()
    base_path = str(args.base).strip()
    if not base_path:
        raise ValueError(
            "Defina o caminho da imagem em BASE_IMAGE_PATH no codigo "
            "ou passe --base no terminal."
        )
    config = config_from_args(args)
    run_cora(
        base_path=base_path,
        area_manual=args.area_manual,
        config=config,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()

