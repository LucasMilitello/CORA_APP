"""Worker de processamento em lote com pipeline adaptativo baseado em proj_ana.py."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import replace
from pathlib import Path
import time
import uuid

import cv2
import numpy as np

try:
    if __package__:
        from ..matlab_style_cora import (
            CORAArtifacts,
            AreaResults,
            contour_mask_with_thickness,
            overlay_perimeter,
            run_cora,
        )
    else:
        from cora_projeto.matlab_style_cora import (
            CORAArtifacts,
            AreaResults,
            contour_mask_with_thickness,
            overlay_perimeter,
            run_cora,
        )
except ImportError:
    from matlab_style_cora import (
        CORAArtifacts,
        AreaResults,
        contour_mask_with_thickness,
        overlay_perimeter,
        run_cora,
    )

try:
    if __package__:
        from .performance_service import ProcessResourceMonitor
    else:
        from cora_projeto.services.performance_service import ProcessResourceMonitor
except ImportError:
    from services.performance_service import ProcessResourceMonitor


BRIGHT_IMAGE_THRESHOLD = 165.0
BRIGHT_0H_IMAGE_THRESHOLD = 155.0
MID_BRIGHT_IMAGE_THRESHOLD = 140.0
DARK_IMAGE_THRESHOLD = 105.0


def _app_text(app, key: str, fallback: str, **kwargs: object) -> str:
    translator = getattr(app, "_t", None)
    if callable(translator):
        try:
            return str(translator(key, **kwargs))
        except Exception:
            pass
    try:
        return fallback.format(**kwargs)
    except Exception:
        return fallback


class _ProjAnaProcessingAdapter:
    """Adaptador do pipeline de processamento do proj_ana para o app principal."""

    DEFAULT_RED_CONTOUR_MIN_LENGTH = 30.0

    def __init__(self, app, temp_dir: Path) -> None:
        self.app = app
        self.config = app.config
        self.temp_dir = temp_dir
        self.roi_by_item = getattr(app, "roi_by_item", {})
        self.processed_by_group = getattr(app, "processed_by_group", {})
        self.red_contour_min_length = self.DEFAULT_RED_CONTOUR_MIN_LENGTH
        self._worker_artifacts: dict[str, dict[str, CORAArtifacts]] = {}

    @staticmethod
    def _is_24h_time_tag(time_tag: str) -> bool:
        return str(time_tag).strip().lower() == "24h"

    @staticmethod
    def _is_0h_time_tag(time_tag: str | None) -> bool:
        return str(time_tag or "").strip().lower() == "0h"

    @staticmethod
    def _uses_0h_brightness_buckets(time_tag: str | None) -> bool:
        return str(time_tag or "").strip().lower() in {"0h", "img"}

    @staticmethod
    def _resize_roi_mask(roi_mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        roi_arr = np.asarray(roi_mask).astype(bool)
        if roi_arr.shape == shape:
            return roi_arr
        return cv2.resize(
            roi_arr.astype(np.uint8),
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0

    @staticmethod
    def _expand_mask_for_roi(mask: np.ndarray, padding_px: int) -> np.ndarray:
        roi = np.asarray(mask).astype(bool)
        if not np.any(roi):
            return roi
        kernel_size = max(3, (2 * int(max(1, padding_px))) + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.dilate(roi.astype(np.uint8), kernel, iterations=1) > 0

    def _artifacts_for_item(self, group_key: str, time_tag: str) -> CORAArtifacts | None:
        local = self._worker_artifacts.get(group_key, {}).get(time_tag)
        if local is not None:
            return local
        proc = self.processed_by_group.get(group_key, {}).get(time_tag)
        if proc is None:
            return None
        return getattr(proc, "artifacts", None)

    def _roi_for_item(self, group_key: str, time_tag: str) -> np.ndarray | None:
        key = (group_key, time_tag)
        roi = self.roi_by_item.get(key)
        if roi is None:
            artifacts = self._artifacts_for_item(group_key, time_tag)
            if artifacts is not None:
                roi = getattr(artifacts, "roi_mask", None)
        return None if roi is None else np.asarray(roi).astype(bool)

    def _derive_24h_roi_from_0h(
        self,
        group_key: str,
        image_shape: tuple[int, int],
    ) -> tuple[np.ndarray | None, str]:
        ref_candidates = (("0h", "0h"), ("img", "img_inicial"))

        for ref_time_tag, ref_label in ref_candidates:
            ref_key = (group_key, ref_time_tag)
            ref_roi = self.roi_by_item.get(ref_key)
            if ref_roi is not None and np.any(ref_roi):
                return self._resize_roi_mask(ref_roi, image_shape), f"roi_24h_limitada_pela_roi_{ref_label}"

            ref_artifacts = self._artifacts_for_item(group_key, ref_time_tag)
            if ref_artifacts is None:
                continue

            roi_mask = getattr(ref_artifacts, "roi_mask", None)
            if roi_mask is not None and np.any(roi_mask):
                return self._resize_roi_mask(roi_mask, image_shape), f"roi_24h_limitada_pelo_processamento_{ref_label}"

            mask_auto = getattr(ref_artifacts, "mask_auto", None)
            if mask_auto is not None and np.any(mask_auto):
                ref_mask = self._resize_roi_mask(mask_auto, image_shape)
                padding_px = max(2, int(round(min(image_shape) * 0.012)))
                expanded = self._expand_mask_for_roi(ref_mask, padding_px)
                return expanded, f"roi_24h_limitada_pela_mascara_{ref_label}"

        return None, "roi_padrao"

    def _read_image_unicode(self, path: Path) -> np.ndarray:
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
        except OSError as exc:
            raise ValueError(
                _app_text(self.app, "processing.read.access_failed", "Nao consegui acessar a imagem: {path}", path=path)
            ) from exc

        if data.size == 0:
            raise ValueError(
                _app_text(
                    self.app,
                    "processing.read.empty_or_corrupt",
                    "Arquivo de imagem vazio ou corrompido: {path}",
                    path=path,
                )
            )

        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raw_image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if raw_image is None:
                raise ValueError(
                    _app_text(
                        self.app,
                        "processing.read.decode_failed",
                        "Nao consegui decodificar a imagem: {path}",
                        path=path,
                    )
                )
            if raw_image.ndim == 2:
                image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
            elif raw_image.ndim == 3 and raw_image.shape[2] == 4:
                image = cv2.cvtColor(raw_image, cv2.COLOR_BGRA2BGR)
            elif raw_image.ndim == 3 and raw_image.shape[2] == 3:
                image = raw_image
            else:
                raise ValueError(
                    _app_text(
                        self.app,
                        "processing.read.unsupported_format",
                        "Formato de imagem nao suportado: {path}",
                        path=path,
                    )
                )

        return np.ascontiguousarray(image)

    @staticmethod
    def _mean_gray_intensity(image_bgr: np.ndarray) -> float:
        return float(np.mean(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)))

    @staticmethod
    def _adjust_gamma(image_bgr: np.ndarray, gamma: float) -> np.ndarray:
        gamma = max(float(gamma), 1e-6)
        table = np.array(
            [np.clip(((i / 255.0) ** gamma) * 255.0, 0, 255) for i in range(256)],
            dtype=np.uint8,
        )
        return cv2.LUT(image_bgr, table)

    @staticmethod
    def _clahe_lab(
        image_bgr: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l2 = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _homomorphic_gray_u8(gray_u8: np.ndarray, sigma: float = 18.0, ksize: int = 81) -> np.ndarray:
        src = gray_u8.astype(np.float32) / 255.0
        src = np.clip(src, 1e-6, 1.0)
        img_log = np.log1p(src)
        k = max(3, int(ksize))
        if (k % 2) == 0:
            k += 1
        illum = cv2.GaussianBlur(img_log, (k, k), sigmaX=float(sigma), sigmaY=float(sigma))
        reflect = img_log - illum
        out = np.expm1(reflect)
        out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _local_clahe_gray(gray_u8: np.ndarray, clip_limit: float = 2.4, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
        return clahe.apply(gray_u8)

    @staticmethod
    def _edge_texture_confidence_gray(gray_u8: np.ndarray, local_win: int = 13) -> tuple[np.ndarray, np.ndarray]:
        src = gray_u8.astype(np.float32) / 255.0
        gx = cv2.Scharr(src, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(src, cv2.CV_32F, 0, 1)
        grad = (1.20 * np.abs(gx)) + (0.35 * np.abs(gy))
        grad = cv2.normalize(grad, None, 0.0, 1.0, cv2.NORM_MINMAX)

        win = max(3, int(local_win))
        if (win % 2) == 0:
            win += 1
        mean = cv2.blur(src, (win, win))
        mean2 = cv2.blur(src * src, (win, win))
        texture = np.sqrt(np.maximum(mean2 - (mean * mean), 0.0))
        texture = cv2.normalize(texture, None, 0.0, 1.0, cv2.NORM_MINMAX)

        low_texture_gate = np.clip(1.0 - (0.72 * texture), 0.0, 1.0)
        edge_support = grad * (0.42 + (0.58 * low_texture_gate))
        edge_support = cv2.normalize(edge_support, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        texture_noise = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return edge_support, texture_noise

    @staticmethod
    def _bright_image_homogeneous_border_enhance(image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        homo = _ProjAnaProcessingAdapter._homomorphic_gray_u8(gray, sigma=24.0, ksize=101)
        smooth = cv2.bilateralFilter(homo, d=9, sigmaColor=26, sigmaSpace=30)
        smooth = cv2.medianBlur(smooth, 5)
        eq = _ProjAnaProcessingAdapter._local_clahe_gray(smooth, clip_limit=1.18, tile_grid_size=(8, 8))

        base = cv2.GaussianBlur(eq, (5, 5), 0)
        se_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        base = cv2.morphologyEx(base, cv2.MORPH_OPEN, se_bg)
        base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, se_bg)
        base = cv2.bilateralFilter(base, d=5, sigmaColor=14, sigmaSpace=16)
        shape_base = cv2.morphologyEx(
            base,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        )

        se_long = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        se_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        se_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        top_long = cv2.morphologyEx(shape_base, cv2.MORPH_TOPHAT, se_long)
        bot_long = cv2.morphologyEx(shape_base, cv2.MORPH_BLACKHAT, se_long)
        grad = cv2.morphologyEx(shape_base, cv2.MORPH_GRADIENT, se_edge)
        top_small = cv2.morphologyEx(shape_base, cv2.MORPH_TOPHAT, se_small)
        bot_small = cv2.morphologyEx(shape_base, cv2.MORPH_BLACKHAT, se_small)
        edge_support, texture_noise = _ProjAnaProcessingAdapter._edge_texture_confidence_gray(base, local_win=17)

        border_support = cv2.addWeighted(top_long, 0.52, bot_long, 0.42, 0.0)
        border_support = cv2.addWeighted(border_support, 1.0, grad, 0.54, 0.0)
        border_support = cv2.addWeighted(border_support, 1.0, edge_support, 0.58, 0.0)
        fine_texture = cv2.addWeighted(top_small, 0.42, bot_small, 0.54, 0.0)
        fine_texture = cv2.addWeighted(fine_texture, 1.0, texture_noise, 0.44, 0.0)

        refined = cv2.addWeighted(base, 0.98, border_support, 0.34, 0.0)
        refined = cv2.addWeighted(refined, 1.0, fine_texture, -0.40, 0.0)
        refined = cv2.bilateralFilter(refined, d=3, sigmaColor=10, sigmaSpace=12)
        return cv2.cvtColor(refined, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _mid_bright_image_enhance(image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        homo = _ProjAnaProcessingAdapter._homomorphic_gray_u8(gray, sigma=12.0, ksize=55)
        eq = _ProjAnaProcessingAdapter._local_clahe_gray(homo, clip_limit=2.2, tile_grid_size=(8, 8))
        eq = cv2.bilateralFilter(eq, d=7, sigmaColor=22, sigmaSpace=22)

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

    @staticmethod
    def _medium_0h_image_enhance(image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        homo = _ProjAnaProcessingAdapter._homomorphic_gray_u8(gray, sigma=18.0, ksize=83)
        eq = _ProjAnaProcessingAdapter._local_clahe_gray(homo, clip_limit=2.9, tile_grid_size=(8, 8))
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
        edge_support, texture_noise = _ProjAnaProcessingAdapter._edge_texture_confidence_gray(base, local_win=13)

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

    @staticmethod
    def _dark_image_edge_enhance_core(
        gray_u8: np.ndarray,
        *,
        homo_sigma: float,
        homo_ksize: int,
        clahe_clip: float,
        bg_kernel: int,
        local_win: int,
        border_gain: float,
        texture_penalty_gain: float,
        shape_open_kernel: int = 11,
        extra_texture_suppression: bool = False,
        base_blur_kernel: int = 5,
        final_blur_kernel: int = 3,
    ) -> np.ndarray:
        homo = _ProjAnaProcessingAdapter._homomorphic_gray_u8(gray_u8, sigma=homo_sigma, ksize=homo_ksize)
        eq = _ProjAnaProcessingAdapter._local_clahe_gray(homo, clip_limit=clahe_clip, tile_grid_size=(8, 8))
        eq = cv2.bilateralFilter(eq, d=11, sigmaColor=30, sigmaSpace=32)
        eq = cv2.medianBlur(eq, 5)

        base_k = max(3, int(base_blur_kernel))
        if (base_k % 2) == 0:
            base_k += 1
        base = cv2.GaussianBlur(eq, (base_k, base_k), 0)
        se_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bg_kernel, bg_kernel))
        base = cv2.morphologyEx(base, cv2.MORPH_OPEN, se_bg)
        base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, se_bg)

        shape_base = cv2.bilateralFilter(base, d=7, sigmaColor=20, sigmaSpace=22)
        shape_base = cv2.morphologyEx(
            shape_base,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shape_open_kernel, shape_open_kernel)),
        )
        if extra_texture_suppression:
            shape_base = cv2.morphologyEx(
                shape_base,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
            )
            shape_base = cv2.bilateralFilter(shape_base, d=9, sigmaColor=18, sigmaSpace=24)

        edge_support, texture_noise = _ProjAnaProcessingAdapter._edge_texture_confidence_gray(shape_base, local_win=local_win)
        grad = cv2.morphologyEx(
            shape_base,
            cv2.MORPH_GRADIENT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )

        se_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        se_mid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        se_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33))
        top_small = cv2.morphologyEx(shape_base, cv2.MORPH_TOPHAT, se_small)
        bot_small = cv2.morphologyEx(shape_base, cv2.MORPH_BLACKHAT, se_small)
        top_mid = cv2.morphologyEx(shape_base, cv2.MORPH_TOPHAT, se_mid)
        bot_mid = cv2.morphologyEx(shape_base, cv2.MORPH_BLACKHAT, se_mid)
        top_large = cv2.morphologyEx(shape_base, cv2.MORPH_TOPHAT, se_large)
        bot_large = cv2.morphologyEx(shape_base, cv2.MORPH_BLACKHAT, se_large)

        border_support = cv2.addWeighted(top_large, 0.56, top_mid, 0.42, 0.0)
        border_support = cv2.addWeighted(border_support, 1.0, bot_mid, 0.20, 0.0)
        border_support = cv2.addWeighted(border_support, 1.0, bot_large, 0.12, 0.0)
        border_support = cv2.addWeighted(border_support, 1.0, grad, 0.74, 0.0)
        border_support = cv2.addWeighted(border_support, 1.0, edge_support, 0.88, 0.0)

        texture_penalty = cv2.addWeighted(top_small, 0.34, bot_small, 0.46, 0.0)
        texture_penalty = cv2.addWeighted(texture_penalty, 1.0, texture_noise, 0.44, 0.0)
        if extra_texture_suppression:
            texture_penalty = cv2.addWeighted(texture_penalty, 1.0, top_mid, 0.22, 0.0)
            texture_penalty = cv2.addWeighted(texture_penalty, 1.0, bot_mid, 0.26, 0.0)

        refined = cv2.addWeighted(base, 1.0, border_support, border_gain, 0.0)
        refined = cv2.addWeighted(refined, 1.0, texture_penalty, -texture_penalty_gain, 0.0)
        refined = cv2.bilateralFilter(refined, d=5, sigmaColor=16, sigmaSpace=18)
        final_k = max(1, int(final_blur_kernel))
        if final_k > 1:
            if (final_k % 2) == 0:
                final_k += 1
            refined = cv2.GaussianBlur(refined, (final_k, final_k), 0)
        return cv2.cvtColor(refined, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _dark_0h_image_enhance(image_bgr: np.ndarray) -> np.ndarray:
        processed = _ProjAnaProcessingAdapter._adjust_gamma(image_bgr, gamma=0.88)
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        return _ProjAnaProcessingAdapter._dark_image_edge_enhance_core(
            gray,
            homo_sigma=18.0,
            homo_ksize=83,
            clahe_clip=3.0,
            bg_kernel=11,
            local_win=17,
            border_gain=0.78,
            texture_penalty_gain=0.46,
            shape_open_kernel=11,
            extra_texture_suppression=False,
            base_blur_kernel=5,
            final_blur_kernel=3,
        )

    @staticmethod
    def _dark_24h_image_enhance(image_bgr: np.ndarray) -> np.ndarray:
        processed = _ProjAnaProcessingAdapter._adjust_gamma(image_bgr, gamma=0.78)
        processed = _ProjAnaProcessingAdapter._clahe_lab(processed, clip_limit=2.8, tile_grid_size=(8, 8))
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        return _ProjAnaProcessingAdapter._dark_image_edge_enhance_core(
            gray,
            homo_sigma=20.0,
            homo_ksize=91,
            clahe_clip=2.8,
            bg_kernel=13,
            local_win=21,
            border_gain=0.74,
            texture_penalty_gain=0.64,
            shape_open_kernel=15,
            extra_texture_suppression=True,
            base_blur_kernel=3,
            final_blur_kernel=1,
        )

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _smooth_mask_boundary(
        mask: np.ndarray,
        roi: np.ndarray | None = None,
        *,
        close_scale: float = 1.0,
        open_scale: float = 1.0,
        blur_scale: float = 1.0,
    ) -> np.ndarray:
        source = np.asarray(mask).astype(bool)
        if not np.any(source):
            return source

        roi_bool = None if roi is None else np.asarray(roi).astype(bool)
        if roi_bool is not None and roi_bool.shape == source.shape:
            source = source & roi_bool

        h, w = source.shape
        min_dim = min(h, w)
        close_k = max(5, int(round(min_dim * 0.012 * float(close_scale))))
        open_k = max(3, int(round(min_dim * 0.006 * float(open_scale))))
        blur_k = max(5, int(round(min_dim * 0.010 * float(blur_scale))))
        if (close_k % 2) == 0:
            close_k += 1
        if (open_k % 2) == 0:
            open_k += 1
        if (blur_k % 2) == 0:
            blur_k += 1

        smooth = source.astype(np.uint8) * 255
        smooth = cv2.morphologyEx(
            smooth,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)),
        )
        smooth = cv2.morphologyEx(
            smooth,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)),
        )
        smooth = cv2.GaussianBlur(smooth, (blur_k, blur_k), 0) >= 127

        if roi_bool is not None and roi_bool.shape == source.shape:
            smooth = smooth & roi_bool
        smooth = _ProjAnaProcessingAdapter._fill_mask_holes(smooth)
        smooth = _ProjAnaProcessingAdapter._keep_largest_mask_component(smooth)
        return smooth

    def _smooth_24h_mask_boundary(self, mask: np.ndarray, roi: np.ndarray) -> np.ndarray:
        smooth = self._smooth_mask_boundary(mask, roi, close_scale=1.85, open_scale=1.35, blur_scale=1.20)
        if not np.any(smooth):
            return smooth
        smooth = cv2.morphologyEx(
            smooth.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        ) > 0
        smooth = self._fill_mask_holes(smooth & np.asarray(roi).astype(bool))
        smooth = self._keep_largest_mask_component(smooth & np.asarray(roi).astype(bool))
        return smooth

    @staticmethod
    def _red_contour_components(overlay_rgb_u8: np.ndarray) -> np.ndarray:
        c0 = overlay_rgb_u8[:, :, 0].astype(np.int16)
        c1 = overlay_rgb_u8[:, :, 1].astype(np.int16)
        c2 = overlay_rgb_u8[:, :, 2].astype(np.int16)
        red_rgb = (c0 >= 150) & (c1 <= 50) & (c2 <= 50) & (c0 - c1 >= 50) & (c0 - c2 >= 50)
        red_bgr = (c2 >= 150) & (c1 <= 50) & (c0 <= 50) & (c2 - c1 >= 50) & (c2 - c0 >= 50)
        return red_rgb | red_bgr

    def _filter_small_red_contours_overlay(
        self,
        overlay_rgb_u8: np.ndarray,
        base_rgb_u8: np.ndarray,
    ) -> tuple[np.ndarray, list[float]]:
        red_bin = self._red_contour_components(overlay_rgb_u8).astype(np.uint8)
        if not np.any(red_bin):
            return overlay_rgb_u8.copy(), []

        num_labels, labels = cv2.connectedComponents(red_bin, connectivity=8)
        filtered_overlay = overlay_rgb_u8.copy()
        contour_lengths: list[float] = []
        min_length = float(getattr(self, "red_contour_min_length", self.DEFAULT_RED_CONTOUR_MIN_LENGTH))

        for label in range(1, num_labels):
            component_mask = (labels == label).astype(np.uint8)
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue
            length = float(cv2.arcLength(contours[0], True))
            contour_lengths.append(length)
            if length < min_length:
                filtered_overlay[labels == label] = base_rgb_u8[labels == label]

        return filtered_overlay, contour_lengths

    def _create_mask_from_red_contours(self, overlay_rgb_u8: np.ndarray) -> np.ndarray:
        red_bin = self._red_contour_components(overlay_rgb_u8).astype(np.uint8)
        if not np.any(red_bin):
            return np.zeros(overlay_rgb_u8.shape[:2], dtype=bool)
        h, w = red_bin.shape
        inverted_mask = np.where(red_bin > 0, 0, 255).astype(np.uint8)
        filled_from_outside = inverted_mask.copy()
        for point in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
            cv2.floodFill(filled_from_outside, None, point, 127)
        return filled_from_outside == 255

    def _apply_filtered_red_contour_artifacts(self, results: AreaResults, artifacts: CORAArtifacts) -> None:
        overlay_rgb = getattr(artifacts, "contour_overlay_rgb_u8", None)
        base_rgb = getattr(artifacts, "base_rgb_u8", None)
        if overlay_rgb is None or base_rgb is None:
            return

        filtered_overlay, red_lengths = self._filter_small_red_contours_overlay(overlay_rgb, base_rgb)
        artifacts.filtered_contour_overlay_rgb_u8 = filtered_overlay
        artifacts.filtered_red_contour_lengths = red_lengths

        red_mask = self._create_mask_from_red_contours(filtered_overlay)
        if np.any(red_mask):
            roi_mask = getattr(artifacts, "roi_mask", None)
            red_mask = self._smooth_mask_boundary(red_mask, roi_mask)
        if np.any(red_mask):
            artifacts.filtered_contour_mask = contour_mask_with_thickness(red_mask, radius=self.config.r_auto)
            artifacts.mask_auto = red_mask
            artifacts.contour_mask = artifacts.filtered_contour_mask
            artifacts.contour_overlay_rgb_u8 = filtered_overlay
            results.area_auto = int(np.count_nonzero(red_mask))
        else:
            artifacts.filtered_contour_mask = getattr(
                artifacts,
                "contour_mask",
                np.zeros(overlay_rgb.shape[:2], dtype=bool),
            )

    def _cleanup_bright_mode_artifacts(
        self,
        results: AreaResults,
        artifacts: CORAArtifacts,
        time_tag: str | None = None,
    ) -> None:
        mask = np.asarray(artifacts.mask_auto).astype(bool)
        roi = np.asarray(artifacts.roi_mask).astype(bool)
        if mask.shape != roi.shape or not np.any(mask):
            return

        cleaned = self._fill_mask_holes(mask)
        cleaned = cv2.morphologyEx(
            cleaned.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        ) > 0
        cleaned = self._fill_mask_holes(cleaned)
        cleaned = self._keep_largest_mask_component(cleaned & roi)
        if self._is_0h_time_tag(time_tag):
            cleaned = self._stabilize_0h_bright_mask(cleaned, roi)
            cleaned = self._suppress_0h_bright_lateral_leakage(cleaned, roi)
        cleaned = self._smooth_mask_boundary(cleaned, roi)
        if not np.any(cleaned):
            return

        artifacts.mask_auto = cleaned
        artifacts.contour_mask = contour_mask_with_thickness(cleaned, radius=self.config.r_auto)
        artifacts.contour_overlay_rgb_u8 = overlay_perimeter(
            artifacts.base_rgb_u8,
            artifacts.contour_mask,
            self.config.col_auto,
        )
        results.area_auto = int(np.count_nonzero(cleaned))

    def _stabilize_0h_bright_mask(self, mask: np.ndarray, roi: np.ndarray) -> np.ndarray:
        source = np.asarray(mask).astype(bool)
        roi_bool = np.asarray(roi).astype(bool)
        if source.shape != roi_bool.shape or not np.any(source):
            return source
        h, w = source.shape
        if min(h, w) < 80:
            return source
        opened = cv2.morphologyEx(
            source.astype(np.uint8) * 255,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        ) > 0
        closed = cv2.morphologyEx(
            opened.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        ) > 0
        closed = self._fill_mask_holes(closed & roi_bool)
        closed = self._keep_largest_mask_component(closed & roi_bool)
        if not np.any(closed):
            return source
        preserved_fraction = float(np.count_nonzero(closed)) / max(float(np.count_nonzero(source)), 1.0)
        if preserved_fraction < 0.58:
            return source
        return closed

    def _suppress_0h_bright_lateral_leakage(self, mask: np.ndarray, roi: np.ndarray) -> np.ndarray:
        source = np.asarray(mask).astype(bool)
        roi_bool = np.asarray(roi).astype(bool)
        if source.shape != roi_bool.shape or not np.any(source):
            return source
        h, w = source.shape
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
        corridor[:, guard : w - guard] = True
        candidate = source & roi_bool & corridor
        candidate = cv2.morphologyEx(
            candidate.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        ) > 0
        candidate = self._fill_mask_holes(candidate)
        candidate = self._keep_largest_mask_component(candidate & roi_bool & corridor)
        candidate_pixels = int(np.count_nonzero(candidate))
        if candidate_pixels == 0:
            return source

        preserved_fraction = float(candidate_pixels) / max(float(total_pixels), 1.0)
        if preserved_fraction < 0.45:
            return source
        return candidate

    def _cleanup_24h_artifacts(self, results: AreaResults, artifacts: CORAArtifacts) -> None:
        mask = np.asarray(artifacts.mask_auto).astype(bool)
        roi = np.asarray(artifacts.roi_mask).astype(bool)
        if mask.shape != roi.shape or not np.any(mask):
            return

        cleaned = self._fill_mask_holes(mask & roi)
        cleaned = cv2.morphologyEx(
            cleaned.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        ) > 0
        cleaned = cv2.morphologyEx(
            cleaned.astype(np.uint8) * 255,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        ) > 0
        cleaned = self._refine_24h_background_separation(cleaned, roi, artifacts.base_rgb_u8)
        cleaned = cv2.morphologyEx(
            cleaned.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        ) > 0
        cleaned = self._fill_mask_holes(cleaned & roi)
        cleaned = self._keep_largest_mask_component(cleaned & roi)
        cleaned = self._smooth_24h_mask_boundary(cleaned, roi)
        if not np.any(cleaned):
            return

        artifacts.mask_auto = cleaned
        artifacts.contour_mask = contour_mask_with_thickness(cleaned, radius=self.config.r_auto)
        artifacts.contour_overlay_rgb_u8 = overlay_perimeter(
            artifacts.base_rgb_u8,
            artifacts.contour_mask,
            self.config.col_auto,
        )
        results.area_auto = int(np.count_nonzero(cleaned))

    def _refine_24h_background_separation(self, mask: np.ndarray, roi: np.ndarray, base_rgb_u8: np.ndarray) -> np.ndarray:
        source = np.asarray(mask).astype(bool)
        roi_bool = np.asarray(roi).astype(bool)
        if source.shape != roi_bool.shape or not np.any(source):
            return source

        gray = cv2.cvtColor(base_rgb_u8, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        local_mean = cv2.blur(gray.astype(np.float32), (21, 21))
        local_mean2 = cv2.blur((gray.astype(np.float32) ** 2), (21, 21))
        local_std = np.sqrt(np.maximum(local_mean2 - (local_mean ** 2), 0.0))
        local_std = cv2.normalize(local_std, None, 0.0, 1.0, cv2.NORM_MINMAX)

        bg_model = cv2.morphologyEx(
            gray,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
        )
        bg_diff = cv2.absdiff(gray, bg_model).astype(np.float32)
        bg_diff = cv2.normalize(bg_diff, None, 0.0, 1.0, cv2.NORM_MINMAX)

        ys, xs = np.nonzero(source)
        roi_ys, roi_xs = np.nonzero(roi_bool)
        if roi_xs.size < 64 or roi_ys.size < 64:
            return source
        if xs.size >= 32 and ys.size >= 32:
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
        else:
            cx = float(np.mean(roi_xs))
            cy = float(np.mean(roi_ys))

        h, w = source.shape
        xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        dist_x = np.abs(xx - cx)
        dist_y = np.abs(yy - cy)
        dist_x = dist_x / max(float(np.max(dist_x[roi_bool])), 1.0)
        dist_y = dist_y / max(float(np.max(dist_y[roi_bool])), 1.0)

        interior_score = (
            (0.48 * (1.0 - local_std))
            + (0.20 * (1.0 - bg_diff))
            + (0.22 * (1.0 - dist_x))
            + (0.10 * (1.0 - dist_y))
        )
        interior_score = np.clip(interior_score, 0.0, 1.0)

        roi_scores = interior_score[roi_bool]
        thr = float(np.quantile(roi_scores, 0.72))
        candidate = roi_bool & (interior_score >= thr)

        corridor_half = max(18, int(round(0.12 * w)))
        corridor = np.abs(xx - cx) <= float(corridor_half)
        corridor = corridor.astype(bool) & roi_bool
        source_seed = source & roi_bool & corridor
        if np.count_nonzero(source_seed) < 32:
            source_seed = candidate & corridor
        if np.count_nonzero(source_seed) < 32:
            source_seed = roi_bool & corridor

        candidate = cv2.morphologyEx(
            candidate.astype(np.uint8) * 255,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        ) > 0
        candidate = cv2.morphologyEx(
            candidate.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        ) > 0
        candidate = self._fill_mask_holes(candidate & roi_bool)
        if not np.any(candidate):
            return source

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate.astype(np.uint8),
            connectivity=8,
            ltype=cv2.CV_32S,
        )
        if n_labels <= 1:
            return source

        seed_mask = source_seed.astype(bool)
        best_mask = None
        best_score = float("-inf")
        seed_area = float(np.count_nonzero(seed_mask))
        for label_idx in range(1, n_labels):
            comp = labels == label_idx
            comp_area = float(stats[label_idx, cv2.CC_STAT_AREA])
            if comp_area <= 0:
                continue
            overlap = float(np.count_nonzero(comp & seed_mask)) / max(seed_area, 1.0)
            comp_scores = interior_score[comp]
            if comp_scores.size == 0:
                continue
            comp_mean = float(np.mean(comp_scores))
            x, _y, comp_w, _comp_h, _ = stats[label_idx]
            comp_cx = x + (comp_w / 2.0)
            center_penalty = abs(comp_cx - cx) / max(float(w), 1.0)
            score = (0.60 * overlap) + (0.32 * comp_mean) - (0.18 * center_penalty)
            if score > best_score:
                best_score = score
                best_mask = comp

        if best_mask is None or not np.any(best_mask):
            return source
        best_mask = cv2.morphologyEx(
            best_mask.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
        ) > 0
        best_mask = self._fill_mask_holes(best_mask & roi_bool)
        best_mask = self._keep_largest_mask_component(best_mask & roi_bool)
        if not np.any(best_mask):
            return source

        preserved_fraction = float(np.count_nonzero(best_mask)) / max(float(np.count_nonzero(source)), 1.0)
        if preserved_fraction < 0.30 and np.count_nonzero(source) > 0:
            return source
        return best_mask

    def _adaptive_preprocess_by_mean_intensity(
        self,
        image_bgr: np.ndarray,
        time_tag: str | None = None,
    ) -> tuple[np.ndarray, dict[str, float | str]]:
        mean_gray = self._mean_gray_intensity(image_bgr)

        if time_tag is not None and self._is_24h_time_tag(time_tag):
            mode = "24h_escura"
            processed = self._dark_24h_image_enhance(image_bgr)
            return processed, {"mode": mode, "mean_gray": mean_gray}

        if self._uses_0h_brightness_buckets(time_tag):
            if mean_gray >= BRIGHT_0H_IMAGE_THRESHOLD:
                mode = "clara"
                processed = self._bright_image_homogeneous_border_enhance(image_bgr)
            elif mean_gray <= DARK_IMAGE_THRESHOLD:
                mode = "escura"
                processed = self._dark_0h_image_enhance(image_bgr)
            else:
                mode = "media"
                processed = self._medium_0h_image_enhance(image_bgr)
            return processed, {"mode": mode, "mean_gray": mean_gray}

        if mean_gray >= BRIGHT_IMAGE_THRESHOLD:
            mode = "clara"
            processed = self._bright_image_homogeneous_border_enhance(image_bgr)
        elif mean_gray <= DARK_IMAGE_THRESHOLD:
            mode = "escura"
            processed = image_bgr.copy()
        elif mean_gray >= MID_BRIGHT_IMAGE_THRESHOLD:
            mode = "intermediaria_clara"
            processed = self._adjust_gamma(image_bgr, gamma=1.25)
            processed = self._clahe_lab(processed, clip_limit=1.9, tile_grid_size=(8, 8))
            processed = self._mid_bright_image_enhance(processed)
        else:
            mode = "normal"
            processed = self._clahe_lab(image_bgr, clip_limit=1.6, tile_grid_size=(8, 8))

        return processed, {"mode": mode, "mean_gray": mean_gray}

    @staticmethod
    def _safe_filename(name: str, max_len: int = 120) -> str:
        clean = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in name).strip("._")
        if len(clean) <= max_len:
            return clean or "resultado"
        return f"{clean[:max_len-11]}_{hashlib.sha1(clean.encode('utf-8')).hexdigest()[:10]}"

    @staticmethod
    def _write_image_file(path: Path, image: np.ndarray) -> bool:
        suffix = path.suffix.lower() or ".png"
        ok, encoded = cv2.imencode(suffix, image)
        if not ok:
            return False
        try:
            encoded.tofile(str(path))
        except Exception:
            return False
        return True

    def _prepare_adaptive_input(
        self,
        path: Path,
        time_tag: str | None = None,
        image_bgr: np.ndarray | None = None,
    ) -> tuple[Path, dict[str, float | str], np.ndarray]:
        if image_bgr is None:
            image_bgr = self._read_image_unicode(path)
        processed_bgr, info = self._adaptive_preprocess_by_mean_intensity(image_bgr, time_tag=time_tag)

        temp_name = f"{self._safe_filename(path.stem)}_adapt.png"
        temp_path = self.temp_dir / temp_name
        if not self._write_image_file(temp_path, processed_bgr):
            raise RuntimeError(
                _app_text(
                    self.app,
                    "processing.temp_save_failed",
                    "Nao foi possivel salvar imagem temporaria adaptada: {path}",
                    path=temp_path,
                )
            )
        return temp_path, info, processed_bgr

    def _run_cora_with_external_preprocess(
        self,
        adaptive_path: Path,
        time_tag: str,
        roi_mask: np.ndarray | None,
        stage_callback,
    ) -> tuple[AreaResults, CORAArtifacts]:
        area_manual_ref = int(getattr(self.app, "_area_reference_px", 1))
        try:
            cfg = replace(
                self.config,
                adaptive_preprocess_enabled=False,
                adaptive_cleanup_enabled=False,
            )
        except Exception:
            cfg = self.config

        return run_cora(
            base_path=str(adaptive_path),
            area_manual=area_manual_ref,
            config=cfg,
            roi_mask=roi_mask,
            show=False,
            verbose=False,
            return_artifacts=True,
            progress_callback=stage_callback,
            time_tag=time_tag,
        )

    def process_item(self, group_key: str, time_tag: str, path: Path, stage_callback):
        pipeline_started_at = time.perf_counter()
        stage_callback("Analisando brilho medio", 0.05)
        loading_started_at = time.perf_counter()
        image_bgr = self._read_image_unicode(path)
        loading_time_s = time.perf_counter() - loading_started_at
        segmentation_started_at = time.perf_counter()
        image_shape = image_bgr.shape[:2]
        roi_mask = self._roi_for_item(group_key, time_tag)
        roi_origin = "roi_padrao"
        if roi_mask is not None:
            roi_origin = "roi_personalizada_ou_atual"
        if roi_mask is None and self._is_24h_time_tag(time_tag):
            roi_mask, roi_origin = self._derive_24h_roi_from_0h(group_key, image_shape)
            stage_callback(f"Preparando ROI especial de 24h ({roi_origin})", 0.09)

        adaptive_path, adaptive_info, processed_bgr = self._prepare_adaptive_input(
            path,
            time_tag=time_tag,
            image_bgr=image_bgr,
        )
        adaptive_info["roi_source"] = roi_origin
        stage_callback(
            f"Modo adaptativo: {adaptive_info['mode']} (media={adaptive_info['mean_gray']:.1f})",
            0.12,
        )

        results, artifacts = self._run_cora_with_external_preprocess(
            adaptive_path=adaptive_path,
            time_tag=time_tag,
            roi_mask=roi_mask,
            stage_callback=stage_callback,
        )

        if self._is_24h_time_tag(time_tag):
            self._cleanup_24h_artifacts(results, artifacts)
        elif adaptive_info.get("mode") == "clara":
            self._cleanup_bright_mode_artifacts(results, artifacts, time_tag=time_tag)

        artifacts.base_rgb_u8 = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
        artifacts.contour_overlay_rgb_u8 = overlay_perimeter(
            artifacts.base_rgb_u8,
            artifacts.contour_mask,
            self.config.col_auto,
        )
        self._apply_filtered_red_contour_artifacts(results, artifacts)
        self._worker_artifacts.setdefault(group_key, {})[time_tag] = artifacts
        adaptive_info["loading_time_s"] = float(loading_time_s)
        adaptive_info["segmentation_time_s"] = float(time.perf_counter() - segmentation_started_at)
        adaptive_info["total_pipeline_time_s"] = float(time.perf_counter() - pipeline_started_at)
        return results, artifacts, adaptive_info


def process_single_item(app, group_key: str, time_tag: str, path: Path) -> tuple[AreaResults, CORAArtifacts]:
    """Processa um unico item com o mesmo pipeline do lote."""
    work_dir = _create_processing_work_dir(app)
    try:
        processor = _ProjAnaProcessingAdapter(app=app, temp_dir=work_dir)
        results, artifacts, _ = processor.process_item(
            group_key=group_key,
            time_tag=time_tag,
            path=path,
            stage_callback=lambda _stage, _progress: None,
        )
        return results, artifacts
    finally:
        _cleanup_processing_work_dir(work_dir)


def run_processing_worker(app, mode: str, items: list[tuple[str, str, Path]], save_after: bool) -> None:
    """Processa itens em lote em thread separada, com pipeline adaptativo do proj_ana."""
    success: list[tuple[str, str]] = []
    failures: list[tuple[str, str, str]] = []
    total = len(items)
    cancelled = False
    batch_context = getattr(app, "_batch_test_context", {}) if mode == "batch_test" else {}
    batch_size = max(1, int((batch_context or {}).get("batch_size", total or 1)))
    batch_monitor: ProcessResourceMonitor | None = None
    batch_started_at: float | None = None
    batch_metrics: dict[str, object] | None = None

    work_dir = _create_processing_work_dir(app)
    try:
        processor = _ProjAnaProcessingAdapter(app=app, temp_dir=work_dir)
        try:
            for idx, (group_key, time_tag, path) in enumerate(items, start=1):
                if app.cancel_event.is_set():
                    cancelled = True
                    break

                batch_position = ((idx - 1) % batch_size) + 1
                if mode == "batch_test" and batch_position == 1:
                    # PT: Cada iteracao parte de um adaptador limpo e usa exatamente o mesmo conjunto de imagens da iteracao anterior.
                    # EN: Each iteration starts with a clean adapter and uses exactly the same image set as the previous iteration.
                    processor = _ProjAnaProcessingAdapter(app=app, temp_dir=work_dir)
                    batch_started_at = time.perf_counter()
                    batch_metrics = {
                        "tamanho_lote": batch_size,
                        "iteracao": ((idx - 1) // batch_size) + 1,
                        "imagens_planejadas": batch_size,
                        "imagens_processadas": 0,
                        "sucessos": 0,
                        "falhas": 0,
                        "tempo_carregamento_s": 0.0,
                        "tempo_segmentacao_s": 0.0,
                        "cancelado": "nao",
                        "erro": "",
                    }
                    batch_monitor = ProcessResourceMonitor()
                    batch_monitor.start()

                app.worker_queue.put(
                    ("item_stage", mode, idx, total, group_key, time_tag, path.name, "Preparando imagem", 0.0)
                )

                def on_stage(stage: str, stage_progress: float) -> None:
                    app.worker_queue.put(
                        ("item_stage", mode, idx, total, group_key, time_tag, path.name, stage, stage_progress)
                    )

                run_started_at = time.perf_counter()
                resource_monitor = ProcessResourceMonitor() if mode == "single_test" else None
                if resource_monitor is not None:
                    resource_monitor.start()

                try:
                    # PT: Cada repeticao do teste deve partir do mesmo estado inicial; um adaptador novo evita reutilizar a ROI da repeticao anterior.
                    # EN: Each test repetition must start from the same initial state; a new adapter prevents reuse of the previous repetition's ROI.
                    item_processor = (
                        _ProjAnaProcessingAdapter(app=app, temp_dir=work_dir)
                        if mode == "single_test"
                        else processor
                    )
                    if mode == "single_test":
                        item_processor.processed_by_group = {}
                    results, artifacts, adaptive_info = item_processor.process_item(
                        group_key=group_key,
                        time_tag=time_tag,
                        path=path,
                        stage_callback=on_stage,
                    )
                    if resource_monitor is not None:
                        peaks = resource_monitor.stop()
                        adaptive_info["peak_ram_mb"] = peaks.ram_mb
                        adaptive_info["peak_cpu_percent"] = peaks.cpu_percent
                        resource_monitor = None
                    if batch_metrics is not None:
                        batch_metrics["imagens_processadas"] = int(batch_metrics["imagens_processadas"]) + 1
                        batch_metrics["sucessos"] = int(batch_metrics["sucessos"]) + 1
                        batch_metrics["tempo_carregamento_s"] = float(
                            batch_metrics["tempo_carregamento_s"]
                        ) + float(adaptive_info.get("loading_time_s", 0.0) or 0.0)
                        batch_metrics["tempo_segmentacao_s"] = float(
                            batch_metrics["tempo_segmentacao_s"]
                        ) + float(adaptive_info.get("segmentation_time_s", 0.0) or 0.0)
                    success.append((group_key, time_tag))
                    app.worker_queue.put(
                        ("item_ok", group_key, time_tag, path, results, artifacts, adaptive_info, idx)
                    )
                except Exception as exc:
                    error_metrics: dict[str, object] = {
                        "total_pipeline_time_s": float(time.perf_counter() - run_started_at),
                    }
                    if resource_monitor is not None:
                        peaks = resource_monitor.stop()
                        error_metrics["peak_ram_mb"] = peaks.ram_mb
                        error_metrics["peak_cpu_percent"] = peaks.cpu_percent
                    err_text = str(exc)
                    image_read_error = any(
                        token in err_text
                        for token in (
                            "Nao consegui ler:",
                            "Nao consegui acessar",
                            "Nao consegui decodificar",
                            "Arquivo de imagem",
                            "Formato de imagem",
                            "Could not access",
                            "Could not decode",
                            "Empty or corrupted image file",
                            "Unsupported image format",
                        )
                    )
                    if image_read_error:
                        err_text = _app_text(
                            app,
                            "processing.error_read_image",
                            "Falha ao ler imagem: {name}\n{error}",
                            name=path.name,
                            error=err_text,
                        )
                    else:
                        err_text = _app_text(
                            app,
                            "processing.error_processing_image",
                            "Falha no processamento: {name}\n{error}",
                            name=path.name,
                            error=err_text,
                        )
                    failures.append((group_key, time_tag, err_text))
                    if batch_metrics is not None:
                        batch_metrics["imagens_processadas"] = int(batch_metrics["imagens_processadas"]) + 1
                        batch_metrics["falhas"] = int(batch_metrics["falhas"]) + 1
                        previous_error = str(batch_metrics.get("erro", "")).strip()
                        batch_metrics["erro"] = (
                            f"{previous_error} | {path.name}: {err_text}" if previous_error else f"{path.name}: {err_text}"
                        )
                    app.worker_queue.put(
                        ("item_error", group_key, time_tag, err_text, error_metrics, idx)
                    )

                app.worker_queue.put(
                    ("item_stage", mode, idx, total, group_key, time_tag, path.name, "Finalizado", 1.0)
                )
                app.worker_queue.put(("progress", mode, idx, total, group_key, time_tag, path.name))

                if mode == "batch_test" and batch_position == batch_size and batch_metrics is not None:
                    elapsed = max(0.0, time.perf_counter() - float(batch_started_at or time.perf_counter()))
                    peaks = batch_monitor.stop() if batch_monitor is not None else None
                    processed = int(batch_metrics["imagens_processadas"])
                    batch_metrics["tempo_total_s"] = elapsed
                    batch_metrics["ram_maxima_mb"] = peaks.ram_mb if peaks is not None else ""
                    batch_metrics["cpu_maxima_percent"] = peaks.cpu_percent if peaks is not None else ""
                    batch_metrics["imagens_por_segundo"] = (float(processed) / elapsed) if elapsed > 0.0 else ""
                    app.worker_queue.put(("batch_iteration_done", dict(batch_metrics)))
                    batch_monitor = None
                    batch_started_at = None
                    batch_metrics = None

                if app.cancel_event.is_set():
                    cancelled = True
                    break
        except Exception as exc:
            failures.append(
                (
                    "__worker__",
                    _app_text(app, "processing.general", "geral"),
                    _app_text(app, "processing.error_unexpected", "Falha inesperada no processamento: {error}", error=exc),
                )
            )
        finally:
            if batch_monitor is not None:
                batch_monitor.stop()
            app.worker_queue.put(("done", mode, success, failures, save_after, cancelled))
    finally:
        _cleanup_processing_work_dir(work_dir)


def _processing_temp_root(app) -> Path:
    """Escolhe uma pasta temporaria gravavel para arquivos intermediarios."""
    candidates: list[Path] = []

    output_folder_var = getattr(app, "output_folder_var", None)
    if output_folder_var is not None:
        try:
            output_folder_value = (
                str(output_folder_var.get()).strip()
                if hasattr(output_folder_var, "get")
                else str(output_folder_var).strip()
            )
        except Exception:
            output_folder_value = ""
        if output_folder_value:
            candidates.append(Path(output_folder_value).expanduser())

    folder_var = getattr(app, "folder_var", None)
    if folder_var is not None:
        try:
            folder_value = str(folder_var.get()).strip() if hasattr(folder_var, "get") else str(folder_var).strip()
        except Exception:
            folder_value = ""
        if folder_value:
            candidates.append(Path(folder_value))

    added_group_parent = False
    for _group_key, time_map in getattr(app, "group_files", {}).items():
        if not isinstance(time_map, dict):
            continue
        for _time_tag, src_path in time_map.items():
            try:
                candidates.append(Path(src_path).resolve().parent)
                added_group_parent = True
                break
            except Exception:
                continue
        if added_group_parent:
            break

    candidates.append(Path.cwd())

    for base in candidates:
        try:
            base.mkdir(parents=True, exist_ok=True)
        except Exception:
            continue
        probe = base / ".cora_tmp_probe"
        try:
            probe.mkdir(exist_ok=False)
            probe.rmdir()
            return base
        except Exception:
            continue

    return Path.cwd()


def _create_processing_work_dir(app) -> Path:
    root = _processing_temp_root(app) / ".cora_tmp"
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"wh_adapt_{uuid.uuid4().hex[:10]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _cleanup_processing_work_dir(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
        parent = path.parent
        if parent.name == ".cora_tmp":
            try:
                parent.rmdir()
            except OSError:
                pass
    except Exception:
        pass
