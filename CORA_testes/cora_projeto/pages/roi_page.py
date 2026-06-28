"""Funcoes de apoio para gerar a imagem realcada usada no editor de ROI."""

import cv2
import numpy as np

try:
    if __package__:
        from ..matlab_style_cora import (
            clahe_matlab_like,
            ensure_gray,
            homomorphic_filter,
            mat2gray,
            morph_matlab_like,
            to_uint8_01,
        )
    else:
        from cora_projeto.matlab_style_cora import (
            clahe_matlab_like,
            ensure_gray,
            homomorphic_filter,
            mat2gray,
            morph_matlab_like,
            to_uint8_01,
        )
except ImportError:
    from matlab_style_cora import (
        clahe_matlab_like,
        ensure_gray,
        homomorphic_filter,
        mat2gray,
        morph_matlab_like,
        to_uint8_01,
    )


def build_roi_editor_enhanced_image(app, image_rgb_u8: np.ndarray) -> np.ndarray:
    """Prepara uma versao realcada da imagem para facilitar edicao manual da mascara."""
    base = np.asarray(image_rgb_u8, dtype=np.uint8)
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    elif base.ndim == 3 and base.shape[2] >= 3:
        base = base[:, :, :3]
    else:
        raise ValueError("Imagem invalida para editor de ROI.")

    # PT: Pipeline de realce voltado para aumentar contraste de bordas e textura da lesao. | EN: Enhancement pipeline designed to increase lesion edge and texture contrast.
    try:
        gray = ensure_gray(base)
        gray01 = mat2gray(gray)
        img_homo = homomorphic_filter(
            gray01,
            sigma=app.config.homomorphic_sigma,
            ksize=app.config.homomorphic_ksize,
        )
        img_eq = clahe_matlab_like(
            img_homo,
            num_tiles=app.config.clahe_num_tiles,
            clip_limit=app.config.clahe_clip_limit,
        )
        img_morph = morph_matlab_like(
            img_eq,
            radius=app.config.morph_radius,
            a_top=app.config.morph_a_top,
            a_both=app.config.morph_a_both,
            return_hat_energy=False,
        )
        enhanced_u8 = to_uint8_01(np.asarray(img_morph))
        return cv2.cvtColor(enhanced_u8, cv2.COLOR_GRAY2RGB)
    # PT: Em caso de falha no realce, retorna a imagem original para nao interromper o fluxo. | EN: If enhancement fails, returns the original image to avoid interrupting the workflow.
    except Exception:
        return base.copy()



