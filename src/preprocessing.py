"""Pre-processamento classico: CLAHE, mascara pulmonar heuristica e zonas anatomicas.

A mascara pulmonar NAO usa redes neurais: limiar de Otsu calculado na regiao central,
morfologia matematica, selecao dos dois maiores componentes internos e fecho convexo
por componente (para recuperar regioes pulmonares opacificadas, mais claras que o ar).
Quando a heuristica falha (area implausivel), usa-se uma ROI retangular fixa e o caso
e marcado (qc_mask_fallback) para analise de erro.
"""
import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image


def apply_clahe(img, clip=2.0, tiles=8):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(img)


def default_roi(shape):
    h, w = shape
    m = np.zeros(shape, dtype=bool)
    m[int(0.12 * h):int(0.90 * h), int(0.08 * w):int(0.92 * w)] = True
    return m


def _ellipse(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _inner_components(dark):
    h, w = dark.shape
    lab = label(dark, connectivity=1)
    cands = []
    for p in regionprops(lab):
        r0, c0, r1, c1 = p.bbox
        if r0 == 0 or c0 == 0 or r1 == h or c1 == w:  # ar externo ao corpo toca a borda
            continue
        if p.area < 0.02 * h * w:
            continue
        cy, _ = p.centroid
        if not (0.15 * h < cy < 0.80 * h):
            continue
        cands.append(p)
    return sorted(cands, key=lambda p: p.area, reverse=True)[:2], lab


def lung_mask(img):
    """Retorna (mascara booleana, usou_fallback).

    Busca em limiares progressivamente mais restritivos e aberturas maiores ate isolar os
    pulmoes do ar externo (a traqueia e a queda de intensidade na borda do torax podem
    conecta-los a borda da imagem)."""
    h, w = img.shape
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    t0 = threshold_otsu(blur[h // 6:-h // 6, w // 6:-w // 6])
    single = None
    for factor in (1.0, 0.9, 0.8, 0.7):
        base = (blur < t0 * factor).astype(np.uint8)
        for k in (9, 21, 35):
            cands, lab = _inner_components(cv2.morphologyEx(base, cv2.MORPH_OPEN, _ellipse(k)))
            if not cands:
                continue
            mask = np.zeros((h, w), dtype=bool)
            for p in cands:
                comp = cv2.dilate((lab == p.label).astype(np.uint8), _ellipse(k)).astype(bool) & base.astype(bool)
                mask |= convex_hull_image(comp)
            frac = mask.mean()
            if not (0.08 <= frac <= 0.60):
                continue
            if len(cands) == 2 or frac >= 0.15:
                return mask, False
            if single is None:
                single = mask
    if single is not None:
        return single, False
    return default_roi((h, w)), True


def mask_bbox(mask, margin=0.05):
    h, w = mask.shape
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    dr, dc = int(margin * h), int(margin * w)
    return (max(rows[0] - dr, 0), min(rows[-1] + 1 + dr, h),
            max(cols[0] - dc, 0), min(cols[-1] + 1 + dc, w))


def lung_zones(mask):
    """Seis zonas: {R,L} x {sup,med,inf}. R = lado esquerdo da imagem (direita do paciente em PA/AP)."""
    r0, r1, c0, c1 = mask_bbox(mask, margin=0.0)
    mid = (c0 + c1) // 2
    edges = np.linspace(r0, r1, 4).astype(int)
    zones = {}
    for side, (a, b) in {"R": (0, mid), "L": (mid, mask.shape[1])}.items():
        for zi, zn in enumerate(["sup", "med", "inf"]):
            z = np.zeros_like(mask)
            z[edges[zi]:edges[zi + 1], a:b] = True
            zones[f"{side}{zn}"] = z & mask
    return zones
