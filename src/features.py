"""Descritores hand-crafted (cinco familias). Cada funcao recebe a imagem uint8 512x512
ja pre-processada e a mascara booleana da ROI, e devolve um dicionario {nome: valor}.
O prefixo do nome (antes do primeiro "_") identifica a familia.
"""
import cv2
import numpy as np
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern

from preprocessing import lung_zones, mask_bbox

EPS = 1e-8
ANGLES = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)


def _entropy(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _resize_pair(img, mask, size):
    im = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)
    return im, m


def intensity_features(img, mask):
    """Estatisticas globais da ROI + estatisticas por zona pulmonar (6 zonas).
    As zonas codificam a distribuicao espacial tipica da COVID-19 (bilateral, periferica,
    predominante em campos inferiores)."""
    v = img[mask].astype(np.float32)
    f = {
        "int_mean": v.mean(), "int_std": v.std(),
        "int_skew": skew(v), "int_kurt": kurtosis(v),
    }
    hist, _ = np.histogram(v, bins=16, range=(0, 256))
    hist = hist / (hist.sum() + EPS)
    f["int_entropy"] = _entropy(hist)
    for q in (5, 25, 50, 75, 95):
        f[f"int_p{q}"] = np.percentile(v, q)
    for i, h in enumerate(hist):
        f[f"int_hist{i}"] = h

    means = {}
    for zn, z in lung_zones(mask).items():
        zv = img[z].astype(np.float32)
        if zv.size < 50:
            f[f"int_{zn}_mean"] = f[f"int_{zn}_std"] = f[f"int_{zn}_p90"] = np.nan
            means[zn] = np.nan
            continue
        f[f"int_{zn}_mean"] = zv.mean()
        f[f"int_{zn}_std"] = zv.std()
        f[f"int_{zn}_p90"] = np.percentile(zv, 90)
        means[zn] = zv.mean()
    right = np.nanmean([means[k] for k in ("Rsup", "Rmed", "Rinf")])
    left = np.nanmean([means[k] for k in ("Lsup", "Lmed", "Linf")])
    inf = np.nanmean([means["Rinf"], means["Linf"]])
    sup = np.nanmean([means["Rsup"], means["Lsup"]])
    f["int_asym_lr"] = abs(right - left)
    f["int_inf_minus_sup"] = inf - sup
    return f


def glcm_features(img, mask, levels=32, distances=(1, 2, 4)):
    """Haralick sobre GLCM restrita a ROI: pixels fora da mascara recebem nivel 0 e a
    linha/coluna 0 e descartada antes de normalizar. Media e amplitude entre 4 angulos."""
    q = (img.astype(np.uint16) * levels // 256).astype(np.uint8) + 1
    q[~mask] = 0
    P = graycomatrix(q, distances=distances, angles=ANGLES, levels=levels + 1, symmetric=True, normed=False)
    P = P[1:, 1:].astype(np.float64)
    P /= P.sum(axis=(0, 1), keepdims=True) + EPS
    f = {}
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"):
        vals = graycoprops(P, prop)
        for i, d in enumerate(distances):
            f[f"glcm_{prop}_d{d}_mean"] = vals[i].mean()
            f[f"glcm_{prop}_d{d}_range"] = np.ptp(vals[i])
    ent = -(P * np.log2(P + EPS)).sum(axis=(0, 1))
    for i, d in enumerate(distances):
        f[f"glcm_entropy_d{d}_mean"] = ent[i].mean()
        f[f"glcm_entropy_d{d}_range"] = np.ptp(ent[i])
    return f


def lbp_features(img, mask):
    """LBP uniforme invariante a rotacao (Ojala et al., 2002), multiescala."""
    small, msmall = _resize_pair(img, mask, 256)
    f = {}
    for im, m, P, R, tag in ((img, mask, 8, 1, "s512p8r1"), (img, mask, 16, 2, "s512p16r2"),
                             (small, msmall, 8, 1, "s256p8r1")):
        codes = local_binary_pattern(im, P, R, method="uniform")
        h, _ = np.histogram(codes[m], bins=P + 2, range=(0, P + 2))
        h = h / (h.sum() + EPS)
        for i, v in enumerate(h):
            f[f"lbp_{tag}_b{i}"] = v
    return f


def gabor_features(img, mask, wavelengths=(4, 8, 16, 32)):
    """Banco de Gabor 4 escalas x 4 orientacoes (magnitude do par em quadratura) em 256x256."""
    im, m = _resize_pair(img, mask, 256)
    im = im.astype(np.float32) / 255.0
    f = {}
    for lambd in wavelengths:
        sigma = 0.56 * lambd
        ks = int(2 * np.ceil(3 * sigma) + 1)
        for t, theta in enumerate(ANGLES):
            kr = cv2.getGaborKernel((ks, ks), sigma, theta, lambd, 0.5, 0, ktype=cv2.CV_32F)
            ki = cv2.getGaborKernel((ks, ks), sigma, theta, lambd, 0.5, np.pi / 2, ktype=cv2.CV_32F)
            kr -= kr.mean()
            r = cv2.filter2D(im, cv2.CV_32F, kr)
            i = cv2.filter2D(im, cv2.CV_32F, ki)
            mag = np.sqrt(r * r + i * i)[m]
            f[f"gabor_l{lambd}_t{t}_mean"] = mag.mean()
            f[f"gabor_l{lambd}_t{t}_std"] = mag.std()
    return f


def hog_features(img, mask):
    """HOG (Dalal & Triggs, 2005) no recorte do bounding box pulmonar redimensionado a 256x256."""
    r0, r1, c0, c1 = mask_bbox(mask)
    crop = cv2.resize(img[r0:r1, c0:c1], (256, 256), interpolation=cv2.INTER_AREA)
    v = hog(crop, orientations=9, pixels_per_cell=(32, 32), cells_per_block=(2, 2),
            block_norm="L2-Hys", feature_vector=True)
    return {f"hog_{i}": x for i, x in enumerate(v)}


FEATURE_FUNCS = {
    "int": intensity_features,
    "glcm": glcm_features,
    "lbp": lbp_features,
    "gabor": gabor_features,
    "hog": hog_features,
}
