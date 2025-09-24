#!/usr/bin/env python3
# -- coding: utf-8 --
import cv2
import numpy as np

# -------------------------
# Alinhamento via ArUco
# -------------------------
def align_aruco(img_bgr, dict_name="DICT_4X4_50", ref_w=1000, ref_h=1400):
    """
    Alinha a folha usando marcadores ArUco (4 cantos).
    """
    # Define dicionário de ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    aruco_params = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, _ = detector.detectMarkers(img_bgr)

    if ids is None or len(ids) < 4:
        raise RuntimeError("Menos de 4 marcadores ArUco detectados")

    # Ordena os cantos
    ids = ids.flatten()
    ref_pts = []
    for marker_id in [0, 1, 2, 3]:
        idx = np.where(ids == marker_id)[0][0]
        ref_pts.append(corners[idx][0][0])  # canto superior esquerdo de cada marcador

    ref_pts = np.array(ref_pts, dtype="float32")

    dst_pts = np.array([
        [0, 0],
        [ref_w - 1, 0],
        [ref_w - 1, ref_h - 1],
        [0, ref_h - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(ref_pts, dst_pts)
    warped = cv2.warpPerspective(img_bgr, M, (ref_w, ref_h))
    return warped, M


# -------------------------
# Alinhamento via contornos
# -------------------------
def align_auto(img_bgr, ref_w=1000, ref_h=1400):
    """
    Alinha automaticamente usando detecção de contornos.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 75, 200)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            rect = _order_points(pts)
            dst = np.array([
                [0, 0],
                [ref_w - 1, 0],
                [ref_w - 1, ref_h - 1],
                [0, ref_h - 1]], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img_bgr, M, (ref_w, ref_h))
            return warped, M

    raise RuntimeError("Não foi possível detectar contornos para alinhamento")


def _order_points(pts):
    """
    Ordena pontos no formato: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


# -------------------------
# Fallback: deskew
# -------------------------
def deskew(img_bgr):
    """
    Corrige inclinação básica da imagem (deskew).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(img_bgr, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated, M


# -------------------------
# Wrapper principal
# -------------------------
def align_image(img_bgr, metodo="auto_fallback", debug_dir=None):
    """
    Wrapper de alinhamento: tenta ArUco -> Auto -> Deskew -> Original.
    Sempre retorna (warped, metodo_usado, ok).
    """
    try:
        if metodo == "aruco":
            warped, M = align_aruco(img_bgr)
            return warped, "aruco", True

        elif metodo == "auto":
            warped, M = align_auto(img_bgr)
            return warped, "auto", True

        elif metodo == "auto_fallback":
            # 1. Tenta ArUco
            try:
                warped, M = align_aruco(img_bgr)
                return warped, "aruco", True
            except Exception as e:
                print(f"[WARN] ArUco falhou: {e}")

            # 2. Tenta Auto
            try:
                warped, M = align_auto(img_bgr)
                return warped, "auto", True
            except Exception as e:
                print(f"[WARN] Auto falhou: {e}")

            # 3. Tenta Deskew
            try:
                warped, M = deskew(img_bgr)
                return warped, "deskew", True
            except Exception as e:
                print(f"[WARN] Deskew falhou: {e}")

            # 4. Se nada funcionar, retorna imagem original
            return img_bgr, "original", False

    except Exception as e:
        print(f"[ERRO align_image] {e}")
        return img_bgr, "error", False