#!/usr/bin/env python3
# -- coding: utf-8 --
import cv2
import numpy as np

# -----------------------------
# Funções auxiliares
# -----------------------------
def _fill_ratio_masked(patch_bin):
    """
    Calcula a proporção de preenchimento considerando
    apenas uma máscara circular no centro da bolha.
    """
    h, w = patch_bin.shape
    mask = np.zeros_like(patch_bin, dtype=np.uint8)
    radius = min(h, w) // 2 - 2
    cv2.circle(mask, (w//2, h//2), radius, 255, -1)
    filled = cv2.countNonZero(cv2.bitwise_and(patch_bin, mask))
    total = cv2.countNonZero(mask)
    return filled / float(total + 1e-6)


def choose_option(warped_bgr, layout, thr_img=None, threshold=0.25, diff_min=0.08, debug=False):
    """
    Detecta alternativas marcadas com base em pixels (máscara circular).
    Retorna respostas e métricas.
    - threshold: % mínima de preenchimento para considerar marcada
    - diff_min: diferença mínima entre a bolha mais marcada e a segunda colocada
    - debug: se True, retorna imagem com bolhas coloridas
    """
    if thr_img is None:
        gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
        thr_img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 35, 10
        )

    answers = {}
    metrics = {}
    dbg_img = warped_bgr.copy()

    for q in layout["questions"]:
        qid = q["qid"]
        ratios = []

        for b in q["boxes"]:
            x, y, w, h = b
            pad = max(1, int(min(w, h) * 0.20))
            xs, ys = max(0, x - pad), max(0, y - pad)
            xe, ye = x + w + pad, y + h + pad
            patch = thr_img[ys:ye, xs:xe]
            r = _fill_ratio_masked(patch)
            ratios.append(r)

        ratios_np = np.array(ratios, dtype=np.float32)
        best_idx = int(np.argmax(ratios_np))
        best_val = ratios_np[best_idx]

        sorted_idx = np.argsort(ratios_np)[::-1]
        second_val = ratios_np[sorted_idx[1]] if len(sorted_idx) > 1 else 0.0

        if best_val < threshold:
            ans = ""   # nenhuma marcada
            res = "blank"
            color = (255, 0, 0)  # azul
        elif (best_val - second_val) < diff_min and second_val >= threshold:
            ans = ""   # múltipla
            res = "multi"
            color = (0, 255, 255)  # amarelo
        else:
            ans = layout["options"][best_idx]
            res = "marked"
            color = (0, 255, 0)  # verde

        answers[qid] = ans
        metrics[qid] = {
            "ratios": [float(x) for x in ratios],
            "best_idx": best_idx,
            "best_val": float(best_val),
            "second_val": float(second_val),
            "threshold": threshold,
            "diff_min": diff_min
        }

        # Debug visual (desenha círculo colorido sobre a bolha escolhida)
        if debug:
            for i, (x, y, w, h) in enumerate(q["boxes"]):
                cx, cy = x + w//2, y + h//2
                if i == best_idx:
                    cv2.circle(dbg_img, (cx, cy), max(w, h)//2, color, 2)
                else:
                    cv2.circle(dbg_img, (cx, cy), max(w, h)//2, (200, 200, 200), 1)

    if debug:
        return answers, metrics, dbg_img
    return answers, metrics


def compare_answers(ans_student, ans_key):
    """
    Compara respostas do aluno com o gabarito.
    Retorna estatísticas e detalhes por questão.
    """
    correct, wrong, blank, multi = 0, 0, 0, 0
    per_q = {}

    for qid, key in ans_key.items():
        stu = ans_student.get(qid, "")
        if stu == "":
            blank += 1
            res = "blank"
        else:
            if isinstance(key, list):
                if stu in key:
                    correct += 1
                    res = "ok"
                else:
                    wrong += 1
                    res = "wrong"
            else:
                if stu == key:
                    correct += 1
                    res = "ok"
                else:
                    wrong += 1
                    res = "wrong"
        per_q[qid] = {"key": key, "student": stu, "result": res}

    total = len(ans_key)
    score = (correct / total) * 100.0 if total > 0 else 0.0

    return {
        "correct": correct,
        "wrong": wrong,
        "blank": blank,
        "multi": multi,
        "score": score,
        "per_q": per_q
    }