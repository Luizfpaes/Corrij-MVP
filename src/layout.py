#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np

def preprocess(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    thr = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 35, 10)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    return thr

def detect_bubbles(thr_img, min_area=120, max_area=5000, min_circ=0.65):
    contours, _ = cv2.findContours(thr_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri == 0: 
            continue
        circularity = 4.0*np.pi*area/(peri*peri)
        if circularity < min_circ:
            continue
        x,y,w,h = cv2.boundingRect(cnt)
        cx,cy = x+w/2, y+h/2
        bubbles.append((cx,cy,x,y,w,h,area,circularity,cnt))
    bubbles.sort(key=lambda b: (b[1], b[0]))
    return bubbles

def cluster_rows(bubbles, y_tol=12):
    rows = []
    for b in bubbles:
        placed = False
        for row in rows:
            if abs(row['y_mean'] - b[1]) <= y_tol:
                row['items'].append(b)
                row['y_mean'] = np.mean([it[1] for it in row['items']])
                placed = True
                break
        if not placed:
            rows.append({'items':[b], 'y_mean': b[1]})
    rows.sort(key=lambda r: r['y_mean'])
    for r in rows:
        r['items'].sort(key=lambda it: it[0])
    return rows

def learn_layout_from_key(warped_bgr, expected_options=5):
    thr = preprocess(warped_bgr)
    bubbles = detect_bubbles(thr)
    rows = cluster_rows(bubbles, y_tol=14)
    filtered = [r for r in rows if len(r['items']) >= expected_options]
    questions = []
    qid = 1
    for r in filtered:
        items = r['items']
        for i in range(0, len(items), expected_options):
            block = items[i:i+expected_options]
            if len(block) < expected_options:
                continue
            boxes = []
            for it in block:
                _,_,x,y,w,h,_,_,_ = it
                boxes.append([int(x),int(y),int(w),int(h)])
            questions.append({"qid": qid, "boxes": boxes})
            qid += 1
    return {"questions": questions, "options": ["A","B","C","D","E"]}, thr
