#!/usr/bin/env python3
# -- coding: utf-8 --

import sys
import os

# garante que a raiz do projeto e a pasta corrij_mvp estejam no sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import argparse
import json
import csv
from datetime import datetime

from corrij_mvp.src.align.align import align_image
from corrij_mvp.src.layout import learn_layout_from_key
from corrij_mvp.src.extract import choose_option, compare_answers
from corrij_mvp.src import export_pdf

def processar_provas(gabarito_path, alunos_dir, out_dir,
                     materia="", turma="", escola="", data="",
                     metodo="auto_fallback"):

    os.makedirs(out_dir, exist_ok=True)
    debug_dir = os.path.join(out_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    # -----------------------
    # Processar gabarito
    # -----------------------
    img_key = cv2.imread(gabarito_path)
    if img_key is None:
        raise FileNotFoundError(f"Gabarito não encontrado: {gabarito_path}")

    warped_key, metodo_key, ok_key = align_image(img_key, metodo=metodo, debug_dir=debug_dir)
    if not ok_key:
        raise RuntimeError("Falha no alinhamento do gabarito")

    layout, thr = learn_layout_from_key(warped_key)
    ans_key, _ = choose_option(warped_key, layout, thr)

    # -----------------------
    # Processar provas dos alunos
    # -----------------------
    resultados = []
    for fname in os.listdir(alunos_dir):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg", ".tif", ".bmp")):
            continue

        aluno_nome = os.path.splitext(fname)[0]
        img_path = os.path.join(alunos_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERRO] Falha ao abrir {img_path}")
            continue

        warped, metodo_usado, ok = align_image(img, metodo=metodo, debug_dir=debug_dir)
        if not ok:
            print(f"[ERRO] Não foi possível alinhar a prova de {aluno_nome}")
            continue

        ans_stu, metrics = choose_option(warped, layout)
        stats = compare_answers(ans_stu, ans_key)

        meta = {
            "aluno": aluno_nome,
            "materia": materia,
            "turma": turma,
            "escola": escola,
            "data": data,
            "score": stats["score"],
            "correct": stats["correct"],
            "total": len(ans_key),
            "percentual": stats["score"]
        }

        # salvar PDF individual
        pdf_path = os.path.join(out_dir, f"{aluno_nome}.pdf")
        export_pdf.export_pdf(pdf_path, meta, stats["per_q"])

        # adicionar ao resumo
        resultados.append({
            "aluno": aluno_nome,
            "nota": stats["score"],
            "acertos": stats["correct"],
            "erros": stats["wrong"],
            "brancos": stats["blank"],
            "multiplas": stats["multi"]
        })

    # -----------------------
    # Salvar CSV e JSON
    # -----------------------
    csv_path = os.path.join(out_dir, "resultados.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["aluno","nota","acertos","erros","brancos","multiplas"])
        writer.writeheader()
        for r in resultados:
            writer.writerow(r)

    json_path = os.path.join(out_dir, "resultados.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    print(f"[OK] Processamento concluído! Resultados salvos em {out_dir}")

def main():
    parser = argparse.ArgumentParser(description="CorriJá - Correção de Provas")
    parser.add_argument("--gabarito", required=True, help="Imagem do gabarito")
    parser.add_argument("--alunos", required=True, help="Pasta com imagens das provas dos alunos")
    parser.add_argument("--out", required=True, help="Diretório de saída")
    parser.add_argument("--materia", default="")
    parser.add_argument("--turma", default="")
    parser.add_argument("--escola", default="")
    parser.add_argument("--data", default=datetime.today().strftime("%d/%m/%Y"))
    parser.add_argument("--metodo", default="auto_fallback", choices=["auto","aruco","auto_fallback"],
                        help="Método de alinhamento (default: auto_fallback)")
    args = parser.parse_args()

    processar_provas(args.gabarito, args.alunos, args.out,
                     materia=args.materia, turma=args.turma,
                     escola=args.escola, data=args.data,
                     metodo=args.metodo)

if __name__ == "_main_":
    main()