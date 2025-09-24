#!/usr/bin/env python3
# -- coding: utf-8 --

import sys
import os

# Garante que a raiz do projeto e a pasta corrij_mvp estejam no sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import streamlit as st
from datetime import datetime
import json
import csv

from corrij_mvp.src.align.align import align_image   # <- corrigido
from corrij_mvp.src import layout, extract, export_pdf

def processar(gabarito_file, alunos_files, out_dir, materia, turma, escola, data, metodo="auto_fallback"):
    os.makedirs(out_dir, exist_ok=True)
    debug_dir = os.path.join(out_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    # Gabarito
    img_key = cv2.imdecode(np.frombuffer(gabarito_file.read(), np.uint8), cv2.IMREAD_COLOR)
    warped_key, metodo_key, ok_key = align_image(img_key, metodo=metodo, debug_dir=debug_dir)
    if not ok_key:
        st.error("Falha no alinhamento do gabarito.")
        return

    layout_data, thr = layout.learn_layout_from_key(warped_key)
    ans_key, _ = extract.choose_option(warped_key, layout_data, thr)

    resultados = []

    for aluno_file in alunos_files:
        img = cv2.imdecode(np.frombuffer(aluno_file.read(), np.uint8), cv2.IMREAD_COLOR)
        aluno_nome = os.path.splitext(aluno_file.name)[0]

        warped, metodo_usado, ok = align_image(img, metodo=metodo, debug_dir=debug_dir)
        if not ok:
            st.warning(f"Não foi possível alinhar a prova de {aluno_nome}")
            continue

        ans_stu, metrics = extract.choose_option(warped, layout_data)
        stats = extract.compare_answers(ans_stu, ans_key)

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

        pdf_path = os.path.join(out_dir, f"{aluno_nome}.pdf")
        export_pdf.export_pdf(pdf_path, meta, stats["per_q"])

        resultados.append({
            "aluno": aluno_nome,
            "nota": stats["score"],
            "acertos": stats["correct"],
            "erros": stats["wrong"],
            "brancos": stats["blank"],
            "multiplas": stats["multi"]
        })

    # Salvar CSV e JSON
    csv_path = os.path.join(out_dir, "resultados.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["aluno","nota","acertos","erros","brancos","multiplas"])
        writer.writeheader()
        for r in resultados:
            writer.writerow(r)

    json_path = os.path.join(out_dir, "resultados.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    return resultados, csv_path, json_path, out_dir


# -------------------------------
# Interface Streamlit
# -------------------------------
st.set_page_config(page_title="CorriJá - Correção de Provas", layout="wide")
st.title("📘 CorriJá - Correção Automática de Provas")

with st.sidebar:
    st.header("Configurações")
    materia = st.text_input("Matéria", "Matemática")
    turma = st.text_input("Turma", "3A")
    escola = st.text_input("Escola", "Colégio Estadual MS")
    data = st.date_input("Data da prova", datetime.today()).strftime("%d/%m/%Y")
    metodo = st.selectbox("Método de alinhamento", ["auto_fallback", "auto", "aruco"], index=0)

gabarito_file = st.file_uploader("Upload do gabarito", type=["jpg","jpeg","png"])
alunos_files = st.file_uploader("Upload das provas dos alunos", type=["jpg","jpeg","png"], accept_multiple_files=True)

if st.button("🚀 Processar") and gabarito_file and alunos_files:
    # Pasta fixa para salvar resultados
    out_dir = os.path.join(os.getcwd(), "saida_streamlit")
    os.makedirs(out_dir, exist_ok=True)

    resultados, csv_path, json_path, out_dir = processar(
        gabarito_file, alunos_files, out_dir, materia, turma, escola, data, metodo
    )

    if resultados:
        st.success("✅ Processamento concluído!")
        st.subheader("📊 Resultados")
        st.dataframe(resultados)

        # Download CSV e JSON
        with open(csv_path, "rb") as f:
            st.download_button("📥 Baixar CSV", f, "resultados.csv")

        with open(json_path, "rb") as f:
            st.download_button("📥 Baixar JSON", f, "resultados.json")

        # PDFs individuais
        st.subheader("📑 Relatórios Individuais")
        for r in resultados:
            pdf_file = os.path.join(out_dir, f"{r['aluno']}.pdf")
            if os.path.exists(pdf_file):
                with open(pdf_file, "rb") as f:
                    st.download_button(f"📥 {r['aluno']}.pdf", f, f"{r['aluno']}.pdf", "application/pdf")