#!/usr/bin/env python3
# -- coding: utf-8 --
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors

def export_pdf(out_path, meta, per_q):
    """
    Gera relatório PDF individual do aluno.
    - meta: dict com aluno, turma, materia, escola, data, score, correct, total
    - per_q: dict com detalhes por questão {qid: {"key": x, "student": y, "result": res}}
    """
    c = canvas.Canvas(out_path, pagesize=A4)
    width, height = A4

    # Cabeçalho
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width/2, height - 80, "Relatório de Correção - CorriJá")

    c.setFont("Helvetica", 12)
    c.drawString(2*cm, height - 120, f"Aluno: {meta.get('aluno','')}")
    c.drawString(2*cm, height - 140, f"Turma: {meta.get('turma','')}   Escola: {meta.get('escola','')}")
    c.drawString(2*cm, height - 160, f"Matéria: {meta.get('materia','')}   Data: {meta.get('data','')}")
    c.drawString(2*cm, height - 180, f"Nota: {meta.get('score',0):.1f} ({meta.get('correct',0)}/{meta.get('total',0)} acertos)")

    # Legenda
    c.setFont("Helvetica", 11)
    legenda_y = height - 210
    c.setFillColor(colors.green);  c.drawString(2*cm, legenda_y, "✅ Correto")
    c.setFillColor(colors.red);    c.drawString(5*cm, legenda_y, "❌ Errado")
    c.setFillColor(colors.blue);   c.drawString(8*cm, legenda_y, "⭕ Em branco")
    c.setFillColor(colors.orange); c.drawString(12*cm, legenda_y, "⚠ Múltipla")
    c.setFillColor(colors.black)

    # Questões
    start_y = legenda_y - 30
    x_col1 = 2*cm
    x_col2 = 10*cm
    y = start_y
    line_height = 18

    for qid, info in per_q.items():
        if y < 3*cm:  # quebra de página
            c.showPage()
            y = height - 100
            c.setFont("Helvetica", 11)

        key = info["key"]
        student = info["student"] if info["student"] != "" else "-"
        result = info["result"]

        if result == "ok":
            c.setFillColor(colors.green)
            mark = "✅"
        elif result == "wrong":
            c.setFillColor(colors.red)
            mark = "❌"
        elif result == "blank":
            c.setFillColor(colors.blue)
            mark = "⭕"
        elif result == "multi":
            c.setFillColor(colors.orange)
            mark = "⚠"
        else:
            c.setFillColor(colors.black)
            mark = "?"

        c.drawString(x_col1, y, f"Q{qid:02d} - Gabarito: {key} | Aluno: {student}")
        c.drawString(x_col2, y, f"Resultado: {mark}")

        y -= line_height

    c.setFillColor(colors.black)
    c.showPage()
    c.save()