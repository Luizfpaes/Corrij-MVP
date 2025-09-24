#!/usr/bin/env python3
# -- coding: utf-8 --
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import shutil
import tempfile
import zipfile
import cv2
import numpy as np

from src.align import align_image
from src.layout import learn_layout_from_key, preprocess
from src.extract import choose_option, compare_answers
from src.export_pdf import export_pdf

app = FastAPI(title="CorriJá API", description="Correção de provas via FastAPI", version="1.0")

def read_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem: {path}")
    return img

def grade_pipeline(gabarito_path, alunos_zip_path, out_dir, metodo="auto_fallback"):
    out_dir = Path(out_dir)
    (out_dir/"csv").mkdir(parents=True, exist_ok=True)
    (out_dir/"json").mkdir(parents=True, exist_ok=True)
    (out_dir/"pdf").mkdir(parents=True, exist_ok=True)
    (out_dir/"debug").mkdir(parents=True, exist_ok=True)

    # 1) Warp gabarito
    key_img = read_image(gabarito_path)
    warped_key, metodo_key, ok_key = align_image(key_img, metodo=metodo, debug_dir=out_dir/"debug")
    if not ok_key:
        raise RuntimeError("Falha no alinhamento do gabarito")

    cv2.imwrite(str(out_dir/"debug"/"warped_key.jpg"), warped_key)

    # 2) Layout do gabarito
    layout, thr_key = learn_layout_from_key(warped_key)
    ans_key, metrics_key = choose_option(warped_key, layout, thr_img=thr_key)
    key_map = {int(k): v for k, v in ans_key.items()}

    # 3) Processar alunos.zip
    notas = []
    with zipfile.ZipFile(alunos_zip_path, 'r') as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if not name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                continue
            data = zf.read(name)
            file_bytes = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                print(f"[WARN] Não consegui abrir {name}")
                continue
            try:
                warped, metodo_usado, ok = align_image(img, metodo=metodo, debug_dir=out_dir/"debug")
                if not ok:
                    print(f"[WARN] Falha no alinhamento de {name}")
                    continue
            except Exception as e:
                print(f"[WARN] Falha no alinhamento de {name}: {e}")
                continue

            thr = preprocess(warped)
            ans_student, metrics = choose_option(warped, layout, thr_img=thr)
            comp = compare_answers(ans_student, key_map)
            comp['total'] = len(key_map)

            meta = {
                "score": comp['score'],
                "correct": comp['correct'],
                "total": comp['total'],
                "aluno": Path(name).stem,
                "materia": "",
                "data": "",
                "turma": "",
                "escola": "",
            }
            pdf_path = out_dir/"pdf"/f"{Path(name).stem}.pdf"
            export_pdf(str(pdf_path), meta, comp['per_q'])

            notas.append((Path(name).stem, comp['score'], comp['correct'], comp['total']))

    # CSV final
    csv_path = out_dir/"csv"/"notas.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("aluno,nota,acertos,total\n")
        for (aluno, nota, acertos, total) in notas:
            f.write(f"{aluno},{nota:.1f},{acertos},{total}\n")

    return csv_path, out_dir/"pdf"

@app.post("/corrigir")
async def corrigir(
    gabarito: UploadFile,
    alunos: UploadFile,
    metodo: str = Form("auto_fallback")
):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            gabarito_path = tmpdir/"gabarito.jpg"
            alunos_zip_path = tmpdir/"alunos.zip"
            out_dir = tmpdir/"resultados"

            # salvar arquivos enviados
            with open(gabarito_path, "wb") as f:
                shutil.copyfileobj(gabarito.file, f)
            with open(alunos_zip_path, "wb") as f:
                shutil.copyfileobj(alunos.file, f)

            # processar
            csv_path, pdf_dir = grade_pipeline(str(gabarito_path), str(alunos_zip_path), str(out_dir), metodo=metodo)

            # retorna CSV como resposta
            return FileResponse(path=csv_path, filename="notas.csv", media_type="text/csv")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
        
        
@app.get("/")
def root():
    return {"status": "ok", "message": "CorriJá API rodando 🚀"}

@app.get("/health")
def health():
    return {"alive": True}