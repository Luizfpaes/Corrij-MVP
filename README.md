# CorriJá MVP — Correção automática por gabarito do professor (sem JSON de ROIs)

## O que faz
- Professor envia **1 foto** do gabarito (`--gabarito`).
- Envia **.zip** com as provas dos alunos (`--zip`).
- O sistema **aprende o layout** (posições das bolhas) **a partir do gabarito**, corrige todas as provas e gera:
  - `saida/csv/notas.csv` e `saida/json/*`
  - **PDF por aluno** com overlay e tabela de acertos (em `saida/pdf/`).

> Não precisa de arquivo `.json` de ROIs. Pode usar **ArUco** nos 4 cantos para warp estável ou **detecção automática** do contorno da folha.
> Para parecer com o exemplo fornecido, o PDF inclui nota, acertos/erros e tabela (referência enviada).

## Instalação (recomendado dentro de venv)
```bash
pip install -r requirements.txt
```

## Como rodar
```bash
python main.py --gabarito ./provas/gabarito_professor.jpg --zip ./provas/alunos.zip --out ./saida --metodo auto
```
- `--metodo auto` usa **detecção automática de cantos** (funciona com qualquer prova).
- `--metodo aruco` usa **ArUco** (requer marcadores IDs 0,1,2,3 nos cantos TL,TR,BL,BR).

### Exemplo (Windows PowerShell)
```powershell
python .\main.py --gabarito ".\provas\gabarito_professor.jpg" --zip ".\provas\alunos.zip" --out ".\saida" --metodo auto
```

## Estrutura
```
corrija_mvp/
  main.py
  src/
    align/
      aruco_align.py
      auto_corners_align.py
    layout.py
    extract.py
    export_pdf.py
  provas/
  saida/
    csv/
    json/
    pdf/
    debug/
  requirements.txt
  README.md
```
