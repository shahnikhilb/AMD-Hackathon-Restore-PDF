# AMD Hacks RestorePDF

Local PDF restoration portal for turning damaged scanned PDFs into cleaner black-and-white PDFs with OCR text output.
Old scanned documents have a yellow tint, random smudges and marks, and general dirt/dust making them unpresentable at best and unreadable at worst. Clean-PDF is a project which tackles this very issue.
We have already used traditional computer vision models in order to perform the preliminary cleaning and now want to move on to more advanced steps like stamp removal, smudge removal, and general artifact removal, making the pipeline extremely strong and robust.



## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

## Outputs

Each upload creates a job in `storage/jobs/<job_id>/` with:

- `original.pdf`
- `cleaned.pdf`
- `searchable.pdf`
- `ocr.txt`
- per-page original/cleaned images
- per-page OCR text
- `manifest.json`

## Gemini

Set `GEMINI_API_KEY` in `.env`. Gemini is used for OCR and page analysis. The image cleanup itself is deterministic OpenCV processing so the app can still restore pages predictably.
