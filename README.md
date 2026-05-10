# AMD Hacks RestorePDF

Local PDF restoration portal for turning damaged scanned PDFs into cleaner black-and-white PDFs with OCR text output.

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
