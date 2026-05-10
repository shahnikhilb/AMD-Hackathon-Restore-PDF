from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import JOBS_DIR, STATIC_DIR
from .pipeline import create_job, load_manifest, process_job


app = FastAPI(title="RestorePDF Local Portal")


@app.post("/api/jobs")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a PDF file.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = Path(tmp.name)

    manifest = create_job(temp_path, file.filename)
    temp_path.unlink(missing_ok=True)
    background_tasks.add_task(process_job, manifest.job_id)
    return manifest


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return load_manifest(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc


@app.post("/api/jobs/{job_id}/reprocess")
def reprocess_job(job_id: str, background_tasks: BackgroundTasks):
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found.")
    background_tasks.add_task(process_job, job_id)
    return {"status": "queued"}


@app.get("/api/jobs/{job_id}/files/{path:path}")
def get_job_file(job_id: str, path: str):
    job_dir = (JOBS_DIR / job_id).resolve()
    requested = (job_dir / path).resolve()
    if not str(requested).startswith(str(job_dir)) or not requested.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(requested)


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
