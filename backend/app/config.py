from pathlib import Path
from dotenv import load_dotenv
import os


ROOT_DIR = Path(__file__).resolve().parents[2]
STORAGE_DIR = ROOT_DIR / "storage"
JOBS_DIR = STORAGE_DIR / "jobs"
STATIC_DIR = ROOT_DIR / "frontend" / "static"

load_dotenv(ROOT_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash")

JOBS_DIR.mkdir(parents=True, exist_ok=True)
