from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image

from .config import JOBS_DIR
from .gemini import GeminiClient


@dataclass
class PageResult:
    page_number: int
    kept: bool
    original_image: str
    cleaned_image: str | None
    ocr_text_file: str | None
    analysis: str
    blank_score: float
    stamp_count: int = 0
    overlapping_stamp_count: int = 0
    needs_manual_stamp_cleanup: bool = False


@dataclass
class JobManifest:
    job_id: str
    source_filename: str
    status: str
    total_pages: int
    kept_pages: int
    dropped_pages: int
    cleaned_pdf: str | None
    searchable_pdf: str | None
    ocr_text: str | None
    pages: list[PageResult]
    errors: list[str]


def create_job(source_pdf: Path, source_filename: str) -> JobManifest:
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    originals_dir = job_dir / "original_pages"
    cleaned_dir = job_dir / "cleaned_pages"
    ocr_dir = job_dir / "ocr_pages"
    for directory in (job_dir, originals_dir, cleaned_dir, ocr_dir):
        directory.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source_pdf, job_dir / "original.pdf")
    manifest = JobManifest(
        job_id=job_id,
        source_filename=source_filename,
        status="created",
        total_pages=0,
        kept_pages=0,
        dropped_pages=0,
        cleaned_pdf=None,
        searchable_pdf=None,
        ocr_text=None,
        pages=[],
        errors=[],
    )
    save_manifest(job_dir, manifest)
    return manifest


def save_manifest(job_dir: Path, manifest: JobManifest) -> None:
    payload = asdict(manifest)
    (job_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_manifest(job_id: str) -> JobManifest:
    job_dir = JOBS_DIR / job_id
    payload = json.loads((job_dir / "manifest.json").read_text(encoding="utf-8"))
    payload["pages"] = [PageResult(**page) for page in payload.get("pages", [])]
    return JobManifest(**payload)


def render_pdf_pages(pdf_path: Path, output_dir: Path, dpi: int = 360) -> list[Path]:
    doc = fitz.open(pdf_path)
    scale = dpi / 72
    matrix = fitz.Matrix(scale, scale)
    paths: list[Path] = []
    for index, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        path = output_dir / f"page_{index:04d}.png"
        pix.save(path)
        paths.append(path)
    return paths


def estimate_blank_score(gray: np.ndarray) -> float:
    resized = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25)
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edge_ratio = float(np.count_nonzero(edges)) / float(edges.size)
    dark_ratio = float(np.count_nonzero(resized < 150)) / float(resized.size)
    return max(edge_ratio * 9.0, dark_ratio * 3.0)


def content_bbox(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        13,
    )
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, 8)
    h, w = gray.shape
    boxes: list[tuple[int, int, int, int]] = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        bw = stats[label, cv2.CC_STAT_WIDTH]
        bh = stats[label, cv2.CC_STAT_HEIGHT]
        if 6 <= area <= h * w * 0.05 and bw <= w * 0.7 and bh <= h * 0.7:
            boxes.append((x, y, x + bw, y + bh))
    if not boxes:
        return None
    return (
        max(0, min(box[0] for box in boxes)),
        max(0, min(box[1] for box in boxes)),
        min(w, max(box[2] for box in boxes)),
        min(h, max(box[3] for box in boxes)),
    )


def bbox_intersection_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area = max(1, (ax2 - ax1) * (ay2 - ay1))
    return intersection / area


def detect_stamp_regions(bgr: np.ndarray) -> list[dict[str, int | bool]]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    min_radius = max(18, int(min(w, h) * 0.025))
    max_radius = max(min_radius + 5, int(min(w, h) * 0.16))
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.25,
        minDist=max(60, min_radius * 2),
        param1=90,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return []

    content = content_bbox(gray)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    stamps: list[dict[str, int | bool]] = []
    for x_f, y_f, r_f in np.round(circles[0, :]).astype(int):
        radius = int(r_f)
        x1 = max(0, x_f - radius)
        y1 = max(0, y_f - radius)
        x2 = min(w, x_f + radius)
        y2 = min(h, y_f + radius)
        if x2 <= x1 or y2 <= y1:
            continue

        roi_gray = gray[y1:y2, x1:x2]
        roi_hsv = hsv[y1:y2, x1:x2]
        yy, xx = np.ogrid[y1:y2, x1:x2]
        dist = np.sqrt((xx - x_f) ** 2 + (yy - y_f) ** 2)
        disk = dist <= radius * 0.98
        ring = (dist >= radius * 0.72) & (dist <= radius * 1.03)
        if not np.any(disk) or not np.any(ring):
            continue

        dark_ratio = float(np.count_nonzero((roi_gray < 170) & disk)) / float(np.count_nonzero(disk))
        ring_dark_ratio = float(np.count_nonzero((roi_gray < 180) & ring)) / float(np.count_nonzero(ring))
        hue = roi_hsv[:, :, 0]
        stamp_colored_ratio = float(
            np.count_nonzero(
                ((hue > 115) & (hue < 170) & (roi_hsv[:, :, 1] > 35) & (roi_gray < 200) & disk)
            )
        ) / float(np.count_nonzero(disk))
        if ring_dark_ratio < 0.14 and stamp_colored_ratio < 0.02:
            continue
        if dark_ratio < 0.015 or dark_ratio > 0.55:
            continue

        bbox = (x1, y1, x2, y2)
        overlaps = bool(content and bbox_intersection_ratio(bbox, content) > 0.18)
        # Some old purple/blue stamps become very faint in HSV after yellowing.
        # Keep overlapping candidates for manual review when the circular ring is clear.
        if overlaps and stamp_colored_ratio < 0.006:
            continue
        if any(bbox_intersection_ratio(bbox, (int(s["x1"]), int(s["y1"]), int(s["x2"]), int(s["y2"]))) > 0.55 for s in stamps):
            continue
        stamps.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "cx": x_f, "cy": y_f, "r": radius, "overlaps_text": overlaps})
    return stamps


def remove_non_overlapping_stamps(bgr: np.ndarray) -> tuple[np.ndarray, list[dict[str, int | bool]], int]:
    stamps = detect_stamp_regions(bgr)
    safe_stamps = [stamp for stamp in stamps if not stamp["overlaps_text"]]
    if not safe_stamps:
        return bgr, stamps, 0
    mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
    for stamp in safe_stamps:
        cv2.circle(mask, (int(stamp["cx"]), int(stamp["cy"])), int(stamp["r"] * 1.08), 255, -1)
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)
    return cv2.inpaint(bgr, mask, 7, cv2.INPAINT_TELEA), stamps, len(safe_stamps)


def crop_to_content(gray: np.ndarray, color: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        11,
    )
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return color
    h, w = gray.shape
    boxes = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        if area > (w * h * 0.0005):
            boxes.append((x, y, x + bw, y + bh))
    if not boxes:
        return color
    x1 = max(0, min(box[0] for box in boxes) - int(w * 0.035))
    y1 = max(0, min(box[1] for box in boxes) - int(h * 0.035))
    x2 = min(w, max(box[2] for box in boxes) + int(w * 0.035))
    y2 = min(h, max(box[3] for box in boxes) + int(h * 0.035))
    return color[y1:y2, x1:x2]


def deskew(binary_inv: np.ndarray, image: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(binary_inv > 0))
    if len(coords) < 100:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.15 or abs(angle) > 8:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def remove_colored_marks(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    red = (hue < 10) | (hue > 165)
    blue = (hue > 90) & (hue < 135)
    green = (hue > 45) & (hue < 90)
    colored_ink = (red | blue | green) & (saturation > 65) & (value > 45)
    mask = colored_ink.astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    if np.count_nonzero(mask) < 20:
        return bgr
    return cv2.inpaint(bgr, mask, 5, cv2.INPAINT_TELEA)


def clean_page(original_path: Path, cleaned_path: Path) -> tuple[bool, float, list[dict[str, int | bool]], int]:
    bgr = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image {original_path}")

    bgr, stamps, removed_stamp_count = remove_non_overlapping_stamps(bgr)
    bgr = remove_colored_marks(bgr)
    gray_for_blank = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blank_score = estimate_blank_score(gray_for_blank)

    cropped = crop_to_content(gray_for_blank, bgr)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    background = cv2.medianBlur(gray, 31)
    normalized = cv2.divide(gray, background, scale=255)
    normalized = cv2.fastNlMeansDenoising(normalized, None, h=8, templateWindowSize=7, searchWindowSize=21)
    normalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(normalized)

    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_inv = 255 - binary
    binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, np.ones((1, 1), np.uint8), iterations=1)

    min_area = max(2, int(binary_inv.size * 0.000001))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_inv, 8)
    filtered = np.zeros_like(binary_inv)
    significant_components = 0
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        if area >= min_area and width <= binary_inv.shape[1] * 0.95 and height <= binary_inv.shape[0] * 0.95:
            filtered[labels == label] = 255
            if area > 8 and height > 3 and width > 1:
                significant_components += 1

    black_ratio = float(np.count_nonzero(filtered)) / float(filtered.size)
    if blank_score < 0.018 or black_ratio < 0.006 or significant_components < 18:
        return False, blank_score, stamps, removed_stamp_count

    bw = 255 - filtered
    bw_bgr = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    bw_bgr = deskew(filtered, bw_bgr)

    h, w = bw_bgr.shape[:2]
    margin_x = max(60, int(w * 0.08))
    margin_y = max(80, int(h * 0.08))
    canvas = np.full((h + margin_y * 2, w + margin_x * 2, 3), 255, dtype=np.uint8)
    canvas[margin_y : margin_y + h, margin_x : margin_x + w] = bw_bgr

    Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(cleaned_path, optimize=True)
    return True, blank_score, stamps, removed_stamp_count


def images_to_pdf(image_paths: list[Path], pdf_path: Path) -> None:
    doc = fitz.open()
    for image_path in image_paths:
        img = Image.open(image_path)
        width, height = img.size
        page = doc.new_page(width=width, height=height)
        page.insert_image(page.rect, filename=str(image_path))
    doc.save(pdf_path, garbage=4, deflate=True)
    doc.close()


def images_to_searchable_pdf(image_paths: list[Path], ocr_texts: list[str], pdf_path: Path) -> None:
    doc = fitz.open()
    for image_path, text in zip(image_paths, ocr_texts):
        img = Image.open(image_path)
        width, height = img.size
        page = doc.new_page(width=width, height=height)
        page.insert_image(page.rect, filename=str(image_path))
        if text.strip():
            rect = fitz.Rect(20, 20, width - 20, height - 20)
            page.insert_textbox(
                rect,
                text,
                fontsize=8,
                fontname="helv",
                color=(1, 1, 1),
                render_mode=3,
            )
    doc.save(pdf_path, garbage=4, deflate=True)
    doc.close()


def gemini_marked_blank(analysis: str) -> bool:
    lower = analysis.lower()
    return '"is_blank": true' in lower or '"is_blank":true' in lower or "is_blank: true" in lower


def gemini_marked_stamp(analysis: str) -> bool:
    lower = analysis.lower()
    return '"has_stamp": true' in lower or '"has_stamp":true' in lower or "has_stamp: true" in lower


def process_job(job_id: str) -> JobManifest:
    job_dir = JOBS_DIR / job_id
    manifest = load_manifest(job_id)
    manifest.status = "processing"
    save_manifest(job_dir, manifest)

    gemini = GeminiClient()
    originals_dir = job_dir / "original_pages"
    cleaned_dir = job_dir / "cleaned_pages"
    ocr_dir = job_dir / "ocr_pages"
    original_pdf = job_dir / "original.pdf"

    try:
        for directory in (originals_dir, cleaned_dir, ocr_dir):
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)

        manifest.pages = []
        manifest.total_pages = 0
        manifest.kept_pages = 0
        manifest.dropped_pages = 0
        manifest.cleaned_pdf = None
        manifest.searchable_pdf = None
        manifest.ocr_text = None
        manifest.errors = []
        save_manifest(job_dir, manifest)

        original_pages = render_pdf_pages(original_pdf, originals_dir)
        manifest.total_pages = len(original_pages)
        cleaned_images: list[Path] = []
        ocr_texts: list[str] = []

        gemini_rate_limited = False

        for page_number, original_image in enumerate(original_pages, start=1):
            cleaned_image = cleaned_dir / f"page_{page_number:04d}.png"
            kept, blank_score, stamps, removed_stamp_count = clean_page(original_image, cleaned_image)
            analysis = ""
            ocr_file: Path | None = None
            overlapping_stamp_count = sum(1 for stamp in stamps if stamp["overlaps_text"])
            has_stamp_from_ai = False

            if kept:
                if gemini.enabled and not gemini_rate_limited:
                    try:
                        analysis = gemini.analyze_page(cleaned_image)
                        has_stamp_from_ai = gemini_marked_stamp(analysis)
                    except Exception as exc:
                        message = str(exc)
                        manifest.errors.append(f"Gemini analysis failed on page {page_number}: {message}")
                        if "429" in message:
                            gemini_rate_limited = True

                if gemini_marked_blank(analysis):
                    kept = False
                    cleaned_image.unlink(missing_ok=True)
                else:
                    cleaned_images.append(cleaned_image)
                    text = ""
                    if gemini.enabled and not gemini_rate_limited:
                        try:
                            time.sleep(1)
                            text = gemini.ocr_page(cleaned_image, page_number)
                        except Exception as exc:
                            message = str(exc)
                            text = f"[Gemini OCR failed on page {page_number}: {message}]"
                            manifest.errors.append(text.strip("[]"))
                            if "429" in message:
                                gemini_rate_limited = True
                    elif gemini_rate_limited:
                        text = "[Gemini OCR skipped because the API rate limit was reached. Reprocess later to fill OCR text.]"
                    ocr_texts.append(text)
                    ocr_file = ocr_dir / f"page_{page_number:04d}.txt"
                    ocr_file.write_text(text, encoding="utf-8")

            manifest.pages.append(
                PageResult(
                    page_number=page_number,
                    kept=kept,
                    original_image=str(original_image.relative_to(job_dir)),
                    cleaned_image=str(cleaned_image.relative_to(job_dir)) if kept else None,
                    ocr_text_file=str(ocr_file.relative_to(job_dir)) if ocr_file else None,
                    analysis=analysis,
                    blank_score=blank_score,
                    stamp_count=max(len(stamps), 1 if has_stamp_from_ai else 0),
                    overlapping_stamp_count=max(overlapping_stamp_count, 1 if has_stamp_from_ai and removed_stamp_count == 0 else 0),
                    needs_manual_stamp_cleanup=bool(has_stamp_from_ai and removed_stamp_count == 0) or overlapping_stamp_count > 0,
                )
            )
            save_manifest(job_dir, manifest)

        manifest.kept_pages = len(cleaned_images)
        manifest.dropped_pages = manifest.total_pages - manifest.kept_pages

        cleaned_pdf = job_dir / "cleaned.pdf"
        searchable_pdf = job_dir / "searchable.pdf"
        ocr_text = job_dir / "ocr.txt"
        images_to_pdf(cleaned_images, cleaned_pdf)
        images_to_searchable_pdf(cleaned_images, ocr_texts, searchable_pdf)
        ocr_text.write_text(
            "\n\n".join(f"--- Page {i + 1} ---\n{text}" for i, text in enumerate(ocr_texts)),
            encoding="utf-8",
        )

        manifest.cleaned_pdf = "cleaned.pdf"
        manifest.searchable_pdf = "searchable.pdf"
        manifest.ocr_text = "ocr.txt"
        manifest.status = "completed"
    except Exception as exc:
        manifest.status = "failed"
        manifest.errors.append(str(exc))

    save_manifest(job_dir, manifest)
    return manifest
