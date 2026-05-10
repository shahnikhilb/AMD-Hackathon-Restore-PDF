const uploadForm = document.querySelector("#uploadForm");
const pdfInput = document.querySelector("#pdfInput");
const statusEl = document.querySelector("#status");
const pageList = document.querySelector("#pageList");
const originalPreview = document.querySelector("#originalPreview");
const cleanPreview = document.querySelector("#cleanPreview");
const ocrText = document.querySelector("#ocrText");
const downloads = document.querySelector("#downloads");
const stampReview = document.querySelector("#stampReview");
const stampReviewButton = document.querySelector("#stampReviewButton");
const stampReviewSummary = document.querySelector("#stampReviewSummary");

let currentJob = null;
let pollTimer = null;
const apiBase = (() => {
  const match = window.location.pathname.match(/^(.*\/proxy\/\d+)/);
  return match ? match[1] : "";
})();

function jobFile(job, path) {
  return `${apiBase}/api/jobs/${job.job_id}/files/${path}`;
}

function setStatus(text) {
  statusEl.textContent = text;
}

async function fetchText(url) {
  const response = await fetch(url);
  if (!response.ok) return "";
  return response.text();
}

async function selectPage(page, button) {
  document.querySelectorAll(".page-row").forEach((row) => row.classList.remove("active"));
  button.classList.add("active");
  originalPreview.src = jobFile(currentJob, page.original_image);
  cleanPreview.src = page.cleaned_image ? jobFile(currentJob, page.cleaned_image) : "";
  cleanPreview.alt = page.kept ? "Cleaned page" : "Dropped blank page";
  if (page.ocr_text_file) {
    ocrText.textContent = await fetchText(jobFile(currentJob, page.ocr_text_file));
  } else {
    ocrText.textContent = page.kept ? "OCR pending or unavailable." : "Dropped as blank or near-blank.";
  }
}

function renderDownloads(job) {
  if (job.status !== "completed") {
    downloads.classList.add("hidden");
    downloads.innerHTML = "";
    return;
  }

  downloads.classList.remove("hidden");
  downloads.innerHTML = `
    <a href="${jobFile(job, job.cleaned_pdf)}" target="_blank">Clean PDF</a>
    <a href="${jobFile(job, job.searchable_pdf)}" target="_blank">Searchable PDF</a>
    <a href="${jobFile(job, job.ocr_text)}" target="_blank">OCR Text</a>
  `;
}

function renderStampReview(job) {
  const pages = (job.pages || []).filter((page) => page.needs_manual_stamp_cleanup);
  if (!pages.length) {
    stampReview.classList.add("hidden");
    stampReviewSummary.textContent = "";
    return;
  }

  const pageNumbers = pages.map((page) => page.page_number).join(", ");
  stampReview.classList.remove("hidden");
  stampReviewSummary.textContent = `Manual cleanup needed on page${pages.length === 1 ? "" : "s"}: ${pageNumbers}`;
  stampReviewButton.onclick = () => {
    alert(`Overlapping stamp detected on page${pages.length === 1 ? "" : "s"}: ${pageNumbers}`);
  };
}

function renderJob(job) {
  currentJob = job;
  const errors = job.errors?.length ? ` Errors: ${job.errors.join(" | ")}` : "";
  setStatus(
    `${job.status.toUpperCase()} · ${job.kept_pages || 0}/${job.total_pages || 0} pages kept · ${job.dropped_pages || 0} dropped.${errors}`,
  );
  renderDownloads(job);
  renderStampReview(job);

  pageList.innerHTML = "";
  job.pages.forEach((page, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `page-row ${page.kept ? "" : "dropped"}`;
    const stampLabel = page.needs_manual_stamp_cleanup ? " · stamp review" : "";
    button.textContent = `Page ${page.page_number} · ${page.kept ? "kept" : "dropped"}${stampLabel}`;
    button.addEventListener("click", () => selectPage(page, button));
    pageList.appendChild(button);
    if (index === 0) selectPage(page, button);
  });
}

async function pollJob(jobId) {
  const response = await fetch(`${apiBase}/api/jobs/${jobId}`);
  if (!response.ok) {
    setStatus("Could not load job status.");
    return;
  }
  const job = await response.json();
  renderJob(job);
  if (job.status === "completed" || job.status === "failed") {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = pdfInput.files?.[0];
  if (!file) {
    setStatus("Choose a PDF first.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  setStatus("Uploading PDF...");

  const response = await fetch(`${apiBase}/api/jobs`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    setStatus("Upload failed.");
    return;
  }

  const job = await response.json();
  renderJob(job);
  setStatus("Processing started. Pages will appear as they finish.");
  clearInterval(pollTimer);
  pollTimer = setInterval(() => pollJob(job.job_id), 2500);
  pollJob(job.job_id);
});
