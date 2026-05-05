from __future__ import annotations

import base64
import json
import logging
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

MAX_TEXT_LENGTH = 80_000

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff", ".tif"}
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml", ".log"}
OFFICE_EXTENSIONS = {".docx", ".xlsx", ".pptx"}
PDF_EXTENSIONS = {".pdf"}

_ALL_SUPPORTED = IMAGE_EXTENSIONS | TEXT_EXTENSIONS | OFFICE_EXTENSIONS | PDF_EXTENSIONS


@dataclass
class FileExtractionResult:
    images: list[str] = field(default_factory=list)
    text_parts: list[str] = field(default_factory=list)

    @property
    def combined_text(self) -> str:
        return "\n\n".join(self.text_parts)


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return "[PDF text extraction unavailable — pypdf not installed]"

    try:
        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append(f"--- Page {i + 1} ---\n{text.strip()}")
        return "\n\n".join(pages) if pages else ""
    except Exception as exc:  # noqa: BLE001
        logging.warning("PDF extraction failed for %s: %s", path.name, exc)
        return f"[PDF extraction error: {exc}]"


def _extract_docx_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(str(path)) as zf:
            with zf.open("word/document.xml") as f:
                tree = ET.parse(f)
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs = tree.findall(".//w:p/w:r/w:t", ns)
        return "\n".join(p.text for p in paragraphs if p.text)
    except Exception as exc:  # noqa: BLE001
        logging.warning("DOCX extraction failed for %s: %s", path.name, exc)
        return f"[DOCX extraction error: {exc}]"


def _extract_xlsx_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(str(path)) as zf:
            with zf.open("xl/sharedStrings.xml") as f:
                tree = ET.parse(f)
        ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        strings = tree.findall(".//s:si/s:t", ns)
        return "\n".join(s.text for s in strings if s.text)
    except Exception as exc:  # noqa: BLE001
        logging.warning("XLSX extraction failed for %s: %s", path.name, exc)
        return f"[XLSX extraction error: {exc}]"


def _extract_pptx_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(str(path)) as zf:
            slides = sorted(name for name in zf.namelist() if name.startswith("ppt/slides/slide") and name.endswith(".xml"))
            parts = []
            for i, slide_name in enumerate(slides, 1):
                with zf.open(slide_name) as f:
                    tree = ET.parse(f)
                ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
                texts = tree.findall(".//a:t", ns)
                slide_text = "\n".join(t.text for t in texts if t.text)
                if slide_text.strip():
                    parts.append(f"--- Slide {i} ---\n{slide_text}")
            return "\n\n".join(parts)
    except Exception as exc:  # noqa: BLE001
        logging.warning("PPTX extraction failed for %s: %s", path.name, exc)
        return f"[PPTX extraction error: {exc}]"


def _truncate(text: str, max_len: int = MAX_TEXT_LENGTH) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n\n... [truncated — file exceeds {max_len:,} characters]"


def _format_text_part(filename: str, content: str) -> str:
    return f"--- Attached file: {filename} ---\n{content}\n--- End of file: {filename} ---"


def process_uploaded_files(file_paths: list[str | Path]) -> FileExtractionResult:
    result = FileExtractionResult()

    for raw_path in file_paths:
        path = Path(raw_path)
        suffix = path.suffix.lower()

        if suffix not in _ALL_SUPPORTED:
            logging.warning("Unsupported file type skipped: %s", path.name)
            continue

        if suffix in IMAGE_EXTENSIONS:
            try:
                data = path.read_bytes()
                b64 = base64.b64encode(data).decode("ascii")
                result.images.append(b64)
            except Exception as exc:  # noqa: BLE001
                logging.warning("Failed to read image %s: %s", path.name, exc)

        elif suffix in TEXT_EXTENSIONS:
            content = _truncate(_read_text_file(path))
            result.text_parts.append(_format_text_part(path.name, content))

        elif suffix in PDF_EXTENSIONS:
            content = _truncate(_extract_pdf_text(path))
            if content:
                result.text_parts.append(_format_text_part(path.name, content))

        elif suffix == ".docx":
            content = _truncate(_extract_docx_text(path))
            if content:
                result.text_parts.append(_format_text_part(path.name, content))

        elif suffix == ".xlsx":
            content = _truncate(_extract_xlsx_text(path))
            if content:
                result.text_parts.append(_format_text_part(path.name, content))

        elif suffix == ".pptx":
            content = _truncate(_extract_pptx_text(path))
            if content:
                result.text_parts.append(_format_text_part(path.name, content))

    return result
