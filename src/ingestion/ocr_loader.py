"""OCR extraction for scanned PDFs."""

from __future__ import annotations

import os
import platform

import pytesseract
from pdf2image import convert_from_path

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_text_from_scanned_pdf(file_path: str) -> str:
    filename = os.path.basename(file_path)
    text_content = f"# Tai lieu Scan: {filename}\n\n"

    images = convert_from_path(file_path)
    for index, image in enumerate(images, start=1):
        page_text = pytesseract.image_to_string(image, lang="vie")
        text_content += f"## Trang {index}\n\n{page_text}\n\n"

    return text_content

