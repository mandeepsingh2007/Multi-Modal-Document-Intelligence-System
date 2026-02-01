import fitz  # PyMuPDF
import numpy as np
from typing import List, Tuple
from PIL import Image
import io

class IngestionService:
    def __init__(self):
        pass

    def convert_pdf_to_images(self, pdf_bytes: bytes) -> List[Image.Image]:
        """
        Convert PDF bytes to a list of PIL Images using PyMuPDF (no Poppler required).
        """
        images = []
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page in doc:
                pix = page.get_pixmap()
                img_data = pix.tobytes("ppm")
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                images.append(img)
            return images
        except Exception as e:
            print(f"Error converting PDF to images: {e}")
            return []

    def load_image(self, file_bytes: bytes) -> Image.Image:
        """
        Load an image from bytes.
        """
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")

    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """
        Attempt to extract text directly from PDF layer (digital PDF).
        Returns empty string if failed or no text found.
        """
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text() + "\n\n"
            return text.strip()
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
