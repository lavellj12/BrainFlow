# OCR Handwriting Model - Enhanced Version with Timestamped Output
# This script converts PDF files to images, preprocesses them, runs OCR,
# and outputs text with confidence scores and versioned output files.

import os
import time
from datetime import datetime
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import cv2
import numpy as np

# =======================
# Configuration
# =======================

PDF_PATH = "/home/vellj/OCRHW/image/test2.pdf"  # Path to the input PDF
OUTPUT_FOLDER = "/home/vellj/OCRHW/output"      # Directory for output files
PAGE_IMAGE_DPI = 300                             # Image resolution for PDF conversion
TESSERACT_CONFIG = r'--oem 1 --psm 6'            # OCR engine and page segmentation mode

# Generate a unique timestamped filename for each run to prevent overwriting
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_TEXT_FILE = os.path.join(OUTPUT_FOLDER, f"extracted_text_{timestamp}.txt")

# =======================
# Image Preprocessing Function
# =======================

def preprocess_image(pil_image):
    """Enhance image quality for better OCR accuracy."""
    # Convert PIL image to OpenCV format
    cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply adaptive thresholding for binarization
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Resize image to improve OCR recognition
    scaled = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Convert back to PIL format
    return Image.fromarray(scaled)

# =======================
# Ensure Output Directory Exists
# =======================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =======================
# Main OCR Pipeline
# =======================

print("🔁 Converting PDF to images...")
start_time = time.time()

try:
    pages = convert_from_path(PDF_PATH, dpi=PAGE_IMAGE_DPI)
except Exception as e:
    print(f"❌ Error converting PDF: {e}")
    exit(1)

print("🧠 Running OCR on each page...\n")

with open(OUTPUT_TEXT_FILE, "w") as output_file:
    for i, page in enumerate(pages):
        page_path = os.path.join(OUTPUT_FOLDER, f"page_{i + 1}.png")
        page.save(page_path, "PNG")

        # Preprocess the image
        preprocessed = preprocess_image(page)

        # Run OCR and collect data
        data = pytesseract.image_to_data(preprocessed, config=TESSERACT_CONFIG, output_type=pytesseract.Output.DICT)

        words = []
        confidences = []
        for j, word in enumerate(data['text']):
            if word.strip() != "":
                conf = int(data['conf'][j])
                words.append(word)
                confidences.append(conf)

        full_text = " ".join(words)
        avg_conf = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

        # Log results
        print(f"--- Page {i + 1} ---")
        print(full_text)
        print(f"📊 Average Confidence: {avg_conf}%\n")

        output_file.write(f"\n--- Page {i + 1} ---\n{full_text}\nAverage Confidence: {avg_conf}%\n")

end_time = time.time()

# =======================
# Final Summary
# =======================

print(f"\n✅ OCR complete. Output saved to: {OUTPUT_TEXT_FILE}")
print(f"⏱️ Total Processing Time: {round(end_time - start_time, 2)} seconds")

