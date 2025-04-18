## OCR Handwriting Model – Development Phase
## Author: VellJ | Version: Enterprise Accuracy Prep

import os
from datetime import datetime
from pdf2image import convert_from_path
import pytesseract
import cv2
import numpy as np
from PIL import Image

# ==========================
# 🧠 Preprocessing Functions
# ==========================

def deskew(image):
    """Deskews image to correct for scanning tilt or alignment issues."""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(np.array(image), M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(rotated)

def preprocess_image(pil_image):
    """Enhances the image for better OCR accuracy."""
    # Convert PIL to OpenCV format
    cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Resize the image (upscale for better recognition)
    scaled = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Convert back to PIL format
    return Image.fromarray(scaled)

# ==========================
# 📁 File and Path Setup
# ==========================

PDF_PATH = "/home/vellj/OCRHW/image/test2.pdf"
OUTPUT_FOLDER = "/home/vellj/OCRHW/output"
OUTPUT_TEXT_FILE = os.path.join(OUTPUT_FOLDER, "extracted_text1.txt")

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================
# 🔄 Process Flow
# ==========================

print(f"📅 Job started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("🔁 Converting PDF to images...")

try:
    pages = convert_from_path(PDF_PATH, dpi=300)
except Exception as e:
    print(f"❌ Error converting PDF: {e}")
    exit(1)

print("🧠 Running OCR on each page...")

with open(OUTPUT_TEXT_FILE, "w") as output_file:
    for i, page in enumerate(pages):
        page_path = os.path.join(OUTPUT_FOLDER, f"page_{i + 1}.png")
        page.save(page_path, "PNG")

        # 🔍 Preprocess and deskew the image
        clean_image = preprocess_image(deskew(page))

        # 🧠 OCR with custom config
        custom_config = r'--oem 1 --psm 6'
        text = pytesseract.image_to_string(clean_image, config=custom_config)

        # 💾 Save results
        print(f"\n--- Page {i + 1} ---\n{text.strip()}")
        output_file.write(f"\n--- Page {i + 1} ---\n{text.strip()}\n")

print(f"\n✅ OCR complete. Output saved to: {OUTPUT_TEXT_FILE}")

