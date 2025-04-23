# OCR Handwriting Model - Enterprise Debug Version
# This script extracts text from PDF with preprocessing, filtering, bounding boxes,
# confidence tiers, and dictionary validation. Includes full terminal output of confidence.

# --- Import Required Libraries ---
import os
from pdf2image import convert_from_path
import pytesseract
import cv2
import numpy as np
from PIL import Image
import time
from datetime import datetime
from spellchecker import SpellChecker

# --- Advanced Image Preprocessing ---
def preprocess_image(pil_image):
    """
    Enhance image quality for OCR by applying:
    - Grayscale conversion
    - Gaussian blur (denoise)
    - Adaptive thresholding (binarization)
    - Morphological closing (fills gaps in strokes)
    - Contrast Limited Adaptive Histogram Equalization (CLAHE)
    - Upscaling
    - Optional: Data augmentations (noise, distortion, thickness)
    """
    cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((2, 2), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized = clahe.apply(closed)

    # Noise injection (salt-and-pepper)
    noisy = equalized.copy()
    prob = 0.01
    mask = np.random.choice((0, 1, 2), size=noisy.shape, p=[1 - prob, prob/2, prob/2])
    noisy[mask == 1] = 255
    noisy[mask == 2] = 0

    # Distortion (random warp)
    h, w = noisy.shape
    src_points = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
    dst_points = src_points + np.random.normal(0, 3, src_points.shape).astype(np.float32)
    matrix = cv2.getAffineTransform(src_points, dst_points)
    warped = cv2.warpAffine(noisy, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Thickness modification (dilation)
    thick_kernel = np.ones((1, 1), np.uint8)
    thickened = cv2.dilate(warped, thick_kernel, iterations=1)

    scaled = cv2.resize(thickened, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(scaled)

# --- Deskew Function ---
def deskew_image(pil_image):
    """
    Automatically corrects skewed (angled) handwriting by rotating the image
    based on the detected text orientation.
    """
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
    coords = np.column_stack(np.where(img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(np.array(pil_image), M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(rotated)

# --- Draw Bounding Boxes Function ---
def draw_bounding_boxes(image, data, save_path):
    """
    Draws bounding boxes and overlays confidence score for each detected word.
    Saves annotated image to the output folder for visual debugging.
    """
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for i in range(len(data['text'])):
        word = data['text'][i]
        conf = int(data['conf'][i]) if data['conf'][i] != '-1' else -1
        if word.strip() and conf > 0:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            color = (0, 255, 0) if conf >= 75 else (0, 255, 255) if conf >= 60 else (0, 0, 255)
            cv2.rectangle(image_cv, (x, y), (x + w, y + h), color, 2)
            label = f"{conf}%"
            cv2.putText(image_cv, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)
    cv2.imwrite(save_path, image_cv)
    print(f"🖼️ Annotated image saved to: {save_path}")

# --- Paths and Timestamps ---
PDF_PATH = "/home/vellj/OCRHW/image/pgtwo.pdf"
OUTPUT_FOLDER = "/home/vellj/OCRHW/output"
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
readable_time = datetime.now().strftime("%Y-%m-%d %I:%M %p")
output_filename = f"extracted_text_{timestamp}.txt"
OUTPUT_TEXT_FILE = os.path.join(OUTPUT_FOLDER, output_filename)

# --- Setup Output Directory ---
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Convert PDF to Images ---
print("🔁 Converting PDF to images...")
try:
    pages = convert_from_path(PDF_PATH, dpi=300)
except Exception as e:
    print(f"❌ Error converting PDF: {e}")
    exit(1)

total_pages = len(pages)
print(f"📄 Total Pages in Document: {total_pages}")
print("\n👉 Enter a page number to scan (e.g., 1), or `0` to scan all pages.")

try:
    selected_page = int(input("🔢 Select page: "))
except ValueError:
    print("❌ Invalid input. Must be a number.")
    exit(1)

if selected_page > total_pages:
    print(f"❌ Page {selected_page} is out of range. Document only has {total_pages} pages.")
    exit(1)

# --- OCR Configuration ---
print("\n🧠 Starting OCR with bounding boxes, filtering, and confidence tiers...\n")
start_time = time.time()
s = SpellChecker()
HIGH_CONF = 75
LOW_CONF = 60
CONFIDENCE_THRESHOLD = 45

total_global_conf = 0
all_valid_words = 0

with open(OUTPUT_TEXT_FILE, "w") as output_file:
    output_file.write(f"🕒 Processed on: {readable_time}\n")
    output_file.write(f"📄 Source File: {os.path.basename(PDF_PATH)}\n")
    output_file.write(f"📥 Output File: {output_filename}\n\n")

    page_range = range(total_pages) if selected_page == 0 else [selected_page - 1]

    for page_index in page_range:
        page = pages[page_index]
        deskewed = deskew_image(page)
        processed_image = preprocess_image(deskewed)
        custom_config = r'--oem 1 --psm 3'
        data = pytesseract.image_to_data(processed_image, config=custom_config, output_type=pytesseract.Output.DICT)

        debug_image_path = os.path.join(OUTPUT_FOLDER, f"annotated_page_{page_index + 1}.png")
        draw_bounding_boxes(processed_image, data, debug_image_path)

        print(f"\n--- Page {page_index + 1} ---")
        output_file.write(f"\n--- Page {page_index + 1} ---\n")

        total_conf = 0
        valid_words = 0
        clean_words = []
        questionable_words = []

        for word, conf in zip(data['text'], data['conf']):
            word = word.strip()
            if word in ['.', ',', '-', '_']:
                continue  # Skip stand-alone punctuation
            if word and conf != '-1':
                conf_val = int(conf)
                print(f"📌 Word: '{word}' | Confidence: {conf_val}")
                if len(word) >= 2:
                    if conf_val >= HIGH_CONF:
                        clean_words.append(word)
                        valid_words += 1
                        total_conf += conf_val
                        output_file.write(f"{word} ")
                    elif LOW_CONF <= conf_val < HIGH_CONF:
                        questionable_words.append(f"{word}({conf_val})")
                        output_file.write(f"_{word}_ ")
                else:
                    print(f"❌ Skipped '{word}' | Conf: {conf_val}")

        avg_conf = total_conf / valid_words if valid_words else 0
        total_global_conf += total_conf
        all_valid_words += valid_words

        print(f"🔎 Avg Confidence (Page {page_index + 1}): {avg_conf:.2f}%")
        output_file.write(f"\n\n🔎 Avg Confidence (Page {page_index + 1}): {avg_conf:.2f}%\n")
        if questionable_words:
            output_file.write(f"\n⚠️  Questionable Words:\n{' '.join(questionable_words)}\n")

# --- Final Wrap Up ---
total_time = time.time() - start_time
if all_valid_words:
    global_avg_conf = total_global_conf / all_valid_words
    print(f"\n📊 Overall Average Confidence: {global_avg_conf:.2f}%")
    with open(OUTPUT_TEXT_FILE, "a") as output_file:
        output_file.write(f"\n📊 Overall Average Confidence: {global_avg_conf:.2f}%\n")

print(f"\n✅ OCR Complete. Text saved to: {OUTPUT_TEXT_FILE}")
print(f"⏱️ Processing Time: {total_time:.2f} seconds")

