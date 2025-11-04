import os
import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt

TEMPLATE_PATH = ""
TEST_IMAGE_PATH = "cv09/some_test_image.jpg"   # změň na vybraný test obrázek z archivu
MIN_MATCH_COUNT = 10
RATIO_TEST = 0.75  # Lowe ratio
# Pokud je potřeba, nastav cestu k tesseractu na Windows:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Tyto boxy jsou v souřadnicích šablony (x1,y1,x2,y2) — uprav podle skutečné šablony.
# Příklad: oblast jména (ulož v šablonních pixelech)
TEMPLATE_NAME_BOX = (200, 120, 900, 220)  # <-- uprav podle šablony
TEMPLATE_PHOTO_BOX = (30, 60, 180, 240)    # <-- uprav podle šablony

# --- Helpery ---
def show_image(title, img, figsize=(10,6)):
    plt.figure(figsize=figsize)
    if img.ndim == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# načtení obrázku (BGR)
template = cv2.imread("cv09/obcansky_prukaz_cr_sablona_2012_2014.png")
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()  # vyžaduje opencv-contrib
kp1, des1 = sift.detectAndCompute(template_gray, None)
print(f"Template keypoints: {len(kp1)}")