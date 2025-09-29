# Modul numpy obsahuje vsechno potrebne pro praci s poli a maticemi. Skladba
# prikazu (nazvy i parametry) velmi verne kopiruje Matlab.
from collections import Counter
import numpy as np

# Knihovna matplotlib se stara o vykreslovani grafu do oken ci souboru
# (podporuje # napr. export do pdf). Submodul pyplot pak nabizi syntaxi velmi
# podobnou jazyku Matlab.
import matplotlib.pyplot as plt

# OpenCV je velmi rozsirena C++ knihovna obsahujici nastroje pro zpracovani a
# rozpoznavani obrazu. Modul cv2 je wrapper umoznujici jeji pouziti primo
# v Pythonu.
import cv2

image_data = cv2.imread("cv02/cv02_01.bmp")
image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2HSV)

image_data_2 = cv2.imread("cv02/cv02_02.bmp")
image_data_2 = cv2.cvtColor(image_data_2, cv2.COLOR_BGR2HSV)

def image_colors(image):
    # zploštíme na seznam pixelů
    result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).reshape(-1, 3)

    # klasifikace pixelů
    labels = [classify_color(h,s,v) for h,s,v in pixels]
    # spočítáme četnosti
    counts = Counter(labels)
    total = counts.total()
    
    top3 = counts.most_common(3)
    
    for i, (color, _) in enumerate(top3):
        cv2.putText(result, f"{color}: {counts.get(color)/total*100:.2f}%", (10, 30 + i*30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    return result


def dominant_hsv_color(cell):
    h_vals = cell[:, :, 0].flatten()
    s_vals = cell[:, :, 1].flatten()
    v_vals = cell[:, :, 2].flatten()

    # histogram pro Hue (větší biny ~ 10°)
    hist_h, bins = np.histogram(h_vals, bins=64, range=(0, 180))
    h_mode_bin = np.argmax(hist_h)
    h_mode = int((bins[h_mode_bin] + bins[h_mode_bin+1]) / 2)

    # Median pro stabilitu
    s_median = int(np.median(s_vals))
    v_median = int(np.median(v_vals))

    return h_mode, s_median, v_median

def classify_color(h, s, v):
    # If saturated enough, treat as color first
    if s >= 40:  # threshold for “is a color”
        if h < 10 or h >= 160:  # red/pink
            if v > 200 and s < 180:
                return "pink"
            return "red"
        elif 10 <= h < 25:
            if v < 150:
                return "brown"
            return "orange"
        elif 25 <= h < 35:
            return "yellow"
        elif 35 <= h < 85:
            return "green"
        elif 85 <= h < 125:
            return "blue"
        elif 125 <= h < 160:
            return "purple"

    # Low saturation: treat as grayscale
    if v < 40:
        return "black"
    if s < 30 and v > 200:
        return "white"
    if s < 40:
        return "gray"

    return "unknown"


def uloha_1():
    plt.imshow(cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB))

def uloha_2(image):
    height, width, _ = image.shape
    grid_rows = 3
    grid_cols = 3

    result = cv2.cvtColor(image.copy(), cv2.COLOR_HSV2RGB)
    cell_height, cell_width = height // grid_rows, width // grid_cols

    for i in range(grid_rows):
        for j in range(grid_cols):
            # výřez buňky
            cell = image[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]

            # detekce barvy
            h_mode, s_median, v_median = dominant_hsv_color(cell)
            color_name = classify_color(h_mode, s_median, v_median)

            # vybrat barvu textu podle jasu buňky
            if v_median < 128:
                text_color = (255, 255, 255)  # světlý text na tmavém pozadí
            else:
                text_color = (0, 0, 0)        # tmavý text na světlém pozadí

            # souřadnice středu buňky
            cx = j*cell_width + cell_width // 2
            cy = i*cell_height + cell_width // 2

            # vykreslení textu
            cv2.putText(
                result,
                color_name,
                (cx-40, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                text_color,
                2,
                cv2.LINE_AA
            )

    return result


if __name__ == "__main__":
    rows = 2
    cols = 3

    plt.figure("Cvičení 2")
    plt.subplot(rows, cols, 1)
    uloha_1()

    plt.subplot(rows, cols, 2)
    plt.imshow(uloha_2(image_data))

    plt.subplot(rows, cols, 3)
    plt.imshow(uloha_2(image_data_2))

    cv01 = cv2.imread("./cv02/cv01_u01.jpg")
    plt.subplot(rows, cols, 4)
    plt.imshow(image_colors(cv01))

    cv02 = cv2.imread("./cv02/cv01_u02.jpg")
    plt.subplot(rows, cols, 5)
    plt.imshow(image_colors(cv02))

    cv03 = cv2.imread("./cv02/cv01_u03.jpg")
    plt.subplot(rows, cols, 6)
    plt.imshow(image_colors(cv03))

    plt.show()
