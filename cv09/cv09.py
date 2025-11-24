import os
import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt

import skimage

TEMPLATE_PATH = ""
MIN_MATCH_COUNT = 10
RATIO_TEST = 0.6

TEMPLATE_NAME_BOX = (200, 120, 900, 220)
TEMPLATE_PHOTO_BOX = (30, 60, 180, 240)
FIGURE_SUBPLOT_GRID = (2, 2)
SAMPLES_SUBPLOT_GRID = (3, 3)

template = cv2.imread("cv09/obcansky_prukaz_cr_sablona_2012_2014.png", cv2.IMREAD_COLOR_RGB)
template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)

sift = skimage.feature.SIFT()

sift.detect_and_extract(template_gray)
template_kps = sift.keypoints
template_desc = sift.descriptors

samples = [
    "cv09/TS10_01.jpg",
    "cv09/TA10_01.jpg",
    "cv09/PA10_30.jpg",
    "cv09/KS10_02.jpg",
    "cv09/KA10_01.jpg",
    "cv09/HA10_06.jpg",
    "cv09/HS10_12.jpg",
    "cv09/CA10_01.jpg",
    "cv09/10_cze_id.jpg",
]

samples = [cv2.imread(x, cv2.IMREAD_COLOR_RGB) for x in samples]
grayscales = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in samples]
keypoints = []
descriptors = []

fig_1, ax_1 = plt.subplots(*SAMPLES_SUBPLOT_GRID, figsize=(12, 12))
fig_1.suptitle("SIFT Keypoints")
fig_1.tight_layout()

fig_2, ax_2 = plt.subplots(*SAMPLES_SUBPLOT_GRID, figsize=(12, 12))
fig_2.suptitle("Matches")
fig_2.tight_layout()

fig_3, ax_3 = plt.subplots(*SAMPLES_SUBPLOT_GRID, figsize=(12, 12))
fig_3.suptitle("Corrected Perspective")
fig_3.tight_layout()

for index, sample_gray in enumerate(grayscales):
    sift.detect_and_extract(sample_gray)
    sample_kps = sift.keypoints
    sample_desc = sift.descriptors

    matches = skimage.feature.match_descriptors(
        template_desc,
        sample_desc,
        max_ratio=RATIO_TEST,
        cross_check=True
    )

    matched_kps0 = template_kps[matches[:, 0]]
    matched_kps1 = sample_kps[matches[:, 1]]
    new_matches = np.column_stack((np.arange(len(matches)), np.arange(len(matches))))

    r = index // 3
    c = index % 3

    skimage.feature.plot_matched_features(
        template_gray,
        sample_gray,
        keypoints0=matched_kps0,
        keypoints1=matched_kps1,
        matches=new_matches,
        ax=ax_1[r, c]
    )

    ax_1[r, c].set_title(f"Features {index}")

    # Druhý figure s obkreslenou kartou
    ax_2[r, c].imshow(samples[index])
    ax_2[r, c].set_title(f"Outlined {index}")

    if len(matches) > MIN_MATCH_COUNT:
        src = template_kps[matches[:, 0]][:, ::-1]
        dst = sample_kps[matches[:, 1]][:, ::-1]

        # projektivní transformace (homografie)
        tform = skimage.transform.estimate_transform('projective', src, dst)

        # rohové body šablony
        h, w = template_gray.shape
        h_1, w_1 = sample_gray.shape
        corners = np.array([[0,0], [w,0], [w,h], [0,h]], dtype=float)

        # promítnutí rohů do vzorku
        projected = tform(corners)

        # obkreslení (čtyřúhelník)
        projected = projected.astype(int)
        for i in range(4):
            p1 = projected[i]
            p2 = projected[(i+1) % 4]
            ax_2[r, c].plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=3, color='red')

        # narovnání podle homografie
        unwarped = skimage.transform.warp(
            sample_gray,
            tform,
            output_shape=(h, w)
        )

        corrected = (unwarped * 255).astype(np.uint8)
        ax_3[r, c].imshow(corrected, cmap="gray")
        ax_3[r, c].set_title(f"Corrected {index}")
        ax_3[r, c].axis('off')

        #(78, 40), (130, 55)        
        surname = corrected[40:55, 78:130]

        #(75, 55), (130, 70)
        firstname = corrected[54:66, 77:115]

        surname_text: str = pytesseract.image_to_string(surname, config='--psm 6')
        fistname_text: str = pytesseract.image_to_string(firstname, config='--psm 6')

        print(f"Sample {index}: {surname_text.strip()} {fistname_text.strip()}")

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(surname)
        plt.subplot(1, 3, 2)
        plt.imshow(firstname)
        plt.subplot(1, 3, 3)
        plt.imshow(corrected[67:200, 10:120])

plt.show()