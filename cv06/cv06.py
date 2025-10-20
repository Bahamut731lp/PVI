import cv2
import numpy as np
from pathlib import Path
from collections import deque
from matplotlib import pyplot as plt

def granulometric_spectrum(binary_image, min_size=5, max_size=50, step=1, return_heatmap=False):
    """
    Vytvoří granulometrické spektrum binárního obrazu.
    
    Parametry:
        binary_image : np.ndarray
            Binární obraz (0/255).
        min_size, max_size : int
            Minimální a maximální velikost SE (čtvercový).
        step : int
            Krok zvětšení SE.
        return_heatmap : bool
            Jestli vrátit heatmapu přežívajících pixelů.

    Vrací:
        spectrum_sizes : np.ndarray
            Velikosti SE (odpovídají diferencovanému spektru).
        estimated_objects : np.ndarray
            Odhadovaný počet objektů pro každou velikost SE.
        heatmap : np.ndarray (volitelně)
            Heatmapa přežívajících pixelů (normalizovaná 0..1).
    """
    binary = np.uint8(binary_image > 0) * 255
    structure_sizes = list(range(min_size, max_size + 1, step))
    
    # počet přežívajících pixelů pro každé SE
    survived_pixels = []
    diffs = []

    # volitelná heatmapa
    if return_heatmap:
        heatmap = np.zeros_like(binary, dtype=np.float32)

    for size in structure_sizes:
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, se)
        survived_pixels.append(np.sum(opened > 0))

        if return_heatmap:
            heatmap += opened / 255.0  # pixel=1 pokud přežije

    survived_pixels = np.array(survived_pixels)

    diff = survived_pixels[:-1] - survived_pixels[1:]
    spectrum_sizes = np.array(structure_sizes[:-1])
    estimated_objects = diff / (spectrum_sizes**2)  # odhad počtu objektů

    diffs.append(diff)

    if return_heatmap:
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return spectrum_sizes, estimated_objects, heatmap, diffs[0]
    else:
        return spectrum_sizes, estimated_objects, diffs[0]



def get_region_values(image, points):
    values = []

    for point in points:
        region_number = image[point[1]][point[0]]
        number_of_pixels = len(np.argwhere(image == region_number))
        
        value = 1

        if number_of_pixels > 4000:
            value = 2
        if number_of_pixels > 4800:
            value = 5
        if number_of_pixels > 5400:
            value = 10
        
        values.append((point, number_of_pixels, value))

    return values

def main():
    image_source = Path("cv06/pvi_cv06_mince.jpg")
    image = cv2.imread(image_source.as_posix(), cv2.IMREAD_COLOR_RGB)

    hue = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    kernel = np.ones((3, 3), np.uint8)

    plt.figure("Segmentation")

    # Hue image
    plt.subplot(3, 3, 1)
    plt.title("Hue Image")
    plt.imshow(hue, cmap="jet")
    plt.colorbar()

    # Hue Histogram
    plt.subplot(3, 3, 2)
    plt.title("Hue Histogram")
    plt.plot(cv2.calcHist([hue], [0], None, [180], [0, 180]))

    plt.subplot(3, 3, 3)
    plt.title("Hue Segmentation")
    segmented_hue = hue.copy()[:, :, 0]
    segmented_hue_threshold = 35
    segmented_hue[segmented_hue < segmented_hue_threshold] = 1
    segmented_hue[segmented_hue >= segmented_hue_threshold] = 0
    plt.imshow(segmented_hue, cmap="jet")
    plt.colorbar()

    # Watershed
    binary = np.uint8(segmented_hue > 0) * 255
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=5)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    segmented_hue_watershed = cv2.watershed(image, markers)

    borders = np.uint8(segmented_hue_watershed == -1)
    borders = cv2.dilate(borders, np.ones((3, 3), np.uint8), iterations=3)
    outlined = image.copy()
    outlined[borders == 1] = [0, 255, 0]

    # Vizualizace markerů (volitelné)
    plt.subplot(3, 3, 4)
    plt.title("Distance Transform")
    plt.imshow(dist_transform, cmap="jet")
    plt.colorbar()

    plt.subplot(3, 3, 5)
    plt.title("Sure Foreground")
    plt.imshow(cv2.cvtColor(sure_fg, cv2.COLOR_BGR2RGB))
    plt.colorbar()

    plt.subplot(3, 3, 6)
    plt.title("Unknown")
    plt.imshow(unknown, cmap="jet")
    plt.colorbar()

    plt.subplot(3, 3, 7)
    plt.title("Markers")
    plt.imshow(markers, cmap="jet")
    plt.colorbar()

    plt.subplot(3, 3, 8)
    plt.title("Watershed Borders")
    plt.imshow(outlined, cmap="jet")
    plt.colorbar()

    plt.subplot(3, 3, 9)
    plt.title("Region Identification")
    uncolored_regions = cv2.subtract(segmented_hue, borders)
    plt.imshow(uncolored_regions, cmap="jet")
    plt.colorbar()

    plt.figure("Identification")
    plt.subplot(2, 2, 1)
    plt.title("Identified Regions")
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(uncolored_regions)
    filtered_labels = np.zeros_like(labels_im, dtype=np.uint8)
    for i in range(1, num_labels):  # skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 1000:
            filtered_labels[labels_im == i] = 255  # keep this region

    # Optionally, relabel filtered regions for visualization
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(filtered_labels)
    plt.imshow(labels_im, cmap="jet")
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.title("Region Centers")
    plt.imshow(uncolored_regions, cmap="jet")
    plt.scatter(*zip(*centroids[1:]), marker="+", color="green")

    plt.subplot(2, 2, 3)
    plt.title("Granulometry")
    spectrum_sizes, estimated_objects, heatmap, diffs = granulometric_spectrum(
        uncolored_regions, min_size=40, max_size=80, step=1, return_heatmap=True
    )

    estimated_objects[estimated_objects < 0.9] = 0
    estimated_objects = np.floor(estimated_objects)

    indices = np.nonzero(estimated_objects)[0]
    
    for index in indices:
        print(f"No. objects: {estimated_objects[index]} size: {spectrum_sizes[index]} x {spectrum_sizes[index]}")

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.imshow(heatmap, cmap='jet')
    plt.colorbar()

    plt.subplot(2, 2, 4)
    print(spectrum_sizes, diffs)
    plt.plot(spectrum_sizes, diffs)
    
    plt.show()

if __name__ == "__main__":
    main()