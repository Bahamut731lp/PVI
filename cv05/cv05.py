import cv2
import numpy as np
from pathlib import Path
from collections import deque
from matplotlib import pyplot as plt

def bfs_coloring(image):
    """Funkce pro barvení oblastí pomocí prohledávání do šířky

    Args:
        image: Binární obrázek k barvení

    Returns:
        Matice s očíslovanými oblastmi
    """
    rows, cols = image.shape
    visited = np.zeros_like(image).astype("bool")
    colors = np.zeros_like(image).astype("uint8")

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def is_valid(x, y):
        return 0 <= x < rows and 0 <= y < cols

    def bfs(x, y, color):
        queue = deque([(x, y)])
        visited[x][y] = True
        colors[x][y] = color

        # Zkoumáme všechny okolní body, dokud je v rámci obrázku, nebyli jsme na něm a je vysegmentovaný jako 1
        while queue:
            current_x, current_y = queue.popleft()

            for dx, dy in directions:
                new_x, new_y = current_x + dx, current_y + dy
                if is_valid(new_x, new_y) and not visited[new_x][new_y] and image[new_x][new_y] == 1:
                    queue.append((new_x, new_y))
                    visited[new_x][new_y] = True
                    colors[new_x][new_y] = color

    color_count = 1
    for i in range(rows):
        for j in range(cols):
            if not visited[i][j] and image[i][j] == 1:
                color_count += 1
                bfs(i, j, color_count)

    return colors

def get_centers_of_mass(image):
    """Funkce pro výpočet těžiště v segmentovaných oblastech

    Args:
        image: Matice s očíslovanými oblastmi. Pozadí je označeno číslem 0.

    Returns:
        Pole bodů těžišť.
    """
    points = []

    for i in range(2, np.max(image) + 1):
        copy = np.zeros_like(image)
        copy[image == i] = 1

        moments = cv2.moments(copy, True)
        points.append([int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])])

    return points

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
    image_source = Path("cv05/pvi_cv05_mince_noise.png")
    image = cv2.imread(image_source.as_posix(), cv2.IMREAD_COLOR_RGB)

    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hue = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    kernel = np.ones((3, 3), np.uint8)

    plt.figure("Segmentation")

    # Grayscale image
    plt.subplot(2, 3, 1)
    plt.title("Grayscale Image")
    plt.imshow(grayscale, cmap="gray")
    plt.colorbar()

    # Grayscale Histogram
    plt.subplot(2, 3, 2)
    plt.title("Grayscale Histogram")
    plt.plot(cv2.calcHist([grayscale], [0], None, [256], [0, 256]))

    # Grayscale Segmentation
    plt.subplot(2, 3, 3)
    plt.title("Grayscale Segmentation")
    segmented_grayscale = grayscale.copy()
    segmented_grayscale_threshold = 140
    segmented_grayscale[segmented_grayscale < segmented_grayscale_threshold] = 0
    segmented_grayscale[segmented_grayscale >= segmented_grayscale_threshold] = 1
    plt.imshow(segmented_grayscale, cmap="jet")
    plt.colorbar()

    # Hue image
    plt.subplot(2, 3, 4)
    plt.title("Hue Image")
    plt.imshow(hue, cmap="jet")
    plt.colorbar()

    # Hue Histogram
    plt.subplot(2, 3, 5)
    plt.title("Hue Histogram")
    plt.plot(cv2.calcHist([hue], [0], None, [180], [0, 180]))

    # Grayscale Segmentation
    plt.subplot(2, 3, 6)
    plt.title("Hue Segmentation")
    segmented_hue = hue.copy()[:, :, 0]
    segmented_hue_threshold = 75
    segmented_hue[segmented_hue < segmented_hue_threshold] = 1
    segmented_hue[segmented_hue >= segmented_hue_threshold] = 0
    plt.imshow(segmented_hue, cmap="jet")
    plt.colorbar()

    plt.figure("Cleanup Process")
    plt.subplot(2, 2, 1)
    plt.title("Hue Segmentation")
    plt.imshow(segmented_hue, cmap="jet")

    plt.subplot(2, 2, 2)
    plt.title("Hue Segmentation - Opening")
    segmented_hue_opened = segmented_hue.copy()
    segmented_hue_opened = cv2.erode(segmented_hue_opened, kernel)
    segmented_hue_opened = cv2.dilate(segmented_hue_opened, kernel)
    plt.imshow(segmented_hue_opened, cmap="jet")

    plt.subplot(2, 2, 3)
    plt.title("Color Labeling")
    regions = bfs_coloring(segmented_hue_opened)
    plt.imshow(regions, cmap="jet")

    plt.subplot(2, 2, 3)
    plt.title("Color Labeling")
    regions = bfs_coloring(segmented_hue_opened)
    plt.imshow(regions, cmap="jet")

    plt.subplot(2, 2, 4)
    plt.title("Color Labeling - Centers")
    points = get_centers_of_mass(regions)
    plt.imshow(regions, cmap="jet")
    plt.scatter(*zip(*points), marker="+", color="green")

    weights = get_region_values(regions, points)
    
    plt.figure("Final Output")
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    for point, test, value in weights:
        plt.text(point[0] + 0.1, point[1] + 0.1, test, color="#0000ff", fontsize=16)

    plt.subplot(1, 2, 2)
    plt.imshow(image)
    for point, test, value in weights:
        plt.text(point[0] + 0.1, point[1] + 0.1, f"{value} CZK", color="#0000ff", fontsize=16)
    
    plt.show()
    

if __name__ == "__main__":
    main()