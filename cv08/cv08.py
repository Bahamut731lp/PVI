import cv2
import math
import scipy
import skimage
import pytesseract
import itertools
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

SPZ_WINDOW_GRID = (3, 3)
SUNFLOWER_WINDOW_GRID = (4, 5)

def order_points(pts):
    # seřadí body: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left má nejmenší součet
    rect[2] = pts[np.argmax(s)]  # bottom-right má největší součet

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right má nejmenší rozdíl (x - y)
    rect[3] = pts[np.argmax(diff)]  # bottom-left má největší rozdíl

    return rect

def calc_hue_hist(region):
    # Převod do HSV
    hsl = cv2.cvtColor(region, cv2.COLOR_RGB2HLS)
    hue = hsl[:, :, 0]

    hist = cv2.calcHist([hue], [0], None, [180], [0, 180])
    hist = hist.flatten()
    hist = hist + 0.001  # malé číslo proti logaritmování nul
    hist /= hist.max()
    return hist

def iou(boxA, boxB):
    # box format: (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - interArea
    if union == 0:
        return 0

    return interArea / union

def read_bounding_box_data(path: Path):
    boxes = []
    with open(path.as_posix()) as f:
        lines = f.read().splitlines()
        for line in lines:
            vec = line.split(' ')
            vec = [int(x) for x in vec]
            boxes.append(vec)

    return boxes

def render_image(image, label, position):
    plt.subplot(*SUNFLOWER_WINDOW_GRID, position)
    plt.title(label)
    plt.imshow(image)

def render_image_grayscale(image, label, position):
    plt.subplot(*SUNFLOWER_WINDOW_GRID, position)
    plt.title(label)
    plt.imshow(image, cmap="gray")
    plt.colorbar()

def render_image_with_bounding_box(image, bounding_boxes, label, position, color=(255,255,255)):
    copy = image.copy()

    for box in bounding_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(copy, (x1, y1), (x2, y2), color, 2)

    plt.subplot(*SUNFLOWER_WINDOW_GRID, position)
    plt.title(label)
    plt.imshow(copy)

def main():
    image_source = Path("cv08/pvi_cv08_spz.png")
    image = cv2.imread(image_source.as_posix(), cv2.IMREAD_COLOR_RGB)

    plt.figure("SPZ")
    plt.subplot(*SPZ_WINDOW_GRID, 1)
    plt.title("Original")
    plt.imshow(image)

    _, segmented = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 0, 1, 0)
    
    plt.subplot(*SPZ_WINDOW_GRID, 2)
    plt.title("Segmented")
    plt.imshow(segmented, cmap="gray")

    closed = cv2.morphologyEx(segmented, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=5)
    
    plt.subplot(*SPZ_WINDOW_GRID, 3)
    plt.title("Closed")
    plt.imshow(closed, cmap="gray")

    harris = cv2.cornerHarris(np.float32(closed.copy()), 2, 3, 0.04)
    harris = cv2.dilate(harris, None, iterations=10)

    plt.subplot(*SPZ_WINDOW_GRID, 4)
    plt.title("Harris")
    plt.imshow(harris, cmap="gray")

    corners = harris.copy()
    _, corners = cv2.threshold(corners, 0.3 * corners.max(), 1, 0)
    corners = np.uint8(corners)
    
    plt.subplot(*SPZ_WINDOW_GRID, 5)
    plt.title("Corners")
    plt.imshow(corners, cmap="jet")

    _, labels, stats, centroids = cv2.connectedComponentsWithStats(corners)
    
    plt.subplot(*SPZ_WINDOW_GRID, 6)
    plt.title("Labels")
    plt.imshow(labels, cmap="jet")

    centroids = np.round(centroids)
    centroids = np.uint64(centroids)

    image_with_corners = image.copy()
    centroids = np.array(list(itertools.islice(centroids, 1, None)))

    for centroid in centroids:
        cv2.circle(image_with_corners, (centroid[0], centroid[1]), 15, (255, 0, 0), 5)

    plt.subplot(*SPZ_WINDOW_GRID, 7)
    plt.title("Labels")
    plt.imshow(image_with_corners, cmap="jet")
    
    hull = cv2.convexHull(centroids)
    edges = [(hull[i][0], hull[(i+1) % len(hull)][0]) for i in range(len(hull))]
    lengths = [np.linalg.norm(e[1] - e[0]) for e in edges]
    i = np.argmax(lengths)
    p1, p2 = edges[i]

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.degrees(math.atan2(dy, dx))

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    size = np.max(image.shape)

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2

    rotated = np.full((size, size, 3), (0, 0, 0), dtype=image.dtype)
    rotated[y_offset:y_offset + h, x_offset:x_offset + w] = image

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(rotated, M, (rotated.shape[0], rotated.shape[1]))

    plt.subplot(*SPZ_WINDOW_GRID, 8)
    plt.title("Rotated Image")
    plt.imshow(rotated)

    _, thresh = cv2.threshold(rotated, 0, 255, cv2.THRESH_BINARY)

    text: str = pytesseract.image_to_string(thresh, config='--psm 6')
    text = text.strip()
    print(text)

    cv2.putText(thresh, text, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255))
    plt.subplot(*SPZ_WINDOW_GRID, 9)
    plt.title("OCR")
    plt.imshow(thresh)

    # --------------------------------------------------------
    # Slunečnice
    # --------------------------------------------------------
    plt.figure("Slunečnice")
    image_source = Path("cv08/pvi_cv08_sunflower_template.jpg")
    image = cv2.imread(image_source.as_posix(), cv2.IMREAD_COLOR_RGB)

    samples = [
        Path("cv08/pvi_cv08_sunflowers1.jpg"),
        Path("cv08/pvi_cv08_sunflowers2.jpg"),
        Path("cv08/pvi_cv08_sunflowers3.jpg"),
        Path("cv08/pvi_cv08_sunflowers4.jpg")
    ]

    bounding_boxes = [
        Path("cv08/pvi_cv08_sunflowers1.txt"),
        Path("cv08/pvi_cv08_sunflowers2.txt"),
        Path("cv08/pvi_cv08_sunflowers3.txt"),
        Path("cv08/pvi_cv08_sunflowers4.txt")
    ]

    samples = [cv2.imread(x.as_posix(), cv2.IMREAD_COLOR_RGB) for x in samples]
    bounding_boxes = [read_bounding_box_data(x) for x in bounding_boxes]

    plt.subplot(*SUNFLOWER_WINDOW_GRID, 1)
    plt.title("Original")
    plt.imshow(image)

    hue = calc_hue_hist(image.copy())
    plt.subplot(*SUNFLOWER_WINDOW_GRID, 6)
    plt.title("Histogram odstínu")
    plt.plot(hue)

    for index, sample in enumerate(samples):
        render_image(sample, f"Vzor {index + 1}", 2 + index)

    for index, sample in enumerate(samples):
        render_image_with_bounding_box(sample, bounding_boxes[index], f"Anotace {index + 1}", 7 + index)

    for index, sample in enumerate(samples):
        result = sample.copy()

        regions = []
        filtered = []
        h, w = sample.shape[:2]
        grayscale = 255 - cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
        blobs_log = skimage.feature.blob_log(grayscale, min_sigma=10, max_sigma=100, num_sigma=20, threshold=0.15)

        for y, x, r in blobs_log:
            cv2.circle(result, (int(x), int(y)), int(r), (128, 0, 255), 5)
            x1 = int(max(x - 2*r, 0))
            y1 = int(max(y - 2*r, 0))
            x2 = int(min(x + 2*r, w))
            y2 = int(min(y + 2*r, h))
    
            regions.append((sample[y1:y2, x1:x2], (x1, y1, x2, y2)))

        render_image(result, f"LoG {index + 1}", 17 + index)
        
        for region, bbox in regions:
            image_hue = hue
            region_hue = calc_hue_hist(region)
            kl_divergence = scipy.stats.entropy(image_hue, region_hue)

            # Filtrace podle prahu divergence
            if kl_divergence < 1.48:
                filtered.append(bbox)

        render_image_with_bounding_box(sample, filtered, "Nalezeno", 12 + index, (255, 0, 0))

        # Ground truth and predicted boxes
        gt_boxes = bounding_boxes[index]
        pred_boxes = filtered

        IOU_THRESHOLD = 0.5

        matched_gt = set()
        TP = 0
        FP = 0

        for pred in pred_boxes:
            best_iou = 0
            best_gt = None

            for i, gt in enumerate(gt_boxes):
                current_iou = iou(pred, gt)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gt = i

            if best_iou >= IOU_THRESHOLD and best_gt not in matched_gt:
                TP += 1
                matched_gt.add(best_gt)
            else:
                FP += 1

        FN = len(gt_boxes) - TP

        accuracy = TP / (TP + FP + FN) if (TP + FP + FN) else 0
        precision = TP / (TP + FP) if (TP + FP) else 0
        recall = TP / (TP + FN) if (TP + FN) else 0

        print(f"Sample {index+1}:")
        print(f"  TP = {TP}, FP = {FP}, FN = {FN}")
        print(f"  Precision = {precision:.3f}")
        print(f"  Recall    = {recall:.3f}")
        print(f"  Accuracy  = {accuracy:.3f}")

    plt.show()

if __name__ == "__main__":
    main()