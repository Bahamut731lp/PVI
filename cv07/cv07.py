import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

def get_projection_vector(image):
    h_proj = image.sum(axis=1) # per-row
    v_proj = image.sum(axis=0) # per-column

    return np.concatenate([h_proj, v_proj])

def crop_zeros(image):
    ys, xs = np.where(image > 0)
    
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    
    cropped = image[y_min:y_max + 1, x_min:x_max + 1]
    return cropped

def get_iou_matrix(pred_boxes, gt_boxes):
    """
    pred_boxes: (N, 4)
    gt_boxes: (M, 4)
    Výstup: matice (N, M) s hodnotami IoU
    """
    # Převod [x, y, w, h] -> [x1, y1, x2, y2]
    pred_xy1 = pred_boxes[:, :2]
    pred_xy2 = pred_boxes[:, :2] + pred_boxes[:, 2:]
    gt_xy1 = gt_boxes[:, :2]
    gt_xy2 = gt_boxes[:, :2] + gt_boxes[:, 2:]

    # Broadcastování
    inter_xy1 = np.maximum(pred_xy1[:, None, :], gt_xy1[None, :, :])
    inter_xy2 = np.minimum(pred_xy2[:, None, :], gt_xy2[None, :, :])
    inter_wh = np.clip(inter_xy2 - inter_xy1, a_min=0, a_max=None)
    inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]

    area_pred = np.prod(pred_boxes[:, 2:], axis=1)
    area_gt = np.prod(gt_boxes[:, 2:], axis=1)

    union = area_pred[:, None] + area_gt[None, :] - inter_area
    iou = inter_area / np.clip(union, a_min=1e-10, a_max=None)
    return iou

def get_metrics(iou_matrix, iou_thresh=0.5):
    max_iou_per_pred = iou_matrix.max(axis=1)
    tp = np.sum(max_iou_per_pred >= iou_thresh)
    fp = np.sum(max_iou_per_pred < iou_thresh)
    
    # Pro každý GT box zjisti, jestli je detekován nějakou detekcí
    max_iou_per_gt = iou_matrix.max(axis=0)
    fn = np.sum(max_iou_per_gt < iou_thresh)
    
    accuracy = (tp ) / (tp + fp + fn) if tp + fp + fn > 0 else 0
    precision = tp / (tp + fn) if (tp + fn) > 0 else 0
    recall = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    return accuracy, precision, recall

def main():
    image_source = Path("cv07/pvi_cv07_text.bmp")
    image = cv2.imread(image_source.as_posix(), cv2.IMREAD_GRAYSCALE)

    characters = Path("cv07/znaky/").glob("*.png")
    characters = [(x.stem, cv2.imread(x.as_posix(), cv2.IMREAD_GRAYSCALE)) for x in characters]
    characters = [(x, get_projection_vector(np.abs(1 - (y / 255)))) for x, y in characters]
    characters = dict(characters)

    image = np.abs(1 - (image / 255))
    
    plt.figure("Segmentation")
    plt.subplot(3, 1, 1)
    plt.title("Original")
    plt.imshow(image, cmap="gray")
    plt.colorbar()

    keys = list(characters.keys())
    matrix = np.stack([characters[k] for k in keys])

    # normalize rows
    normalized = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    ret, markers = cv2.connectedComponents(np.uint8(image))
    plt.subplot(3, 1, 2)
    plt.title("Region Identification")
    plt.imshow(markers, cmap="jet")
    plt.colorbar()

    result = ""
    for i in range(1, ret):
        character = (markers == i).astype(np.uint8)
        character = crop_zeros(character)
        projection = get_projection_vector(character)
        vector = np.trim_zeros(projection)

        query_norm = vector / np.linalg.norm(vector)

        # compute cosine similarity for all at once
        scores = np.dot(normalized, query_norm)
        best_index = np.argmax(scores)
        best_key = keys[best_index]
        result += best_key

    plt.subplot(3, 1, 3)
    plt.text(0.01, 0.01, result)

    plt.figure("People")
    plt.subplot(1, 3, 1)
    plt.title("Original")
    people = cv2.imread("cv07/pvi_cv07_people.jpg", cv2.IMREAD_COLOR_RGB)
    plt.imshow(people)

    plt.subplot(1, 3, 2)
    plt.title("With Reference")
    boxes = []
    reference = people.copy()
    with open('cv07/pvi_cv07_boxes_01.txt') as f:
        lines = f.read().splitlines()
        for line in lines:
            vec = line.split(' ')
            vec = [int(x) for x in vec]
            boxes.append(vec)
    boxes = np.array(boxes)
    for (x, y, w, h) in boxes:
        cv2.rectangle(reference, (x, y), (x + w, y + h), (255, 0, 0), 2)
    plt.imshow(reference)

    plt.subplot(1, 3, 3)
    plt.title("Detected")
    faceCascade = cv2.CascadeClassifier("cv07/pvi_cv07_haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(cv2.cvtColor(people, cv2.COLOR_RGB2GRAY), scaleFactor=1.4, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    detected = reference.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(detected, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.imshow(detected)

    iou = get_iou_matrix(faces, boxes)
    acc, prec, rec = get_metrics(iou)

    print(f"Accuracy: {acc}, Precision: {prec}, Recall: {rec}")
    plt.show()

if __name__ == "__main__":
    main()