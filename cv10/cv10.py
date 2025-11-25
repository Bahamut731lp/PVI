import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from camshift import camshift

CUT_THRESHOLD = 70
BG_THRESHOLD = 0

def get_video_frames(path: str):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, bgr = cap.read()

        if not ret:
            break

        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        
    cap.release()

    return frames


def get_video_cuts(frames):
    differences = []
    latest_pixel_sum = None

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        current_pixel_sum = cv2.calcHist([gray], [0], None, [256], [0, 255])

        if latest_pixel_sum is None:
            latest_pixel_sum = current_pixel_sum
            continue

        difference = np.sum(np.abs(current_pixel_sum - latest_pixel_sum))
        differences.append(difference)

        latest_pixel_sum = current_pixel_sum

    differences = np.array(differences, dtype=np.float32)
    differences *= 100 / differences.max()
    cuts = np.concatenate([
        [0],
        np.where(differences > CUT_THRESHOLD)[0],
        [len(differences)]
    ])

    return cuts

def get_keyframes(frames, cuts):
    indicies = []

    for index in range(0, len(cuts) - 1):
        indicies.append(np.floor((cuts[index] + cuts[index + 1]) / 2))

    indicies = np.int64(indicies)
    return np.array(frames)[indicies]

def background_subtraction(video: str, start: int, end: int, threshold: int = 10, min_area: int = 150):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise ValueError("Cannot open video file")

    frames = []
    index = 0

    # --- Load frames in the desired interval ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if index < start:
            index += 1
            continue

        if index > end:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frames.append(gray)
        index += 1

    cap.release()

    if not frames:
        return None

    reference = frames[0].astype(np.int16)
    segmented = []
    all_boxes = []

    for frame in frames:
        f = frame.astype(np.int16)
        # Absolute difference = changes between frame and background
        diff = np.abs(f - reference)
        # Foreground mask based on threshold
        mask = (diff > threshold).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=5)

        segmented.append(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue  # ignore tiny objects (outliers)

            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))

        all_boxes.append(boxes)

    return all_boxes

def get_biggest_boxes(all_boxes):
    biggest = []
    for boxes in all_boxes:
        if not boxes:
            biggest.append(None)
            continue

        # select box with max area
        biggest_box = max(boxes, key=lambda b: b[2] * b[3])
        biggest.append(biggest_box)

    return biggest

def play_frames(frames_list, interval=30):
    """
    frames_list = [video1_frames, video2_frames, ...]
    Works even if their lengths differ.
    """

    n = len(frames_list)
    max_len = max(len(f) for f in frames_list)

    fig, axes = plt.subplots(1, n)

    if n == 1:
        axes = [axes]

    # create image objects
    imgs = []
    for ax, frames in zip(axes, frames_list):
        img = ax.imshow(frames[0], cmap='gray', vmin=0, vmax=255)
        ax.axis('off')
        imgs.append(img)

    def update(i):
        for img, frames in zip(imgs, frames_list):
            # if i exceeds video length → repeat last frame
            frame = frames[i] if i < len(frames) else frames[-1]
            img.set_data(frame)
        return imgs

    ani = FuncAnimation(
        fig,
        update,
        frames=max_len,
        interval=interval,
        blit=False
    )

    plt.show()

if __name__ == "__main__":
    video = 'cv10/pvi_cv10_video_in.mp4'
    frames = get_video_frames(video)
    cuts = get_video_cuts(frames)
    keyframes = get_keyframes(frames, cuts)

    print("Cuts", cuts)

    # Show keyframes
    number_of_keyframes = len(keyframes)
    keyframes_figure, ax_1 = plt.subplots(1, number_of_keyframes, figsize=(number_of_keyframes * 4, number_of_keyframes + 1))
    keyframes_figure.canvas.manager.set_window_title('Keyframes')
    keyframes_figure.tight_layout()

    for index, keyframe in enumerate(keyframes):
        ax_1[index].imshow(keyframe)
        ax_1[index].set_title(f"Keyframe {index + 1}")

    # First scene
    boxes = background_subtraction(video, cuts[0], cuts[1], 30, 150)
    boxes = get_biggest_boxes(boxes)

    first_scene = frames[cuts[0]:cuts[1]].copy()
    for index, frame in enumerate(first_scene):
        if boxes[index] is None:
            continue

        x, y, w, h = boxes[index]
        first_scene[index] = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Second Scene
    second_scene = camshift(
        cv2.imread("cv10/pvi_cv10_vzor_pomeranc.bmp", cv2.IMREAD_COLOR_RGB),
        frames[cuts[1] + 1:cuts[2]]
    )

    # Third scene
    boxes = background_subtraction(video, cuts[2]+1, cuts[3], 30, 150)
    boxes = get_biggest_boxes(boxes)
    third_scene = frames[cuts[2]+1:cuts[3]].copy()
    for index, frame in enumerate(third_scene):
        if boxes[index] is None:
            continue

        x, y, w, h = boxes[index]
        third_scene[index] = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    play_frames([first_scene, second_scene, third_scene])