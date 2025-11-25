import cv2
import numpy as np

x1, y1, x2, y2 = 0, 0, 0, 0

def zeroth_moment(back_projection):
    x, y = np.meshgrid(np.arange(back_projection.shape[1]), np.arange(back_projection.shape[0]))

    x_t = np.sum(x * back_projection) / np.sum(back_projection)
    y_t = np.sum(y * back_projection) / np.sum(back_projection)

    return x_t, y_t

def first_frame(bgr, roi_hist):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_RGB2HSV) 
    hue = hsv[:,:,0]
    back_projection = roi_hist[hue]
    x_t, y_t = zeroth_moment(back_projection)

    return x_t, y_t

def other_frames(bgr, prev_xy, roi_hist):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_RGB2HSV) 
    hue = hsv[:,:,0]
    back_projection = roi_hist[hue]
    back_projection = back_projection[prev_xy[1]:prev_xy[3], prev_xy[0]:prev_xy[2]]
    
    tmp_x_t, tmp_y_t = zeroth_moment(back_projection)
    
    x_t = int(prev_xy[0] + tmp_x_t)
    y_t = int(prev_xy[1] + tmp_y_t)
    return x_t, y_t

def camshift(image, frames):
    y1, x1, _ = np.floor_divide(image.shape, 2)
    y2, x2, _ = np.floor_divide(image.shape, 2)
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue = frame[:,:,0]

    roi_hist, _ = np.histogram(hue, 180, (0, 180))
    roi_hist = roi_hist / np.max(roi_hist)
    is_first = True
    result = []

    for frame in frames:
        if is_first:
            x_t, y_t = first_frame(frame, roi_hist)
            is_first = False
        else:
            x_t, y_t = other_frames(frame, prev_xy, roi_hist)

        # draw a rectangle around the center of mass  based on x1, y1, x2, y2
        x1_t = abs(int(x_t - x1))
        y1_t = abs(int(y_t - y1))
        x2_t = abs(int(x_t + x2))
        y2_t = abs(int(y_t + y2))
        prev_xy = (x1_t, y1_t, x2_t, y2_t)
        cv2.rectangle(frame, (x1_t, y1_t), (x2_t, y2_t), (0, 255, 0), 2)

        result.append(frame)

    return result
