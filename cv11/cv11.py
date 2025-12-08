import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def load_and_preprocess(paths, size=None, to_gray=True):
    imgs = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Can't read image: {p}")
        if to_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if size is not None:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        imgs.append(img.astype(np.float32))
    return imgs

def images_to_column_matrix(imgs):
    """
    Converts images into array of FFT2 features - each column in separate image
    """
    H, W = imgs[0].shape
    P = H * W
    n = len(imgs)
    X = np.zeros((P, n), dtype=np.complex128)
    for i, img in enumerate(imgs):
        F = np.fft.fft2(img)
        X[:, i] = F.flatten()
    
    return X, (H, W)

def train_mace(X, eps=1e-6):
    """
    X: (P, n) matrix where columns are flattened FFTs of training images (complex)
    returns H (P,) complex frequency-domain MACE filter and D_vec
    """
    P, n = X.shape

    # D: Hodnoty spočítané jako průměr výkonového spektra z řádků matice X (diagonální matice)
    D_vec = np.mean(np.abs(X) ** 2, axis=1)  # Vrací jako (P, 1)
    D_vec = D_vec + eps # Přičtení malé hodnoty, ať nedělíme nulou
    Dinv = 1.0 / D_vec
    # Dinv[:, None] přidá osu a udělá z toho sloupcový vektor
    # Každý řádek matice X vynásobíme číslem z Dinv.
    Y = Dinv[:, None] * X
    # Komplexně sdruženou matici, kterou transponujeme, maticově vynásobíme
    A = X.conj().T @ Y

    # Matice korelací, a respektive peaků
    u = np.ones((n,), dtype=np.complex128)
    reg = 1e-8 * np.eye(n) # np.eye() vytváří jednotkovou matici
    a = np.linalg.solve(A + reg, u)  # shape (n,)
    Ha = X @ a  # final filter H = D^{-1} * (X @ a)   -> shape (P,)
    H = Dinv * Ha  # element-wise multiply -> (P,)
    H = np.asarray(H, dtype=np.complex128).reshape(-1)

    return H, D_vec


def apply_filter(H, D_vec, img, shape_hw):
    """
    H: (P,) complex filter (frequency domain)
    D_vec: PSD (for diagnostics if needed)
    img: 2D array image (same shape as shape_hw)
    shape_hw: (H,W)
    returns:
      corr: 2D real correlation surface (ifft of H*F_img)
      peak_value: maximum of corr (real)
      p2s: peak-to-sidelobe ratio (peak / mean of rest)
    """
    H = H.reshape(-1)
    Hf = H.reshape(shape_hw)
    F = np.fft.fft2(img)
    corr_freq = np.conj(Hf) * F
    corr = np.fft.ifft2(corr_freq)
    corr_real = np.real(corr)
    # shift so that zero lag is centered (optional)
    corr_shift = np.fft.fftshift(corr_real)

    peak = np.max(corr_shift)
    # compute sidelobe mean excluding small neighborhood around peak
    peak_idx = np.unravel_index(np.argmax(corr_shift), corr_shift.shape)
    ps = corr_shift.copy()
    # zero out a small 5x5 window around peak to compute sidelobe stats
    hh, ww = corr_shift.shape
    r = 2
    y0, x0 = peak_idx
    y1, y2 = max(0, y0 - r), min(hh, y0 + r + 1)
    x1, x2 = max(0, x0 - r), min(ww, x0 + r + 1)
    ps[y1:y2, x1:x2] = 0
    sidelobe_mean = np.mean(np.abs(ps))
    p2s = peak / (sidelobe_mean + 1e-12)
    return corr_shift, float(peak), float(p2s)

if __name__ == "__main__":
    class_1 = [
        "./cv11/p11.bmp",
        "./cv11/p12.bmp",
        "./cv11/p13.bmp"
    ]

    class_2 = [
        "./cv11/p21.bmp",
        "./cv11/p22.bmp",
        "./cv11/p23.bmp",
    ]

    class_3 = [
        "./cv11/p31.bmp",
        "./cv11/p32.bmp",
        "./cv11/p33.bmp"
    ]

    unknown = ["./cv11/unknown.bmp"]

    # load and preprocess (convert to grayscale and resize to same size)
    # pick target size from first training image
    example = cv2.imread(class_1[0], cv2.IMREAD_COLOR)
    if example is None:
        raise FileNotFoundError(f"Can't read example image {class_1[0]}")
    example = cv2.cvtColor(example, cv2.COLOR_BGR2GRAY)
    target_size = (64, 64)

    imgs1 = load_and_preprocess(class_1, size=target_size, to_gray=True)
    imgs2 = load_and_preprocess(class_2, size=target_size, to_gray=True)
    imgs3 = load_and_preprocess(class_3, size=target_size, to_gray=True)
    unk_imgs = load_and_preprocess(unknown, size=target_size, to_gray=True)

    # form frequency domain matrices
    X1, hw = images_to_column_matrix(imgs1)
    X2, _ = images_to_column_matrix(imgs2)
    X3, _ = images_to_column_matrix(imgs3)

    # train filters for each class
    H1, D1 = train_mace(X1)
    H2, D2 = train_mace(X2)
    H3, D3 = train_mace(X3)

    # evaluate unknowns
    for i, uimg in enumerate(unk_imgs):
        corr1, peak1, p2s1 = apply_filter(H1, D1, uimg, hw)
        corr2, peak2, p2s2 = apply_filter(H2, D2, uimg, hw)
        corr3, peak3, p2s3 = apply_filter(H3, D3, uimg, hw)

        print(f"Unknown image #{i}: peaks = [class1={peak1:.3f}, class2={peak2:.3f}, class3={peak3:.3f}]")
        print(f"Peak-to-sidelobe = [c1={p2s1:.3f}, c2={p2s2:.3f}, c3={p2s3:.3f}]")

        # choose class by highest peak (or highest P2S)
        peaks = np.array([peak1, peak2, peak3])
        p2s = np.array([p2s1, p2s2, p2s3])
        chosen_by_peak = np.argmax(peaks) + 1
        chosen_by_p2s = np.argmax(p2s) + 1
        print(f"Decision (by peak): class_{chosen_by_peak}")
        print(f"Decision (by P2S): class_{chosen_by_p2s}")

        # optionally: save correlation surfaces for inspection
        cv2.normalize(corr1, corr1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(corr2, corr2, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(corr3, corr3, 0, 1, cv2.NORM_MINMAX)

        plt.figure()
        shape = (2, 3)
        plt.subplot(*shape, 1)
        plt.imshow(corr1, cmap="jet")
        plt.colorbar()
        plt.subplot(*shape, 2)
        plt.imshow(corr2, cmap="jet")
        plt.colorbar()
        plt.subplot(*shape, 3)
        plt.imshow(corr3, cmap="jet")
        plt.colorbar()

        plt.subplot(*shape, 4)
        plt.imshow(unk_imgs[0], cmap="gray")
        plt.title("Unknown")

        plt.subplot(*shape, 5)
        plt.title("Class")
        if chosen_by_p2s == 1:
            plt.imshow(imgs1[0], cmap="gray")
        elif chosen_by_p2s == 2:
            plt.imshow(imgs2[0], cmap="gray")
        elif chosen_by_p2s == 3:
            plt.imshow(imgs3[0], cmap="gray")

        plt.show()