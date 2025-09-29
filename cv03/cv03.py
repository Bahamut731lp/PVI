"""
    Modul pro řešení úlohy 1 - viz zadání.pdf
"""
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.fft import dctn
import cv2
import numpy as np

def get_amplitude_spectrum(path: Path):
    image = cv2.imread(path.as_posix())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    fft2 = np.fft.fft2(grayscale)
    spectrum = np.abs(fft2)
    shifted_spectrum = np.fft.fftshift(spectrum)

    return image, spectrum, shifted_spectrum

def display_amplitude_spectrum(image, spectrum: np.ndarray, shifted_spectrum: np.ndarray):
    fig = plt.figure("Amplitudové spektrum obrázku")

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    
    plt.subplot(1, 3, 2)
    plt.imshow(np.log(spectrum), cmap='jet')
    plt.title('Amplitudové spektrum')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(np.log(shifted_spectrum), cmap='jet')
    plt.title('Amplitudové spektrum s posunutými kvadranty')
    plt.colorbar()

    return fig

def get_grayscale(filepath):
    image_data = cv2.imread(filepath)
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    grayscale = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
    histogram = cv2.calcHist([grayscale], [0], None, [256], [0, 256])
    histogram = cv2.normalize(histogram, histogram, norm_type=cv2.NORM_MINMAX)

    return {
        "histogram": histogram,
        "image": image_data
    }

def get_hue(filepath):
    image_data = cv2.imread(filepath)
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    hue = cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV)
    histogram = cv2.calcHist([hue], [0], None, [180], [0, 180])
    histogram = cv2.normalize(histogram, histogram, norm_type=cv2.NORM_MINMAX)

    return {
        "histogram": histogram,
        "image": image_data
    }

def get_dctn(filepath):
    image_data = cv2.imread(filepath)
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    dct = dctn(image_data)
    R = 5
    vector = dct[0:R, 0:R].flatten()
    
    return {
        "histogram": vector,
        "image": image_data
    }

def create_comparision_window(title: str, data):
    images = plt.figure(title)
    dimension = len(data)

    for image_index, _ in enumerate(data):    
        image_distances = [
            cv2.compareHist(
                data[image_index]["histogram"],
                data[x]["histogram"],
                cv2.HISTCMP_INTERSECT 
            ) for x, _ in enumerate(data)
        ]

        np_image_distances = np.array(image_distances)
        indices = np.argsort(np_image_distances)[::-1]

        for index in range(len(indices)):
            plt.figure(images)
            plt.subplot(dimension, dimension, (image_index * (dimension)) + index + 1)
            plt.imshow(data[indices[index]]["image"])

def create_comparision_window_2(title: str, data):
    images = plt.figure(title)
    dimension = len(data)

    for image_index, _ in enumerate(data):    
        image_distances = [
            np.linalg.norm(data[image_index]["histogram"] - data[x]["histogram"]) for x, _ in enumerate(data)
        ]

        np_image_distances = np.array(image_distances)
        indices = np.flip(np.argsort(np_image_distances)[::-1])

        for index in range(len(indices)):
            plt.figure(images)
            plt.subplot(dimension, dimension, (image_index * (dimension)) + index + 1)
            plt.imshow(data[indices[index]]["image"])


def main():
    folder = Path("./cv03")
    images = folder.glob("*.jpg")
    images = sorted(list(images))

    spectrum_input = Path("./cv03/pvi_cv03_im07.jpg")
    image, spectrum, shifted_spectrum = get_amplitude_spectrum(spectrum_input)
    display_amplitude_spectrum(image, spectrum, shifted_spectrum)

    hues = [get_hue(x) for x in images]
    grayscales = [get_grayscale(x) for x in images]
    dtcns = [get_dctn(x) for x in images]

    create_comparision_window("Features - Hist. Hue", hues)
    create_comparision_window("Features - Hist. Gray", grayscales)
    create_comparision_window_2("Features - DCT 5x5", dtcns)

    plt.show()

if __name__ == "__main__":
    main()