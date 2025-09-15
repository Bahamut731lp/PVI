# -*- coding: utf-8 -*-

# Import potrebnych modulu (knihoven):
# ------------------------------------

# Modul numpy obsahuje vsechno potrebne pro praci s poli a maticemi. Skladba
# prikazu (nazvy i parametry) velmi verne kopiruje Matlab.
import numpy as np

# Knihovna matplotlib se stara o vykreslovani grafu do oken ci souboru
# (podporuje # napr. export do pdf). Submodul pyplot pak nabizi syntaxi velmi
# podobnou jazyku Matlab.
import matplotlib.pyplot as plt

# OpenCV je velmi rozsirena C++ knihovna obsahujici nastroje pro zpracovani a
# rozpoznavani obrazu. Modul cv2 je wrapper umoznujici jeji pouziti primo
# v Pythonu.
import cv2

# Pro nacteni obrazku z disku pouzijeme knihovnu OpenCV. Obrazek bude nacten
# jako trojrozmerna matice o rozmerech (vyska, sirka, pocet kanalu). OpenCV
# nacte kanaly v poradi B, G, R, tj. bgr[:,:,0] bude modra slozka.

plt.close('all')

def preprocess(filepath):
    image_data = cv2.imread(filepath)
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    histogram = cv2.calcHist([image_data], [2], None, [256], [0, 256])

    return {
        "histogram": histogram,
        "image": image_data
    }

auto = preprocess("./cv01/cv01_auto.jpg")
jablko = preprocess("./cv01/cv01_jablko.jpg")
mesic = preprocess("./cv01/cv01_mesic.jpg")

categories = [auto, jablko, mesic]
images = ["./cv01/cv01_u01.jpg", "./cv01/cv01_u02.jpg", "./cv01/cv01_u03.jpg"]
images = [preprocess(x) for x in images]

rows = 4
cols = 3

plt.figure("Porovnání (obrázky)")
plt.subplot(rows, cols, 1)
plt.imshow(auto["image"])

plt.subplot(rows, cols, 2)
plt.imshow(jablko["image"])

plt.subplot(rows, cols, 3)
plt.imshow(mesic["image"])

plt.subplot(rows, cols, 4)
plt.plot(auto["histogram"])

plt.subplot(rows, cols, 5)
plt.plot(jablko["histogram"])

plt.subplot(rows, cols, 6)
plt.plot(mesic["histogram"])

for index, image in enumerate(images):
    distances = [cv2.compareHist(image["histogram"], x["histogram"], cv2.HISTCMP_CORREL ) for x in categories]
    np_image_distances = np.abs(np.array(distances))
    min_distance = np.argmax(np_image_distances)
    print(np_image_distances)
    print(min_distance)
    
    plt.subplot(rows, cols, 7 + min_distance)
    plt.plot(image["histogram"])

    plt.subplot(rows, cols, 10 + min_distance)
    plt.imshow(image["image"])

plt.show()