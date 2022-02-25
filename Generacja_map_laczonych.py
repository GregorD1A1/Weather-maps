import os
import cv2
from scipy import ndimage
from PIL import Image
import numpy as np
from tqdm import tqdm


MAP_WIDTH = 740
MAP_HEIGHT = 665
YELLOW = (0, 255, 255)
DEEP_PURPLE = (90, 60, 90)
SPRING_GREEN = (150, 255, 0)
PURPLE = (240, 32, 160)
BLACK = (0, 0, 0)


def map_transform(map, rot_angle, zoom_x, zoom_y, x, y):
    # obrót
    map = ndimage.rotate(map, rot_angle)
    # skalowanie
    map = ndimage.zoom(map, (zoom_y, zoom_x, 1))
    # wycinanie obszaru
    map = map[y: y+MAP_HEIGHT, x: x+MAP_WIDTH]
    return map


def weather_fronts_extraction(map, brightnes_threshold = 120, min_contour_vertices_nr = 50):
    # kowetujemy na jednokanałowy (szary) obraz
    map = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
    # nakładamy rozmycie
    map = cv2.GaussianBlur(map, (15, 15), 1)
    # Kolorujemy piksele o jasności poniżej wartości progowej na biało, resztę na czarno
    map = (map <= brightnes_threshold).astype(np.uint8) * 255
    # pogrubiamy kreski, by zalać przerwy między nimi (dłuższe kontury) i zmniejszyć ilość punktów narożnych w szumach
    map = cv2.dilate(map, np.array([15, 15]))
    # znajdujemy obrysy obiektów
    contours, hierarchy = cv2.findContours(map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # filtracja obrysów o ilości punktów większej niż wartość progowa:
    contours = [contour for contour in contours if contour.shape[0] > min_contour_vertices_nr]
    # pocieniamy z powrotem
    map = cv2.erode(map, np.array([15, 15]))
    # tworzymy maskę
    mask = np.zeros_like(map)
    mask = cv2.fillPoly(mask, contours, 1)





    
    map = map * mask

    return map


def imgw_unify_colors(map):
    # zmiana czerwonych linii na ciemnofioletowe
    almost_red = (0, 0, 200)
    red = (100, 100, 255)
    mask = cv2.inRange(map, almost_red, red)
    map[mask == 255] = DEEP_PURPLE
    # zmiana fioletowych linii na ciemnofioletowe
    mask = cv2.inRange(map, PURPLE, PURPLE)
    map[mask == 255] = DEEP_PURPLE
    # zmiana niebieskich linii na ciemnofioletowe
    almost_blue = (200, 0, 0)
    blue = (255, 100, 100)
    mask = cv2.inRange(map, almost_blue, blue)
    map[mask == 255] = DEEP_PURPLE
    almost_pink = (200, 0, 200)
    pink = (255, 100, 255)
    mask = cv2.inRange(map, almost_pink, pink)
    map[mask == 255] = DEEP_PURPLE
    return map


def combine_maps(mapa_imgw, mapa_dwd, mapa_ukmet):
    # kolorujemy wszystkie fronty z imgw na niebiesko dla lepszej czytelności
    map = imgw_unify_colors(mapa_imgw)
    # dodajemy fornty z dwd i ukmet w odpowiednich kolorach
    map[mapa_dwd == 255] = YELLOW
    map[mapa_ukmet == 255] = SPRING_GREEN

    return map


def put_description(map):
    cv2.rectangle(map, (0, 610), (480, 665), (255, 255, 255), -1)
    cv2.rectangle(map, (10, 625), (50, 650), DEEP_PURPLE, -1)
    cv2.rectangle(map, (10, 625), (50, 650), BLACK, 1)
    cv2.putText(map, 'IMGW', (60, 645), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 2)
    cv2.rectangle(map, (180, 625), (220, 650), SPRING_GREEN, -1)
    cv2.rectangle(map, (180, 625), (220, 650), BLACK, 1)
    cv2.putText(map, 'UKMET', (230, 645), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 2)
    cv2.rectangle(map, (350, 625), (390, 650), YELLOW, -1)
    cv2.rectangle(map, (350, 625), (390, 650), BLACK, 1)
    cv2.putText(map, 'DWD', (400, 645), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 2)


def map_loading(folder_name):
    # listowanie nazw map
    mapa_dwd_name, mapa_ukmet_name, mapa_imgw_name = os.listdir(f'Dane/{folder_name}')
    # wczytywanie obrazów map
    mapa_imgw = cv2.imread(f'Dane/{folder_name}/{mapa_imgw_name}')
    mapa_ukmet = cv2.imread(f'Dane/{folder_name}/{mapa_ukmet_name}')
    mapa_dwd = cv2.imread(f'Dane/{folder_name}/{mapa_dwd_name}')
    date = mapa_dwd_name[:8]
    return mapa_imgw, mapa_ukmet, mapa_dwd, date

# program główny
print('Obrabiam mapy. To może trochę potrwać...')
for folder in tqdm(os.listdir('Dane')):
    # ładowanie map
    mapa_imgw, mapa_ukmet, mapa_dwd, date = map_loading(folder)

    # sprawdzanie wymiarów mapy imgw
    if  mapa_imgw.shape[0] != MAP_HEIGHT or mapa_imgw.shape[1] != MAP_WIDTH:
        mapa_imgw = cv2.resize(mapa_imgw, (MAP_WIDTH, MAP_HEIGHT))

    # transformacje map, by nakładały się na siebie
    mapa_ukmet = map_transform(mapa_ukmet, rot_angle=-45, zoom_x=1.15, zoom_y=1.2, x=520, y=515)
    mapa_dwd = map_transform(mapa_dwd, rot_angle=1, zoom_x=1.2, zoom_y=1.15, x=470, y=150)

    # znajdujemy fronty atmosferyczne na mapach
    mapa_dwd = weather_fronts_extraction(mapa_dwd, 120, 50)
    mapa_ukmet = weather_fronts_extraction(mapa_ukmet, 120, 100)

    # łaczymy mapy
    mapa_wspolna = combine_maps(mapa_imgw, mapa_dwd, mapa_ukmet)

    # dodajemy opis
    put_description(mapa_wspolna)

    # zapisujemy mapę łączoną
    cv2.imwrite(f'Laczone/l_{date}.png', mapa_wspolna)
