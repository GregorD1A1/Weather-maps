import cv2
from scipy.spatial import distance
import math


DARK_GREEN = (0, 100, 50)

def km_per_px_at_latitude(latitude):
    r_Earth = 6378
    px_per_10_degree_at_60n = 95
    px_per_10_degree_at_40n = 147
    # tworzymy funkcję do obliczania długości 10 stopni równoleżnika
    r_at_latitude = lambda x: r_Earth * math.sin(math.radians(90 - x))
    len_of_10deg_at_latitude = lambda x: 2 * math.pi * r_at_latitude(x) / 36
    # wyznaczamy funkcję odległości per ilośc pikseli
    km_per_px_60n = len_of_10deg_at_latitude(60) / px_per_10_degree_at_60n
    km_per_px_40n = len_of_10deg_at_latitude(40) / px_per_10_degree_at_40n
    km_per_px_by_latitude_fcn = build_linear_fcn(40, km_per_px_40n, 60, km_per_px_60n)

    return km_per_px_by_latitude_fcn(latitude)


def build_linear_fcn(x1, y1, x2, y2):
    a = (y2 - y1) / (x2 - x1)
    b = (y1 * x2 - y2 * x1) / (x2 - x1)
    return lambda x: a * x + b


class DistanceMeasurement():
    def __init__(self, image):
        self.first_click = True
        self.image = image
        self.latitude_per_y_fcn = build_linear_fcn(90, 40, 470, 60)

    def get_pos(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # pierwsze kliknięcie
            if self.first_click:
                self.first_click_x, self.first_click_y = x, y
                self.first_click = False
            # drugie kliknięcie
            else:
                distance_km = self.distance_measurement((self.first_click_x, self.first_click_y), (x, y))
                cv2.line(self.image, (self.first_click_x, self.first_click_y), (x, y), DARK_GREEN, 3)
                cv2.putText(self.image, f'{distance_km} km', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DARK_GREEN, 1)
                self.first_click = True
                cv2.imshow('pomiar odleglosci', self.image)

    def distance_measurement(self, point1, point2):
        _, y1 = point1
        _, y2 = point2
        distance_px = distance.euclidean(point1, point2)

        y_mid = min(y1, y2) + abs(y2 - y1) / 2
        latitude = self.latitude_per_y_fcn(y_mid)
        km_per_pix = km_per_px_at_latitude(latitude)
        distance_km = int(km_per_pix * distance_px)

        return distance_km


# program
# wczytywanie mapy
map_name = input('Wprowadź nazwę mapy do wczytania (z folderu "Laczone"):')
mapa = cv2.imread(f'Laczone/{map_name}')
measurer = DistanceMeasurement(mapa)

# okno pomiaru odległości
cv2.namedWindow('pomiar odleglosci')
cv2.setMouseCallback('pomiar odleglosci', measurer.get_pos)

# pokazuje mapę przed rysowaniem pierwszej kreski
cv2.imshow('pomiar odleglosci', measurer.image)
cv2.waitKey(0)
