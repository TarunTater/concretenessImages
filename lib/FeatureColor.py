#!/usr/bin/env python
# flake8: noqa

from lib.Feature import Feature, FeatureType
import cv2


class FeatureColor(Feature):
    def __init__(self, type, image_set):
        super().__init__(type, image_set)
        self.size = 10000
        self.img_size = 512

    def process(self):
        return super().process(self.image_set[0], force=False)

    def extract(self, image, imagepath):
        resized = cv2.resize(image, (self.img_size, self.img_size))

        if self.type == FeatureType.ColorHSV:
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            channels = cv2.split(hsv)
        else:
            channels = cv2.split(resized)

        hist = []
        if self.type == FeatureType.ColorHSV:
            hist_h = cv2.calcHist([channels[0]], [0], None, [36], [0, 180])
            hist_s = cv2.calcHist([channels[1]], [0], None, [36], [0, 256])

            hist.extend(hist_h.flatten())
            hist.extend(hist_s.flatten())
        else:
            hist_r = cv2.calcHist([channels[0]], [0], None, [36], [0, 256])
            hist_g = cv2.calcHist([channels[1]], [0], None, [36], [0, 256])
            hist_b = cv2.calcHist([channels[2]], [0], None, [36], [0, 256])

            hist.extend(hist_r.flatten())
            hist.extend(hist_g.flatten())
            hist.extend(hist_b.flatten())

        return hist
