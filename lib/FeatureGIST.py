#!/usr/bin/env python
import gist # https://github.com/tuttieee/lear-gist-python
import cv2

from lib.Feature import Feature


class FeatureGIST(Feature):
    def __init__(self, type, image_set):
        super().__init__(type, image_set)
        self.size = 1

    def process(self):
        return super().process(self.image_set[0], force=False)

    def extract(self, image, imagepath):
        resized = cv2.resize(image, (256, 256))
        descriptor = gist.extract(resized)

        return descriptor
