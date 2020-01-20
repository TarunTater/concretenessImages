#!/usr/bin/env python
# flake8: noqa

from lib.Feature import Feature, FeatureType
from lib.Imageset import DATASET_LOCATION
import cv2

class FeatureVisualConcepts(Feature):
    def __init__(self, type, image_set):
        import pickle
        import json
        with open('data/autotags.pickle', 'rb') as handle:
            self.autotags = pickle.load(handle)

        with open(DATASET_LOCATION + '/imageability_flickr_autotags-cur.json', 'r') as handle:
            self.dataset = json.load(handle)

        super().__init__(type, image_set)

    def process(self):
        return super().process(self.image_set[0], force=False)

    def extract(self, image, imagepath):
        fname = imagepath.split("/")[-1]
        try:
            tags = self.dataset[fname]
        except:
            tags = []

        resultvector = [0] * len(self.autotags)
        for tag in tags:
            index = self.autotags.index(tag)
            value = tags[tag]
            #print(tag, index, value)
            resultvector[index] = value

        return resultvector
