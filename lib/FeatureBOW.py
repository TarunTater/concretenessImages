#!/usr/bin/env python
# flake8: noqa

import cv2
import logging
import numpy as np

from lib.Feature import *
from lib.Imageset import *
#from memory_profiler import profile

class FeatureBOW(Feature):
    @property
    def filename(self):
        if self.testing_hash is None:
            return self.directory + self.hash + "_results.npy"
        return self.directory + self.testing_hash + "_results.npy"

    def __init__(self, type, image_set, trained_hash, testing_hash, trained_directory):
        self.testing_hash = testing_hash
        self.trained_directory = trained_directory

        if trained_hash is not None:
            self._hash = trained_hash + "_" + image_set[0].hash

        super().__init__(type, image_set)
        self.size = 0.01
        self.bowsize = 4096

        if self.type is FeatureType.FREAK:
            self.size = 0.001

        matcher = self.initMatcher(type)
        self.initFeature(type)

        self.dictionary = None
        self.bow_extract = cv2.BOWImgDescriptorExtractor(self.extractor, matcher)
        self.bow_train = cv2.BOWKMeansTrainer(self.bowsize)

        #print(testing_hash)
        try:
            if(self.trained_directory is not None): 
                dictfile = "cache/" + str(self.name) + "/" + self.trained_directory + "/" + trained_hash + "_dict.npy"
            else:
                dictfile = self.directory + self.hash + "_dict.npy"
            logging.debug(dictfile)
            self.dictionary = np.load(dictfile)
            #logging.debug("Loaded dict cache.")

        except FileNotFoundError:
            logging.debug("Could not load dict cache file for this feature.")

        #print(self.hash)

    def initMatcher(self, type):
        if type == FeatureType.SURF:
            FLANN_INDEX_KDTREE = 1
            flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
            search_params = dict()
            matcher = cv2.FlannBasedMatcher(flann_params, search_params)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        return matcher

    def initFeature(self, type):
        self.surfsize = 800

        if type == FeatureType.SURF:
            self.extractor = cv2.xfeatures2d.SURF_create(self.surfsize)
            self.detector = cv2.xfeatures2d.SURF_create(self.surfsize)
        elif type == FeatureType.ORB:
            self.extractor = cv2.ORB_create()
            self.detector = cv2.ORB_create()
        elif type == FeatureType.FREAK:
            self.detector = cv2.xfeatures2d.SURF_create(self.surfsize)
            self.extractor = cv2.xfeatures2d.FREAK_create()
        else:
            raise NotImplementedError()

    def process(self):
        if self.dictionary is None:
            for s in self.image_set:
                self.train_bow(s)

            logging.info("cluster vocabulary")
            try:
                self.dictionary = self.bow_train.cluster()
                np.save(self.directory + self.hash + "_dict.npy", self.dictionary)
            except Exception as e:
                logging.error(e)

        if self.type is not FeatureType.SURF:
            self.dictionary = self.dictionary.astype(np.uint8)

        try:
            self.bow_extract.setVocabulary(self.dictionary)
        except Exception as e:
            logging.error(e)

        # print("bow vocab", np.shape(dictionary), dictionary)

        if len(self.results) == 0:
            for s in self.image_set:
                super().process(s, force=True, save=False)

        # NODO force hack
        #for s in self.image_set:
        #    super().process(s, force=True, save=False)

        self.save()

    def train_bow(self, im_set):
        # training BOW
        logging.info("%s %s %s %s %s %s %s",
                      "train bow", self.name,
                      #"for", im_set.folder,
                      "[", im_set.start,
                      ",", im_set.end,
                      "]")

        import progressbar
        print(f"====> im_set.images in FeatureBOW = {len(im_set.images)}")
        bar = progressbar.ProgressBar(max_value=len(im_set.images))
        counter = 0
        bar.update(counter)
        from lib.Imageset import get_real_filename

        for file in im_set.images:
            try:
                file = get_real_filename(file)
                #print(file)
                img = cv2.imread(file, 0)
            except Exception as e:
                logging.error(f"Unexpected FeatureBOW: {str(e)}")
                counter += 1
                bar.update(counter)
                continue
            key = self.detector.detect(img)
            #des = freak_extract.compute(im, key)[1]
            des = self.extractor.compute(img, key)[1]

            # necessary for non-SURF descriptors, as they can
            # also detect nothing, resulting in NoneType

            if des is not None:
                if des.size > 0:
                    #if self.type is not FeatureType.SURF:
                    #    des = des.astype(np.float32)
                    self.bow_train.add(des)


            counter += 1
            bar.update(counter)
            del img
            del des
            del key

            #import gc  
            #gc.collect()

    def extractKeypointsBow(self, imagepath):
        image = cv2.imread(imagepath, 0)
        # TODO: fix for unloadable images
        if image is None:
            logging.info("Use empty array")
            return np.zeros([1, self.bowsize])

        det = self.detector.detect(image)

        data = []

        #print(len(det))

        for d in det:
            datapoint = self.bow_extract.compute(image, [d])
            data.append(datapoint)

        return data

    def extractKeypoints(self, imagepath):
        image = cv2.imread(imagepath, 0)
        if image is None:
            logging.info("Use empty array")
            return np.zeros([1, self.bowsize])

        det = self.detector.detect(image)

        return det 


    def extract(self, image, imagepath):
        # TODO: fix for unloadable images
        if image is None:
            logging.info("Use empty array")
            return np.zeros([1, self.bowsize])

        det = self.detector.detect(image)
        data = self.bow_extract.compute(image, det)

        # binary descriptors like ORB, FREAK can have "none" result
        if data is None:
            return np.zeros([1, self.bowsize])

        # print(data.shape)

        # print(data)
        return data
