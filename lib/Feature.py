#!/usr/bin/env python
# flake8: noqa

from abc import *
from enum import IntEnum
from multiprocessing.dummy import Pool as ThreadPool

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

import numpy as np
import logging
import hashlib
import cv2
import time
import progressbar

from lib.Imageset import Imageset
from lib.Imageset import get_real_filename

from numpy.linalg import norm
from scipy.stats import entropy


def jsd(P, Q):
    """Calculates Jensen-Shannon Divergence"""
    P = np.array(P)
    Q = np.array(Q)

    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))


class FeatureType(IntEnum):
    # Color histogram based
    ColorHSV = 0,
    ColorRGB = 1,

    # Pattern matrix based
    GIST = 2,
    GLCM = 3, # old merged together

    # BoW based
    SURF = 4,
    ORB = 5,
    FREAK = 6,

    # Deep learning-based
    CNN = 7,

    # GLCM split
    GLCM_CONTR = 20,
    GLCM_DISSIM = 21,
    GLCM_HOMOGEN = 22,
    GLCM_ENERGY = 23,
    GLCM_ASM = 24,
    GLCM_CORREL = 25,

    # High level
    VisualConcepts = 50,

    YOLO_NUM = 60,
    YOLO_NUM_9000 = 61,
    YOLO_COMPOSITION = 62,



def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        logging.debug('%s function took %0.3f ms' % (f.__name__, (time2 - time1) * 1000.0))
        return ret

    return wrap


class Feature(metaclass=ABCMeta):
    @staticmethod
    def factory(type: FeatureType, im_set: Imageset = None, trained_hash = None, testing_hash = None, trained_directory = None):
        import lib.FeatureColor
        #import lib.FeatureGIST
        #import lib.FeatureGLCM
        import lib.FeatureBOW
        #import lib.FeatureCNN
        import lib.FeatureVisualConcepts
        import lib.FeatureYOLO

        if type == FeatureType.ColorRGB or type == FeatureType.ColorHSV:
            return lib.FeatureColor.FeatureColor(type, im_set)
        elif type == FeatureType.GIST:
            import lib.FeatureGIST
            return lib.FeatureGIST.FeatureGIST(type, im_set)
        #elif type == FeatureType.GLCM or type >= FeatureType.GLCM_CONTR:
        #    return FeatureGLCM.FeatureGLCM(type, im_set)
        elif type == FeatureType.SURF:
            return lib.FeatureBOW.FeatureBOW(type, im_set, trained_hash, testing_hash, trained_directory)
        elif type == FeatureType.ORB:
            return lib.FeatureBOW.FeatureBOW(type, im_set)
        elif type == FeatureType.FREAK:
            return lib.FeatureBOW.FeatureBOW(type, im_set)
        #elif type == FeatureType.CNN:
        #    return FeatureCNN.FeatureCNN(type, im_set)
        elif type == FeatureType.VisualConcepts:
            return lib.FeatureVisualConcepts.FeatureVisualConcepts(type, im_set)
        elif type == FeatureType.YOLO_NUM or type == FeatureType.YOLO_NUM_9000 or type == FeatureType.YOLO_COMPOSITION:
            return lib.FeatureYOLO.FeatureYOLO(type, im_set)

    def __init__(self, type: FeatureType, image_set: Imageset, load=True):
        self.image_set = []
        self.image_set.extend(image_set)
        self.results = []
        self.type = type
        self.size = 1

        #logging.debug(self.info)

        # load cache
        if load:
            self.load()


    def compare(self, set1, set2):
        return jsd(set1, set2)


    @property
    def info(self):
        setinfo = []
        for s in self.image_set:
            setinfo.append(s.info)

        return (str(self.type), setinfo, self.hash)

    @property
    def directory(self):
        import os
        directory = "cache/" + str(self.name) + "/"

        for s in self.image_set:
            directory += str(s.name) + "_"
        directory = directory[:-1] + "/"

        if not os.path.exists(directory):
            os.makedirs(directory)

        return directory

    @property
    def name(self):
        return self.type

    @property
    def coefficient(self):
        return self.size

    @property
    def hash(self):
        try:
            if(self._hash):
                return self._hash
        except:
            pass

        m = hashlib.sha224()
        m.update(repr(self.type).encode("utf-8"))
        for s in self.image_set:
            m.update(s.hash.encode("utf-8"))

        logging.debug("feature " + str(m.hexdigest()))

        return m.hexdigest()

    @property
    def filename(self):
        return self.directory + self.hash + "_results.npy"

    def save(self):
        np.save(self.filename, self.results)

        with open(self.directory + "log.txt", "a") as myfile:
            myfile.write(str(len(self.results)) + " records in hash " + self.hash)

    def load(self):
        try:
            logging.debug(self.filename)
            self.results = np.load(self.filename)
        except FileNotFoundError:
            logging.debug("Could not load cache file for this feature.")

    def clear_data(self):
        self.results = []

    # @timing
    def extract_file(self, filen):
        logging.debug("process file %s", filen)
        img = cv2.imread(filen, 1)
        if img is None:
            logging.error("Could not open " + str(filen))
            #return []
        return self.extract(img, filen)

    @abstractmethod
    def process(self, image_set, force=False, save=True):
        # TODO: force does not really work because after loading,
        # self.results is an ndarray which has no append() etc
        if force == False:
            if self.results is not None and len(self.results) > 0:
                # return
                # print("limit ", len(self.results))
                # print("limit ", image_set.number)
                if len(self.results) >= len(image_set.images) - 1:
                    #logging.debug("Already calculated, skip.")
                    return
        else:
            self.clear_data()

        #logging.info("%s %s %s %s %s %s %s",
        #              "process", self.name,
        #              "[", image_set.start,
        #              ",", image_set.end,
        #              "]")

        bar = progressbar.ProgressBar(max_value=len(image_set.images))
        counter = 0
        bar.update(counter)
        #from tqdm import tqdm
        for ifile in image_set.images:#, desc="feature extraction", leave=True, mininterval=0):
            ifile = get_real_filename(ifile)
            img = cv2.imread(ifile)
#            print("file", ifile)
            file_results = self.extract(img, ifile)

            self.results.append(file_results)
            counter += 1
            bar.update(counter)

        if save:
            self.save()

        self.load()

    @abstractmethod
    def extract(self, image):
        pass
