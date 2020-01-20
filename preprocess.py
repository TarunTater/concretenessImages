#!/usr/bin/env python3
# -*- coding: utf-8 -*-

######################################################################

# dataset location
prefix = "/work/kastnerm/test"

# list of words
# script expects a subfolder prefix/ for each word with num_images number of images
words = ["a", "b"]

# number of images for each word
num_images = 1000

# feature selection
# select which features to preprocess
from lib.Feature import FeatureType
features = [FeatureType.ColorHSV]

# all features needed for pretrained models: 
# features = [FeatureType.ColorHSV, FeatureType.SURF, FeatureType.GIST, FeatureType.YOLO_NUM_9000, FeatureType.YOLO_COMPOSITION]

# FeatureType.ColorHSV -> anywhere ok, no special dependencies
# FeatureType.SURF -> sakura? no special dependencies
# FeatureType.GIST -> sakura? lear_gist_python needed
# FeatureType.YOLO_NUM_9000, FeatureType.YOLO_COMPOSITION -> kon? pydarknet needed


######################################################################

# get visual features for each word
from lib.create_matrix import processLooseWord

import logging
logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                datefmt='%m/%d/%Y %I:%M:%S %p',
                level=logging.DEBUG
                )

for word in words:
    # get visual features for each term
    # function will try to load as much as possible from cache/
    # or calculate them newly if not found
    location = prefix + "/" + word + "/"
    logging.info("start " + word)
    _x = processLooseWord(word, location, features, num_images)
    logging.info("finished " + word)


