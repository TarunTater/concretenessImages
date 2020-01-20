#!/usr/bin/env python3
# -*- coding: utf-8 -*-

######################################################################

# number of images for each word
num_pics = 5000

# feature selection
# select which features to for training
from lib.Feature import Feature, FeatureType
features = [FeatureType.YOLO_NUM_9000, FeatureType.YOLO_COMPOSITION, FeatureType.SURF, FeatureType.ColorHSV, FeatureType.GIST]
#features = [FeatureType.ColorHSV]

######################################################################

import numpy as np
import statistics
import logging

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from lib.create_matrix import getValidDatasets, processImageabilityDatasetTerm, getImageability
from lib.imageability import getImageability as getImageabilityDataset
from lib.create_matrix import getNormalizedPOS

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)

gt_dataset = getImageabilityDataset()
terms = getValidDatasets(5000)

blacklist = ["image", "heavy", "logo", "talk", "cream", "freedom", "texture", "balloon", "production", "elephant", "volcano", "month", "quality", "cactus", "jet", "sister", "empty", "clock", "wheel", "dessert", "date", "common", "among", "private", "earth", "entire", "bee", "era", "deer", "double", "bag", "king", "lamp", "machine", "opportunity", "match", "ago", "skate", "care", "pumpkin", "cologne", "battle", "block", "zombie", "movement", "wish", "neighborhood", "foundation", "frame", "album", "system", "magazine", "bench", "ring", "rice", "believe", "flood", "lacrosse", "garage", "bread", "indian", "writing", "fight", "glacier", "edge", "wave", "control", "bottom", "member", "oil", "fan", "native", "kingdom", "tunnel", "blossom", "moment", "angel", "mall", "collage", "permission", "recent", "variety", "contrast", "bottle", "mare", "dawn", "capture", "section", "mud", "engagement"]

not_dictionary = ["evaluation", "algorithm", "stress", "pain", "adventure", "energy", "love", "motivation", "noise", "computer", "bicycle", "blackboard", "scarf", "shoes", "glasses", "document", "handicraft", "origami", "black", "white", "blue", "red", "green", "bright", "dark", "run", "walk", "cheer", "read", "write", "jump", "peaceful", "colorful", "annoying", "mysterious", "magical"]

for b in blacklist:
    try:
        terms.remove(b)
    except:
        pass


for b in not_dictionary:
    try:
        terms.remove(b)
    except:
        pass

not_dictionary = ["adventure", "energy", "love", # maybe low imageability nouns
             "computer", "bicycle", "shoes", # maybe high imageability nouns
             "black", "white", "blue", "red", "green", "bright", "dark", # adjectives
             "run", "walk", "read", "jump", # verbs
             "colorful", # adv
            ]

num_terms = len(terms)
logging.info("%i images each for %i terms" % (num_pics, num_terms))


def normalizeImageability(value):
    return ((value - 100) / 6)

np.random.seed(3310) # for reproducibility

logging.info(features)
vector = []
imageability = []
for term in terms:
    try:
        vector.append(processImageabilityDatasetTerm(term, features, num_pics))
    except:
        pass
    imageability.append(normalizeImageability(getImageability(term)))

train_size = int(num_terms * .8)
test_size = num_terms - train_size

vector_train = vector[0:train_size]
imageability_train = imageability[0:train_size]
vector_test = vector[train_size:]
imageability_test = imageability[train_size:]

terms_test = terms[train_size:]

imageabilities = []
imageabilities.extend(imageability_train)
imageabilities.extend(imageability_test)

# prepare right format
train_X = np.array(vector_train)
train_Y = np.array(imageability_train)

test_X = np.array(vector_test)
test_Y = np.array(imageability_test)

results_rf_mae = []
results_rf_corr = []
results_pred = []
rf_predictors = []

for i in range(10):
   trees = RandomForestRegressor(n_estimators=100)

   # train
   y_tree = trees.fit(train_X, train_Y).predict(train_X)

   # test
   test_tree = trees.predict(test_X)

   # eval
   tree_corr = np.corrcoef(test_Y, test_tree)[0, 1]
   tree_mae = mean_absolute_error(test_Y, test_tree)

   results_rf_corr.append(tree_corr)
   results_rf_mae.append(tree_mae)
   results_pred.append(test_tree)
   rf_predictors.append(trees)

from operator import itemgetter
best = min(enumerate(results_rf_mae), key=itemgetter(1))[0] 
logging.info("Random Forest: MAE %f \t\t Corr %f" % (results_rf_mae[best], results_rf_corr[best]))

import pickle
with open('models/randomforest_'+str(num_pics)+'.pickle', 'wb') as handle:
    pickle.dump(rf_predictors[best], handle, protocol=pickle.HIGHEST_PROTOCOL)
