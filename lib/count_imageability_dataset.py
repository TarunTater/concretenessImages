#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, os.path
import json

from lib.Imageset import DATASET_LOCATION

def checkFile(fname):
    try:
        import cv2
        img = cv2.imread(fname, 1)
        img_size = 512
        resized = cv2.resize(img, (img_size, img_size))
        return True
    except Exception as e:
        print(str(e))
        return False

# def createDict():

#     # path joining version for other paths
#     dataset_directory = '/ssdwork/flickr/'
#     terms = [name for name in os.listdir(dataset_directory) if os.path.isdir(os.path.join(dataset_directory, name))]

#     def fileList(source):
#         matches = []
#         for root, dirnames, filenames in os.walk(source):
#             for filename in filenames:
#                 fullname = os.path.join(root, filename)
#                 if os.path.getsize(fullname) > 0:
#                     if checkFile(fullname):
#                         matches.append(fullname)
#         return matches

#     data = {}

#     for term in terms:
#         term_directory = dataset_directory + term + '/'
#         imgs = fileList(term_directory)
#         print(term, len(imgs))
#         data[term] = imgs

#     import pickle
#     with open('imageability_dataset.pickle', 'wb') as handle:
#         pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

#     return data

def loadDict():
    import pickle
    #try:
    with open(DATASET_LOCATION + '/imageability_flickr.json', 'r') as handle:
        #data = pickle.load(handle)
        data = json.load(handle)

    #except FileNotFoundError:
    #    data = createDict()

    return data

def roundToInt(a):
    import numpy as np
    return int(np.around(a))

def analyzeData(data, image_threshold):
    #print("For >=", image_threshold, "images per term:")
    blacklist = ["image"]
    nums = {}
    for term in data:
        #print(term, len(data[term]))
        if term not in blacklist:
            if len(data[term]) >= image_threshold:
                nums[term] = len(data[term])

    import operator
    import numpy as np
    sorted_nums = sorted(nums.items(), key=operator.itemgetter(1))

    #print("Number of terms with dataset:", len(sorted_nums))
    #print("Average #images per term:", roundToInt(np.average(list(nums.values()))))
    #print("Standard deviation #images per term:", roundToInt(np.std(list(nums.values()))))
    #print()
    return len(sorted_nums)


if __name__ == '__main__':
    data = loadDict()  # Loads the datset from the json 
    limits = [2500, 3000, 4000, 5000, 6000, 7000, 7500, 8000, 9000, 10000, 15000, 20000, 25000, 30000, 50000]
    for limit in limits:  # For all limits, checking the number of images if exists. 
        print(format(limit, ">7n"), "images per term:", format(analyzeData(data, limit), ">5n"), "words available")
