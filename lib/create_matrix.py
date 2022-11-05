#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from lib.count_imageability_dataset import loadDict as loadImagesetDict
from lib.imageability import getImageability as getImageabilityDataset
import logging
import numpy as np

imagesets = loadImagesetDict()
pretrainedBoW = ["imageability_flickr_20181017", "a4e4e8c68af991bfc0c48bcc22e10b74ba42d76004d72251f4d5e687"]

def getNormalizedPOS(term):
    import nltk
    wtype_ = nltk.tag.pos_tag([term])[0]
    if(wtype_[1] == "NN" or wtype_[1] == "NNS" or wtype_[1] == "NNP" or wtype_[1] == "NNPS"):
        wtype = "noun"
    elif(wtype_[1] == "JJ" or wtype_[1] == "JJR" or wtype_[1] == "JJS"):
        wtype = "adjective"
    elif(wtype_[1] == "RB"):
        wtype = "adverb"
    #elif(wtype_[1] == "IN"):
    #    wtype = "preprosition"
    elif(wtype_[1] == "MD" or wtype_[1] == "VBG" or wtype_[1] == "VBD" or wtype_[1] == "VB" or wtype_[1] == "VBN" or wtype_[1] == "VBP" or wtype_[1] == "VBZ"):
        wtype = "verb"
    else:
        wtype = "other"

    return wtype

def getTaggedImageability(terms=imagesets):
    import nltk
    imageability_tagged = []
    for term in terms:
        word = term
        wtype = getNormalizedPOS(word)
        imageability_tagged.append((word, wtype))

    tag_fd = nltk.FreqDist(tag for (word, tag) in imageability_tagged)
    print(tag_fd.most_common())

    return imageability_tagged


def clampImageset(data, image_threshold):
    valids = []
    blacklist = ["image"]
    for term in data:
        if term not in blacklist:
            if len(data[term]) >= image_threshold:
                valids.append(term)
    return valids


def getVisualVector(imset, featureType):
    try:
        from lib.Feature import FeatureType
        from lib.Feature import Feature
        if(featureType is FeatureType.SURF):
            print(f"imset {imset} from from if statement")
            feature = Feature.factory(type=featureType, im_set=[imset], trained_hash=pretrainedBoW[1], trained_directory=pretrainedBoW[0])
        else:
            print(f"imset {imset} from from else statement")
            feature = Feature.factory(type=featureType, im_set=[imset])
        feature.process()
        print(f"Got features")

        return feature.results
    except Exception as e:
        print(str(e))
        return None


def getSimilarityMatrix(visvec):
    num_images = len(visvec)

    matrix = [[0] * num_images for i in range(num_images)]
    #print(visvec)
    #print(visvec.shape)
    for i in range(num_images):
        for j in range(num_images):
            import cv2
            # the SURF/BOW has weird dimensions (n, 1, m) instead of (n, m)
            left = visvec[i].flatten()
            right = visvec[j].flatten()

            #print(left.shape)
            #print(right.shape)

            left = np.float32(left)
            right = np.float32(right)

            matrix[i][j] = cv2.compareHist(left, right, method=0)

    result = np.array(matrix)
    np.set_printoptions(precision=3)
    #print(result)
    return result


def hash(term, feature, num_images):
    import hashlib
    m = hashlib.sha224()
    m.update(term.encode("utf-8"))
    m.update(repr(num_images).encode("utf-8"))
    m.update(repr(str(feature)).encode("utf-8"))
    m.update(repr(6).encode("utf-8")) # Versioning Salt

    return m.hexdigest()


def getEigenvalueFilename(term, feature, num_images):
    import os
    directory = os.path.join(os.path.abspath(os.path.dirname(__file__))[:-4], "eigenvalues")
    location = directory + "/" + str(hash(term, feature, num_images)) + ".pickle"

    import os
    if not os.path.exists(directory):
        os.makedirs(directory)

    return location


def loadEigenvalues(term, feature, num_images):
    filename = getEigenvalueFilename(term, feature, num_images)
    #print(term, feature, num_images, filename)
    import pickle
    try:
        print(f"Looking for eigenvalue cache in {filename}")
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
    except FileNotFoundError:
        print(term, "could not find eigenvalue cache.")
        #raise Exception("ABC")
        data = None
    return data


def saveEigenvalues(term, vector, feature):
    num_images = len(vector)
    filename = getEigenvalueFilename(term, feature, num_images)
    print(term, feature, num_images, filename)

    import pickle
    with open(filename, 'wb') as handle:
        pickle.dump(vector, handle, protocol=pickle.HIGHEST_PROTOCOL)


def processImageSet(term, files, features, returnEigenvalues):
    eigenvalues = []
    for feature in features:
        eigenvalue = loadEigenvalues(term, feature, len(files))
        if eigenvalue is None:
            logging.debug(str(feature) + " couldn't find cache, calculate.")
            from lib.Imageset import Imageset
            imset = Imageset(term, files)
            print(term, "get visual vectors")
            visvec = getVisualVector(imset, feature)
            print(f"==> Got visvec term. {visvec}")
            if visvec is None:
                print(term, "failed to get vector")
                return None
            if not returnEigenvalues:
                return visvec

            print(term, "get distance matrix")
            matrix = getSimilarityMatrix(visvec)

            print(term, "get eigenvalues")
            import numpy.linalg as LA
            eigenvalue, eigenvector = LA.eigh(matrix)
            eigenvalue = LA.eigvalsh(matrix)

            print(term, "save eigenvalues")
            saveEigenvalues(term, eigenvalue, feature)

        # only top 100
        eigenvalue = eigenvalue[-30:]
        eigenvalues.extend(eigenvalue)



        #eigenvalues.append(eigenvalue)

    return eigenvalues


# this function gets files from imageability dataset
# /work/kastnerm/imageability_flickr/
def processImageabilityDatasetTerm(term, features, num_images, returnEigenvalues=True):
    files = imagesets[term][0:num_images]

    return processImageSet(term, files, features, returnEigenvalues)


# this function gets fiels from a folder
def processLooseWord(word, location, features, num_images, returnEigenvalues=True):
    import glob
    files = []
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
    for ext in extensions:
        files.extend(glob.glob(location + ext))
    #print(files)

    return processImageSet(word, files, features, returnEigenvalues)


def getValidDatasets(num_images):
    valids = clampImageset(imagesets, num_images)

    return valids

def getImageability(term):
    imageability = getImageabilityDataset()
    # print(f"Imageability are: {imageability}")
    value = imageability[term]
    return value

if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG
                    )
    from Feature import Feature, FeatureType

    features = [FeatureType.ColorHSV, FeatureType.SURF, FeatureType.GIST, FeatureType.VisualConcepts]

    import socket
    if(socket.gethostname()[0:3] == "kon"):
        features = [FeatureType.YOLO_NUM_9000, FeatureType.YOLO_COMPOSITION]

    num_images = 7500
    used = []

    terms = getValidDatasets(num_images)

    i = 0

    termlist = terms

    if(socket.gethostname() == "kon04"):
        termlist = reversed(terms)

    for term in termlist:
        print("("+str(i)+"/"+str(len(terms))+") process", term)
        used.append(term)
        result = processTerm(term, features, num_images)
        i = i + 1

    print(used)
    print(len(used))
