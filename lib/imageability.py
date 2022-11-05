#!/usr/bin/env python3
# import collections


def getCortese():

    filename = "data/cortese.txt"

    with open(filename) as f:
        content = f.readlines()

    content = [x.strip() for x in content]

    mode = 0
    cortese = dict()

    item = []
    meanRT = []
    sdRT = []
    meanR = []
    sdR = []

    for l in content:
        # print(mode)

        if l == "":
            mode = mode + 1
            continue

        if mode == 6:
            assert len(item) == len(meanRT) == len(sdRT) == len(meanR) == len(sdR)
            for i, _ in enumerate(item):
                cortese[item[i]] = [meanRT[i], sdRT[i], meanR[i], sdR[i]]
            mode = 0

            item = []
            meanRT = []
            sdRT = []
            meanR = []
            sdR = []

        if mode == 0:
            item.append(l)

        if mode == 1:
            meanRT.append(float(l))

        if mode == 2:
            sdRT.append(float(l))

        if mode == 3:
            meanR.append(int(float(l) * 100))

        if mode == 4:
            sdR.append(int(float(l) * 100))

    assert len(item) == len(meanRT) == len(sdRT) == len(meanR) == len(sdR)
    for i, _ in enumerate(item):
        cortese[item[i]] = [meanRT[i], sdRT[i], meanR[i], sdR[i]]
    print(f"From cortese")
    for it, mrt, sdrt, mr, sdr in zip(item, meanRT, sdRT, meanR, sdR):
        print(it, mrt, sdrt, mr, sdr)
    # print(cortese)
    return cortese


def getReilly():
    filename = "data/reilly_1.txt"

    with open(filename) as f:
        content =  f.readlines() #["apple", "pen", "shining", "world"] #f.readlines()

    content = [x.strip() for x in content]

    mode = 0
    reilly = dict()
    item = []
    vals = []
    # print(content)
    for l in content:
        if l == "":
            mode = mode + 1
            continue

        if mode == 0:
            item.append(l.lower())

        if mode == 1:
            vals.append(int(l))
    # print(f"From getReilly, we obtained: ")
    # for v, it in zip(vals, item):
    #     print(v, it)
    for i, _ in enumerate(item):
        reilly[item[i]] = vals[i]

    return reilly


def getImageability():
    # cortese = getCortese()
    reilly = getReilly()

    imageability = dict()

    for k, v in reilly.items():
        #print(k,v)
        imageability[k] = v

    # for k, v in cortese.items():
    #     #print(k,v)
    #     try: 
    #         if(imageability[k]):
    #             imageability[k] = int((imageability[k] + v[2]) / 2)
    #     except:
    #         imageability[k] = v[2]

    #s = [(k, imageability[k]) for k in sorted(imageability, key=imageability.get, reverse=False)]
    #for k, v in s:
    #    print(k, v)
    #print(len(imageability))

    return imageability


def getSynsetFromImageNet(ln):
    from nltk.corpus import wordnet as wn
    return wn.synset_from_pos_and_offset('n', int(ln[1:]))


def getWordsForSynset(synset, lang="eng"):
    words = []
    import re
    for w in synset.lemma_names(lang):
        w = re.sub("_", " ", w)
        words.append(w)
    return words


def getImageNet(imageability):
    #import os
    filename = "imagenet_new.txt"
    threshold = 750

    with open(filename) as f:
        content = f.readlines()

    synsets = []
    ss = dict()
    for l in content:
        if not l:
            continue

        ln, lc = l.split(",")

        try:
            synset = getSynsetFromImageNet(ln)
            words = getWordsForSynset(synset)
            #print(words)

            # check if word overlaps with imageability datasets
            for w in words:
                if w in imageability:
                    #print(w, imageability[w])

                    # check number of images
                    imgcount = int(lc)
                    #print(synset, imgcount)
                    if imgcount >= threshold:
                        synsets.append(ln)
                        ss[ln] = imageability[w]
        except Exception as e:
            print(e)

    #s = [(k, ss[k]) for k in sorted(ss, key=ss.get, reverse=True)]
    #for k, v in s:
        #print(k, v)

    return ss


def getVisualVector(imset):
    try:
        from Feature import Feature, FeatureType
        feature = Feature.factory(type=FeatureType.GIST, im_set=[imset])
        feature.process()

        return feature.results
    except Exception as e:
        print(e)
        return None

def getCluster(X):
    import numpy as np
    from sklearn.cluster import MeanShift, estimate_bandwidth
    # The following bandwidth can be automatically detected using
    bandwidth = estimate_bandwidth(X, quantile=0.5, n_jobs=-1)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=False, cluster_all=False, n_jobs=-1)

    ms.fit(X)
    labels = ms.labels_
    #cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique) + list(labels).count(-1)

    return n_clusters_


def processImageSet(imagenet_id):
    #synset = getSynsetFromImageNet(imagenet_id)
    path = "/muradata/imagenet/" + imagenet_id + "/"

    #import glob
    #imgs = []
    #try:
    #    imgs = glob.glob(path + "*")
    #except Exception as e:
    #    print(e)

    from Imageset import Imageset
    imset = Imageset(imagenet_id, path)
    #print(imset.images)
    print(imagenet_id, "get visual vectors")
    visvec = getVisualVector(imset)
    if visvec is None:
        print(imagenet_id, "failed to get vector")
        return None

    print(imagenet_id, "get estimation")
    return getCluster(visvec)


def normalizeResultsToLikert(resultset):
    lower, upper = 100, 700

    highest_v = max(resultset.values())
    for k, v in resultset.items():
        v_norm = v / highest_v
        resultset[k] = int(lower + (upper - lower) * v_norm)

    return resultset


def evaluate(resultsetA, resultsetB):
    # just to make sure that both lists are in same order, because we don't want trouble
    A = []
    B = []

    for k in sorted(resultsetB, key=resultsetB.get):
        try:
            if(resultsetA[k] and resultsetB[k]):
                A.append(resultsetA[k])
                B.append(resultsetB[k])
        except Exception as e:
            print(str(e))

    print(A)
    print(B)

    from scipy.stats import stats
    corr = stats.spearmanr(A, B)[0]

    from sklearn import metrics
    mse = metrics.mean_absolute_error(A, B)

    print(corr, mse)
    return corr, mse

def main():
    imageability_dataset = getImageability()
    result_groundtruth = getImageNet(imageability_dataset)
    print(len(result_groundtruth))
    result_estimation = dict()

    for synset, label in result_groundtruth.items():
        val = processImageSet(synset)
        if val is None:
            continue
        result_estimation[synset] = val

    result_estimation = normalizeResultsToLikert(result_estimation)

    for k, v in result_estimation.items():
        print(k, v, result_groundtruth[k])

    evaluate(result_estimation, result_groundtruth)


def main2():
    imageability_dataset = getImageability()

    new_words = ["evaluation", "algorithm", "stress", "pain", "adventure", "energy", "love", "motivation", "noise", # maybe low imageability nouns
                 "computer", "bicycle", "blackboard", "scarf", "shoes", "glasses", "document", "handicraft", "origami", # maybe high imageability nouns
                 "black", "white", "blue", "red", "green", "bright", "dark", # adjectives
                 "run", "walk", "cheer", "read", "write", "jump", # verbs
                 "peaceful", "colorful", "annoying", "mysterious", "magical" # adverbs
                ]


    for word in new_words:
        status = (word in imageability_dataset)
        if(status is False):
            print(word, status)
        else:
            print(word, imageability_dataset[word])
    print(imageability_dataset)

if __name__ == '__main__':
    main2()