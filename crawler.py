#!/usr/bin/env python3

from nltk.corpus import wordnet as wn
import bz2
# import shutil
import logging
import os
import urllib.request
import re
import json
from collections import defaultdict
from shutil import copyfile

start = 0

dataset = "/ssdwork/yfcc100m/yfcc100m_dataset.bz2"
autotags = "/ssdwork/yfcc100m/yfcc100m_autotags.bz2"


dataset_base = "/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/imagibility_testing_dataset"
location = dataset_base + "/imageability_flickr/"
jsonfile = dataset_base + "/imageability_flickr.json"
autotags_json = dataset_base + "/imageability_flickr_autotags.json"


logfile = location + "/log.txt"
logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.WARNING)


#dictionary = defaultdict(lambda: defaultdict(dict))
#
#def analyze_synset(root):
#
#    words = []
#    for w in root.lemma_names('eng'):
#        w = re.sub("_", " ", w)
#        words.append(w)
#
#    for word in words:
#        # logging.info("{} -> {}".format(word, str(root)))
#
#        if dictionary[str(word)] is list:
#            dictionary[str(word)].append(str(root))
#        else:
#            dictionary[str(word)] = [str(root)]
#
#    for sub in root.hyponyms():
#        analyze_synset(sub)

#analyze_synset(wn._synset_from_pos_and_offset('n', 4524313))  # vehicle

import lib.imageability
imageability_dataset = imageability.getImageability()

#print(imageability_dataset)
terms = []

for k, v in imageability_dataset.items():
	print(k, v)
	terms.append(k)


# this function basically tests whether an image is corrupted
# or not by opening it and doing some simple image processing
def testImage(filen):
    try:
        import cv2
        # dummy opening and dummy resize
        img = cv2.imread(filen, 1)
        resized = cv2.resize(img, (512, 512))
        return True
    except Exception as e:
        print(e)
        return False


# dl images
def dl(synsets, entry):
    url = entry[16]
    #folder = location + synsets[0] + "/"

    hashname = entry[2]
    filename = hashname + "." + entry[23]

    folder = location + filename[0:3] + "/"

    if not os.path.exists(folder):
        os.makedirs(folder)

    try:
        urllib.request.urlretrieve(url, folder + filename)  # noqa: E501
        
        #from testImages import testImage
        if(not testImage(folder + filename)):
                #print("the image file seems to have some issues. delete.")
                os.remove(folder+filename)
                return None
        #print("file ok")

        return filename
    except Exception as e:
        print(str(e))
        return None

def searchdict(word, entry):
    word = re.sub("\+", " ", word)
    if word in terms:
        #logging.info("{} in list ({})".format(str(word), imageability_dataset[word]))
        logging.info("{} in list".format(str(word)))

        dl(word, entry)
        return word
    return None

with open(jsonfile, 'r') as infile:
        archive = json.load(infile)

with open(autotags_json, 'r') as infile:
        tagarchive = json.load(infile)

i = 0

import progressbar
bar = progressbar.ProgressBar(max_value=100000000, redirect_stdout=True)
#from tqdm import tqdm
#bar = tqdm(total=100000000, desc="images")
with open(logfile, "w") as logging_file:
    with bz2.open(dataset, "rt") as fin_data:
        with bz2.open(autotags, "rt") as fin_tags:
            for t in zip(fin_data, fin_tags):
                line = t[0]
                line_tags = t[1]
                bar.update(i)
                if i <= start:
                    i = i+1
                    continue

                entry = line.split("\t")
                entry_tags = line_tags.split("\t")
                user_tags = entry[10]
                machine_tags = entry[11]

                title = entry[8]
                description = entry[9]

                fits = []
                for tag in user_tags.split(","):
                    ret = searchdict(tag, entry)
                    if ret is not None:
                        fits.append(ret)

                # for tag in machine_tags.split(","):
                #     searchdict(tag, entry)

                # TODO fix n-grams for multiple word synsets
                for word in title.split("+"):
                    ret = searchdict(word, entry)
                    if ret is not None:
                        fits.append(ret)

                for word in description.split("+"):
                    ret = searchdict(word, entry)
                    if ret is not None:
                        fits.append(ret)

                #print(fits)
                hashname = None
                if(len(fits) is not 0):
                    hashname = dl(fits, entry)

                    if hashname is not None:
                        print(hashname, fits)
                        tags = entry_tags[1]
                        tagdict = {}
                        if tags is not "\n":
                            for tag in tags.split(","):
                                tagname = tag.split(":")[0]
                                tagval = tag.split(":")[1]

                                tagdict[tagname] = float(tagval)
                        tagarchive[hashname] = tagdict
                        print(tagdict)

                # print("title: ", title)
                # print("description: ", description)

                for f in fits:
                    if hashname is not None:
                        #print(hashname, "->", f)
                        try:
                            if(hashname not in archive[f]):
                                archive[f].append(hashname)
                        except KeyError:
                            archive[f] = []
                            archive[f].append(hashname)
                
                print()
                
                if i % 25000 == 0:
                    logging.info("save %d" % i)
                    with open(jsonfile, "w") as json_file:
                        json.dump(archive, json_file)

                    with open(autotags_json, "w") as json_file:
                        json.dump(tagarchive, json_file)

                i = i+1
bar.close()


# 0 * Line number
# 1 * Photo/video identifier
# 2 * Photo/video hash
# 3 * User NSID
# 4 * User nickname
# 5 * Date taken
# 6 * Date uploaded
# 7 * Capture device
# 8 * Title
# 9 * Description
# 10 * User tags (comma-separated)
# 11 * Machine tags (comma-separated)
# 12 * Longitude
# 13 * Latitude
# 14 * Accuracy of the longitude and latitude coordinates (1=world level accuracy, ..., 16=street level accuracy)
# 15 * Photo/video page URL
# 16 * Photo/video download URL
# 17 * License name
# 18 * License URL
# 19 * Photo/video server identifier
# 20 * Photo/video farm identifier
# 21 * Photo/video secret
# 22 * Photo/video secret original
# 23 * Extension of the original photo
# 24 * Photos/video marker (0 = photo, 1 = video)