#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

# call like this: "./cleanup.py /work/kastnerm/test/"
# script will go in all subdirectories from the given path

# alternatively, set the fallback path here:
prefix = "/work/kastnerm/abc/"

#####################################

if (len(sys.argv) == 2):
    prefix = sys.argv[1]

print(prefix)

def check(root, filterGifs = True, filterDuplicates = True, filterCorrupts = True):
    from collections import defaultdict
    from PIL import Image
    import os
    import imagehash
    import glob
    import logging
    import cv2

    logging.debug("check {}".format(str(root)))
    for item in os.listdir(root):
        obj = os.path.join(root, item)
        if os.path.isdir(obj):
            check(obj)

    hashtable = defaultdict(lambda: defaultdict(dict))
    pre_am = len(glob.glob(root + "/*.*"))
    for imagePath in glob.glob(root + "/*.*"):
        if os.path.isfile(imagePath):
            # Filter GIFs
            if(filterGifs): 
                logging.debug("filter gifs")
                #import re
                #if (re.search(".gif", imagePath, re.IGNORECASE)):
                if (imagePath.endswith(".gif")):
                    try:
                        logging.debug("gif file: {}".format(imagePath))
                        os.remove(imagePath)
                    except:
                        pass
                    continue

            # Filter duplicates
            if(filterDuplicates):
                logging.debug("filter duplicates")
                try:
                    image = Image.open(imagePath)
                    logging.info(imagePath)
                    h = str(imagehash.dhash(image))
                    if (not hashtable[h]):
                        hashtable[h] = imagePath
                    else:
                        logging.debug("collision: {} {}".format(imagePath, hashtable[h]))
                        os.remove(imagePath)
                        continue
                except:
                    logging.debug("unreadable file: {}".format(imagePath))
                    try:
                        os.remove(imagePath)
                    except:
                        pass
                    continue

            # Check corrupt files
            # this approach is stupid but it makes sure that all 
            # common imaging libraries can read this file.
            if(filterCorrupts):
                logging.debug("filter corrupts")
                try:
                    image = Image.open(imagePath) # Open image in PIL usually already crashes for corrupt ones
                    imagehash.dhash(image) # Do something

                    image = cv2.imread(imagePath) # Or maybe OpenCV doesn't like it                  
                    cv2.resize(image, (1234, 1234)) # Do something

                    # by now something should have crashed if the file isn't processable!
                except:
                    logging.debug("unreadable file: {}".format(imagePath))
                    try:
                        os.remove(imagePath)
                    except:
                        pass
                    continue

    post_am = len(glob.glob(root + "/*.*"))
    logging.info("deleted {} files for {}".format(pre_am - post_am, str(root)))


def main():
    import logging
    logging.basicConfig(
        format='[%(asctime)s %(levelname)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)

    check(prefix, filterGifs=True, filterDuplicates=True, filterCorrupts=True)

if __name__ == "__main__":
    main()
