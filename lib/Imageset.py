#!/usr/bin/env python

import hashlib
import logging

import socket
hostname = socket.gethostname()
#if hostname[0:3] == "kon" or hostname[0:3] == "sak":
#    logging.debug("[imageset.py] detected server")
#    DATASET_LOCATION = "/work/kastnerm/"
#else:
#    logging.debug("[imageset.py] detected workstation")
#    DATASET_LOCATION = "/ssdwork/"

DATASET_LOCATION = "/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/imagibility_testing_dataset"

def get_real_filename(filen):
    #print(filen)
    if (filen[0] == "/") or (filen[0:1] == "./"):
        return filen
    else:
        return DATASET_LOCATION + "imageability_flickr/" + filen[0:3] + "/" + filen

class Imageset(object):
    def __init__(self, name, images):
        self.name = name
        self.images = images

    @property
    def size(self):
        try:
            return len(self.images)
        except:
            return -1

    @property
    def start(self):
        return 0

    @property
    def end(self):
        return self.size - 1


    @property
    def hash(self):
        m = hashlib.sha224()
        m.update(self.name.encode("utf-8"))
        m.update(repr(self.size).encode("utf-8"))

        m.update(repr(1).encode("utf-8")) # Versioning Salt

        logging.debug(self.name)
        logging.debug(self.size)
        logging.debug(m.hexdigest())

        return m.hexdigest()

    @property
    def info(self):
        return (self.name,
                self.size)
