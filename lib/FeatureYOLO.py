#!/usr/bin/env python
# flake8: noqa

from lib.Feature import Feature, FeatureType
import cv2
import numpy as np

class FeatureYOLO(Feature):
    def __init__(self, type, image_set):
        super().__init__(type, image_set)

        self.allclasses = []
        self.gridsize = 10

        if(type is FeatureType.YOLO_NUM):
            with open("data/coco.names", 'r') as infile:
                for line in infile:
                    self.allclasses.append(line.rstrip())

            from pydarknet import Detector
            self.net = Detector(bytes("cfg/yolov3.cfg", encoding="utf-8"), bytes("weights/yolov3.weights", encoding="utf-8"), 0, bytes("cfg/coco.data", encoding="utf-8"))
        elif(type is FeatureType.YOLO_NUM_9000 or type is FeatureType.YOLO_COMPOSITION):
            with open("data/9k.names", 'r') as infile:
                for line in infile:
                    self.allclasses.append(line.rstrip())

            from pydarknet import Detector
            import pydarknet
            #pydarknet.set_cuda_device(1)
            self.net = Detector(bytes("cfg/yolo9000.cfg", encoding="utf-8"), bytes("weights/yolo9000.weights", encoding="utf-8"), 0, bytes("cfg/combine9k.data", encoding="utf-8"))

    def process(self):
        return super().process(self.image_set[0], force=False)


    def gridcoords(self, n, m, img_w, img_h):
        x_size = img_w / self.gridsize
        y_size = img_h / self.gridsize

        return (x_size * n), (x_size * (n+1))-1, (y_size * m), (y_size * (m+1))-1


    def extract(self, image, imagepath):
        img = np.array(image)
        img = img[:,:,::-1] # RGB to BGR

        from pydarknet import Image
        img2 = Image(img)

        results = self.net.detect(img2)

        # converter process
        from merge_yolo9000 import YOLO9000Converter
        conv = YOLO9000Converter()

        if(type is FeatureType.YOLO_NUM_9000 or type is FeatureType.YOLO_NUM):
            #resultvector = [0] * len(self.allclasses)
            resultvector = [0] * conv.get_num()
            
            for tag in results:
                #index = self.allclasses.index(tag[0].decode("utf-8"))
                _index = self.allclasses.index(tag[0].decode("utf-8"))
                index_l = conv.get_hypernym_indices_from_index(_index)
                value = tag[1]

                #resultvector[index] = resultvector[index] + value
                for index in index_l:
                    resultvector[index] = resultvector[index] + value

            return resultvector

        elif(type is FeatureType.YOLO_COMPOSITION):
            img_h = np.size(image, 0)
            img_w = np.size(image, 1)

            resultvector = [0] * (self.gridsize * self.gridsize)
            for cat, score, bounds in results:
                x, y, w, h = bounds

                x1 = int(x-w/2) 
                y1 = int(y-h/2)
                x2 = int(x+w/2)
                y2 = int(y+h/2)

                i = 0
                for n in range(self.gridsize):
                    for m in range(self.gridsize):
                        g_x1, g_x2, g_y1, g_y2 = self.gridcoords(n, m, img_w, img_h)
                        x_overlaps = (x1 <= g_x2) and (x2 >= g_x1)
                        y_overlaps = (y1 <= g_y2) and (y2 >= g_y1)
                        collision = x_overlaps and y_overlaps

                        if(collision):
                            resultvector[i] = resultvector[i] + 1
                        else:
                            pass

                        i = i + 1
            
            return resultvector

        else:
            # unimplemented?
            return []
