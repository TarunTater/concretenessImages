#!/usr/bin/env python

# import torch.nn as nn
# import torchvision.models as models
# import torchvision.transforms as transforms

from lib.Feature import Feature
# from PIL import Image
# from torch.autograd import Variable

from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import numpy as np

class FeatureCNN(Feature):
    def __init__(self, type, image_set):
        super().__init__(type, image_set)
        self.size = 1

        # self.model = models.alexnet(pretrained=True)
        # # remove last fully-connected layer
        # new_classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])
        # self.model.classifier = new_classifier

        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        # self.preprocess = transforms.Compose([
        #     transforms.Scale(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     normalize
        # ])

        self.model = VGG19(weights='imagenet', include_top=False)

    def process(self):
        return super().process(self.image_set[0], force=False)

    def extract(self, inputimg, imgpath):
        # print(imgpath)
        img = image.load_img(imgpath, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        features = self.model.predict(x)
        # print(features.shape)
        return features.flatten()

        # img_pil = Image.fromarray(image)
        # img_tensor = self.preprocess(img_pil)
        # img_tensor.unsqueeze_(0)
        # img_variable = Variable(img_tensor)
        # fc_out = self.model(img_variable)
        # return fc_out.data.numpy()[0]
