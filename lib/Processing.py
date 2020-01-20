#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def process_word(word, num_images, features):
    for feature in features:

def get_vector(prefix, words, num_images, features):
    X = []

    for word in words:
        dataset = prefix + "/" + word
        X.append(process_word(word, num_images, features))
        
    return X