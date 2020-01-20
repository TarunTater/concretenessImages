#!/usr/bin/env python3
from nltk.corpus import wordnet as wn

class YOLO9000Converter():
    lookupTable = dict() # SS->SS
    lookupTable2 = [] # IND->IND
    all_hypernyms = []
    original_synsets = []

    def reduce(self, synsets, steps=1, start=True):
        if(steps == 0):
            return synsets

        hypernyms = []
        for synset in synsets:
            hypernym = synset.hypernyms()
            
            if(not hypernym):
                if(start):
                    self.lookupTable[str(synset)] = [synset]
                hypernyms.append(synset)
                continue

            if(start):
                self.lookupTable[str(synset)] = hypernym
            else:
                for k, v in self.lookupTable.items():
                    if(synset in v):
                        self.lookupTable[k].remove(synset)
                        self.lookupTable[k].extend(hypernym)
                        self.lookupTable[k] = list(set(self.lookupTable[k]))
            hypernyms.extend(hypernym)

        hypernyms = list(set(hypernyms))

        return self.reduce(hypernyms, steps-1, start=False)


    def __init__(self, steps=2):
        with open("data/9k.labels", "r") as _labels:
            for l in _labels:
                synset = wn._synset_from_pos_and_offset('n', int(l[1:]))
                self.original_synsets.append(synset)

        #print("all size: ", len(synsets))
        self.all_hypernyms = self.reduce(self.original_synsets, steps)
        #print("hypernyms size:", len(self.all_hypernyms))
        print("reduced", len(self.original_synsets), "to", len(self.all_hypernyms))

        for i in range(len(self.original_synsets)):
            ss_name = self.original_synsets[i]

            ss_hypernyms = self.get_hypernyms(ss_name)
            ss_hypernyms_indices = []
            for h in ss_hypernyms:
                ss_hypernyms_indices.append(self.get_new_index(h))

            #print(ss_hypernyms, "->", ss_hypernyms_indices)

            self.lookupTable2.append(ss_hypernyms_indices)


    def get_hypernyms(self, synset):
        return self.lookupTable[str(synset)]

    def get_hypernyms_from_index(self, index):
        synset = self.original_synsets[index]

        return self.get_hypernyms(synset)

    def get_new_index(self, synset):
        return self.all_hypernyms.index(synset)

    def get_num(self):
        return len(self.all_hypernyms)

    def get_hypernym_indices_from_index(self, index):
        return self.lookupTable2[index]


#conv = YOLO9000Converter()

#synsets = []
#with open("data/9k.labels", "r") as _labels:
#    for l in _labels:
#        synset = wn._synset_from_pos_and_offset('n', int(l[1:]))
#        synsets.append(synset)

#for ss in synsets:
#    hypernyms = conv.get_hypernyms(ss)
#    for h in hypernyms:
#        print(h, conv.get_new_index(h))