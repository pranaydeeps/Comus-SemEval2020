# -*- coding: utf-8 -*-
import os, sys, codecs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Force imports from grandparent directory
import random
from lt3util import featurizer, utilities
from preprocess import clean
import cyberbullying.features as cbf


# use_featurizer for huso19 (starting from training data for hateval (semeval19))
# FLAVOR: no prepro, no tok (only pattern)
# only for Task A
if __name__ == "__main__":
    FEATURIZER_FP = "huso19_noPrepro_featurizer.pkl" # path to which the featurizer is saved
    svmvector_fp = "huso19_noPrepro.svm"

    ### Get list of test post tuples
    ## TODO Nina/Els: simply read the data into two lists like below
    ## TODO When labels are not binary: set binary=False on f.featurize

    ### SPLITTING TRIAL DATA
    #reading in training data file
    # ZET DIT AAN VOOR EIGENLIJKE FEATURES AAN TE MAKEN 
    #f = codecs.open('training-huso19\\huso19_train_en_randomized.txt', 'r', 'utf-8')
    f = codecs.open('huso19_train_en_randomized.txt', 'r', 'utf-8')
    text = f.read()
    f.close()

    # create a list of all lines of this file
    l_lines = text.splitlines()
    #print(l_lines)

    # create nested list of all lines, splitting the string on tab delimiter
    l_split_lines = []
    for line in l_lines:
        line = line.split("\t")
        l_split_lines.append(line)
    # print(l_split_lines)

    # create a list of all posts; and three separate lists for each label
    posts = [item[1] for item in l_split_lines] # changed index into training data from 0 to 1, because hateval data includes tweet ID in first column

    g = codecs.open('no_prepro_huso_posts.txt', 'w', 'utf-8')
    for item in posts:
        g.write(item + "\n")
    g.close()

    l_label1 = [item[2] for item in l_split_lines]
    # print(l_label1)
    l_label2 = [item[3] for item in l_split_lines]
    # print(l_label2)
    l_label3 = [item[4] for item in l_split_lines]
    # print(l_label3)
    labels = []
    for elem in l_label1:
        if elem == '1':
            labels.append(True)
        else:
            labels.append(False)
 
    assert len(posts) == len(labels)
    doc_list = [(i,p) for i,p in enumerate(posts)] 
    label_list = [(i, l) for i, l in enumerate(labels)]
    features = [cbf.Feature("orig", fg) for fg in sorted(cbf.offenseval_fndict.keys())]

    f = featurizer.Featurizer()
    f.add_features(cbf.offenseval_fndict, features)
    f.featurize(doc_list, label_list, format="svm", path=svmvector_fp, binary=True)
    f.save(FEATURIZER_FP)