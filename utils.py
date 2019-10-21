#!usr/bin/env python
# encoding: utf-8

"""
utils_els.py

Script to predict sentiment for memes based on text for SemEval 2020 task "Memotion"

Sentiment features adapted from script created by Marjan Van de Kauter on 2015-01-12.
Copyright (c) 2019 LT3. All rights reserved.

sys.argv[1] = working directory
    -> directory in which (unfixed) dictionaries and input files for preprocessing tools are stored
    -> be sure to use your own personal directory!
    -> this directory should contain a file features.txt , which indicates the features that should be extracted
    -> this directory should also contain a subdirectory called prepro, in which the input and output files of
       the preprocessing pipeline are stored
sys.argv[2] = modus ('train' or 'test')
sys.argv[3] = tab delimited input file
    -> first column: tokenized instance (with tokens separated by spaces)
    -> second column: label
sys.argv[4] = output file
    -> the extracted feature vectors will be written to this file
sys.argv[5] = feature index file (this argument is optional)
    -> the index number(s) for each extracted feature (set) will be written to this file

Which features can be extracted?

* Token ngram features
* Character ngram features
* Flooding features
* Capitalization features
* Punctuation features
* Hashtag features
* Sentiment lexicon and emoticon features

All feature vectors are written to the given output file path
"""

from nltk.tokenize import word_tokenize
import codecs, sys, os
import string
import re
import math
import csv
from spacy.lang.en import English
import subprocess
from Instance import Instance
from Token import Token
from FeatureSelection import FeatureSelection


'task: perform classification of Memes for SemEval 2020 Memotion task'

def prep_task1(data_list):
    task1_list = []
    for item in data_list:
        if len(item)==9:
            temp_list = []
            temp_list.append(item[0])
            print (temp_list)
            temp_list.append(item[2])
            print (temp_list)
            if 'positive' in item[8]:
                temp_list.append(1)
            if 'neutral' in item[8]:
                temp_list.append(0)
            if 'negative' in item[8]:
                temp_list.append(2)
            task1_list.append(temp_list)
    print(task1_list)
    return task1_list

def load_nrc_emotion_lex(lex_path):

    with open(lex_path) as f:
        content = f.readlines()
    content = [x.strip() for x in content] 
    
    pos_words = []
    neg_words = []
    neutral_words = []
    pos_check = 0
    for words in content:
        processed = words.split('\t')
        if len(processed)==3:
            if processed[1] == 'positive' and processed [2] == '1':
                pos_words.append(processed[0])
            if processed[1] == 'negative' and processed [2] == '1':
                neg_words.append(processed[0])
    
    for words in content:
        processed = words.split('\t')
        if len(processed)==3:
            if processed[0] not in pos_words and processed[0] not in neg_words and processed[0] not in neutral_words:
                neutral_words.append(processed[0])

    #Els: TODO add overal sentiment lexicon value

    pos_words = list(set(pos_words))
    neg_words = list(set(neg_words))
    neutral_words = list(set(neutral_words))            
    
    return pos_words, neg_words, neutral_words


def lex_feature_extractor(pos_words, neg_words, neutral_words, string):
    print("Speaking from the Lex Feature Extractor. Obtained String: {}".format(string))
    words = string.split()
    pos_score = 0.0
    neg_score = 0.0
    neutral_score = 0.0
    total_score = 0
    for word in words:
        if word in pos_words:
            pos_score+=1
        if word in neg_words:
            neg_score+=1
        if word in neutral_words:
            neutral_score+=1

    total_score = pos_score - neg_score
    pos_score = pos_score
    neg_score = neg_score
    neutral_score = neutral_score

    return [total_score, pos_score, neg_score, neutral_score]
    
def createTrainingLists(instanceObjectList):
    '''
    Extracts lists of token ngrams, character ngrams, lemma ngrams, lemma+PoStag ngrams and dependency relations 
    from the training data and stores them in text files to be used to create the token ngram, character ngram, 
    lemma ngram, lemma+PoStag ngram, dependency relation and PMI ngram features
    '''
    print ('Creating training dictionaries...')
    tokenNgramFeatureRange = FeatureSelection.getInstance(featureFile).tokenNgramFeatureRange   
    characterNgramFeatureRange = FeatureSelection.getInstance(featureFile).characterNgramFeatureRange
    trainingTokenNgrams = [[] for _ in tokenNgramFeatureRange]
    trainingTokenUnigrams = []
    trainingTokenUniBigrams = []
    trainingCharacterNgrams = [[] for _ in characterNgramFeatureRange]
    for instanceObject in instanceObjectList:
        if tokenNgramFeatureRange != range(0, 1):
            for i, ngramList in enumerate(instanceObject.getTokenNgrams(tokenNgramFeatureRange)):
                for ngram in ngramList:
                    trainingTokenNgrams[i].append(ngram)
        if characterNgramFeatureRange != range(0, 1):
            for i, ngramList in enumerate(instanceObject.getAllCharacterNgrams(characterNgramFeatureRange)):
                for ngram in ngramList:
                    trainingCharacterNgrams[i].append(ngram)
    trainingTokenNgrams = [list(set(ngramList)) for ngramList in trainingTokenNgrams]
    trainingTokenUnigrams = list(set(trainingTokenUnigrams))
    trainingTokenUniBigrams = list(set(trainingTokenUniBigrams))
    trainingCharacterNgrams = [list(set(ngramList)) for ngramList in trainingCharacterNgrams]
    if trainingTokenNgrams:
        tokenNgramTrainingListFileName = os.path.join(workingDirectory, 'tokenNgramList.txt')
        if os.path.isfile(tokenNgramTrainingListFileName):
            open(tokenNgramTrainingListFileName, 'w').close()
        tokenNgramTrainingListFile = codecs.open(tokenNgramTrainingListFileName, 'a', 'utf8')
        nr = 1
        for ngramList in trainingTokenNgrams:
            for ngram in ngramList:
                tokenNgramTrainingListFile.write(ngram+'\t'+str(nr))
                tokenNgramTrainingListFile.write('\n')
                nr+=1
        tokenNgramTrainingListFile.close()
    if trainingCharacterNgrams:
        characterNgramTrainingListFileName = os.path.join(workingDirectory, 'characterNgramList.txt')
        if os.path.isfile(characterNgramTrainingListFileName):
            open(characterNgramTrainingListFileName, 'w').close()
        characterNgramTrainingListFile = codecs.open(characterNgramTrainingListFileName, 'a', 'utf8')
        nr = 1
        for ngramList in trainingCharacterNgrams:
            for ngram in ngramList:
                characterNgramTrainingListFile.write(ngram+'\t'+str(nr))
                characterNgramTrainingListFile.write('\n')
                nr+=1
        characterNgramTrainingListFile.close()

def createFeatureVectors(instanceObjectList, outputFileName, featureIndexFileName):
    '''
    Creates feature vectors for sentiment polarity classification from the instances in the Instance object list
    and writes them to the given output file path (LIBSVM sparse features format is used)
    The index number(s) for each extracted feature (set) are written to the feature index file (if desired)
    '''
    print ('Creating feature vectors...')
    pos_words, neg_words, neutral_words = load_nrc_emotion_lex('data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')
    lexiconfeatures = True
    
    if os.path.isfile(outputFileName):
        open(outputFileName, 'w').close()
    outputFile = codecs.open(outputFileName, 'a', 'utf8')
    if featureIndexFileName:
        if os.path.isfile(featureIndexFileName):
            open(featureIndexFileName, 'w').close()
        featureIndexFile = open(featureIndexFileName, 'a')
    for instanceNr, instance in enumerate(instanceObjectList):
        sentimentLabel = instance.label

        featureIndex = 0
        featureDictionary = {}
        countFloodedTokens = FeatureSelection.getInstance(featureFile).countFloodedTokens
        if countFloodedTokens:
            floodingFeature = instance.countFloodedTokens()
            if floodingFeature > 0:
                featureDictionary[featureIndex+1] = floodingFeature
            featureIndex+=1
            if instanceNr == 0:
                featureIndexInfo = 'countFloodedTokens\t%d\n' % (featureIndex)
                if featureIndexFileName:
                    featureIndexFile.write(featureIndexInfo)
        countCapitalizedTokens = FeatureSelection.getInstance(featureFile).countCapitalizedTokens
        if countCapitalizedTokens:
            capitalizationFeature = instance.countCapitalizedTokens()
            if capitalizationFeature > 0:
                featureDictionary[featureIndex+1] = capitalizationFeature
            featureIndex+=1
            if instanceNr == 0:
                featureIndexInfo = 'countCapitalizedTokens\t%d\n' % (featureIndex)
                if featureIndexFileName:
                    featureIndexFile.write(featureIndexInfo)
        countFloodedPunctuationTokens = FeatureSelection.getInstance(featureFile).countFloodedPunctuationTokens
        if countFloodedPunctuationTokens:
            floodedPunctuationFeature = instance.countTokensWithFloodedPunctuation()
            if floodedPunctuationFeature > 0:
                featureDictionary[featureIndex+1] = floodedPunctuationFeature
            featureIndex+=1
            if instanceNr == 0:
                featureIndexInfo = 'countFloodedPunctuationTokens\t%d\n' % (featureIndex)
                if featureIndexFileName:
                    featureIndexFile.write(featureIndexInfo)
        punctuationLastToken = FeatureSelection.getInstance(featureFile).punctuationLastToken
        if punctuationLastToken:
            punctuationLastTokenFeature = instance.getPunctuationLastToken()
            if punctuationLastTokenFeature:
                featureDictionary[featureIndex+1] = 1
            featureIndex+=1
            if instanceNr == 0:
                featureIndexInfo = 'punctuationLastToken\t%d\n' % (featureIndex)
                if featureIndexFileName:
                    featureIndexFile.write(featureIndexInfo)
        tokenNgramFeatureRange = FeatureSelection.getInstance(featureFile).tokenNgramFeatureRange
        if tokenNgramFeatureRange != range(0, 1):
            tokenNgramFeatureIndexList, numberOfTokenNgramFeaturesDictionary = instance.createTokenNgramFeatures(tokenNgramFeatureRange, workingDirectory)
            for index in tokenNgramFeatureIndexList:
                featureDictionary[index+featureIndex] = 1
            for n in tokenNgramFeatureRange:
                if instanceNr == 0:
                    featureIndexInfo = 'token%dgramFeatures\t%d-%d\n' % (n, featureIndex+1, featureIndex+numberOfTokenNgramFeaturesDictionary[n])
                    if featureIndexFileName:
                        featureIndexFile.write(featureIndexInfo)
                featureIndex+=numberOfTokenNgramFeaturesDictionary[n]
        characterNgramFeatureRange = FeatureSelection.getInstance(featureFile).characterNgramFeatureRange
        if characterNgramFeatureRange != range(0, 1):
            characterNgramFeatureIndexList, numberOfCharacterNgramFeaturesDictionary = instance.createCharacterNgramFeatures(characterNgramFeatureRange, workingDirectory)
            for index in characterNgramFeatureIndexList:
                featureDictionary[index+featureIndex] = 1
            for n in characterNgramFeatureRange:
                if instanceNr == 0:
                    featureIndexInfo = 'character%dgramFeatures\t%d-%d\n' % (n, featureIndex+1, featureIndex+numberOfCharacterNgramFeaturesDictionary[n])
                    if featureIndexFileName:
                        featureIndexFile.write(featureIndexInfo)
                featureIndex+=numberOfCharacterNgramFeaturesDictionary[n]
        countHashtags = FeatureSelection.getInstance(featureFile).countHashtags
        if countHashtags:
            hashtagFeature = instance.countHashtags()
            if hashtagFeature > 0:
                featureDictionary[featureIndex+1] = hashtagFeature
            featureIndex+=1
            if instanceNr == 0:
                featureIndexInfo = 'countHashtags\t%d\n' % (featureIndex)
                if featureIndexFileName:
                    featureIndexFile.write(featureIndexInfo)

        if lexiconfeatures:
            print("CHECKING INSTANCES: {}".format(instance.content))
            if instance.content:
                lexiconfeature = lex_feature_extractor(pos_words, neg_words, neutral_words, instance.content)
                featureIndex+=4
                featureDictionary[featureIndex+1] = lexiconfeature[0]
                featureDictionary[featureIndex+2] = lexiconfeature[1]
                featureDictionary[featureIndex+3] = lexiconfeature[2]
                featureDictionary[featureIndex+4] = lexiconfeature[3]

                if instanceNr == 0:
                    featureIndexInfo = 'NRC_Lexicon Features\t%d-%d\n' % (featureIndex+1, featureIndex+4)
                    if featureIndexFileName:
                        featureIndexFile.write(featureIndexInfo)
        featureVector = str(sentimentLabel) + " " + " ".join("%d:%f" % (key, featureDictionary[key]) for key in sorted(featureDictionary.keys()))
        outputFile.write(featureVector)
        outputFile.write('\n')
    outputFile.close()
    if featureIndexFileName:
        featureIndexFile.close()


def createInstanceObjectList(processed_dataset):
    '''
    Creates a list of Instance objects from the tokenized input + label
    '''
    print ('Reading instances...')
    instanceObjects = []

    #Els: read in tokenised lines
    #processed_data = []
    for item in processed_dataset:
        tokenized = []
        line = item[1]
        data = line.strip('\n')
        if data:
            all_words = word_tokenize(data)
            content = ' '.join([str(elem) for elem in all_words])
        label = item[2]
        #processed_data.append(tokenized + '\t' + str(label))
        instanceObject = Instance(content, label)
        for i, token in enumerate(content.split()):
            instanceObject.tokenDictionary[i+1] = Token(token)
        if FeatureSelection.getInstance(featureFile).normalizeInstances:
            instanceObject.tokenDictionary = instanceObject.normalizeTokens()
        instanceObjects.append(instanceObject)
    return instanceObjects



def main():

    global workingDirectory
    global featureFile
    global preproDirectory

    #Els: use small test file to test code
    with open('data/data_7000.csv', 'r') as f:
    # with open('data/test.csv', 'r') as f:
        reader = csv.reader(f)
        data_list = list(reader)

    #Code Pranay for lexion lookup
    print('Read Data From CSV...')
    pos_words, neg_words, neutral_words = load_nrc_emotion_lex('data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')
    print('NRC Emotion Lexicon Loaded...')
    processed_dataset = prep_task1(data_list)
    # print('Prepared Dataset...')
    # lex_classifier(pos_words, neg_words, neutral_words, processed_dataset)

 
    #Els: Check command Line
    try:
        if len(sys.argv) not in [5, 6]:
            raise ValueError()
    except ValueError:
        print( "Error: number of command line arguments should be 4 or 5 \
            (working directory, modus, input file, output file and optionally feature index file)")
        sys.exit("Process terminated due to incorrect number of command line arguments")
    CLArgumentsCorrect = True
    try:
        workingDirectory = sys.argv[1]
        if not os.path.isdir(workingDirectory):
            CLArgumentsCorrect = False
            raise ValueError()
    except ValueError:
        print ("Error: first command line argument (working directory) should be an existing directory")
    try:
        modus = sys.argv[2].lower()
        if modus not in ['train', 'test']:
            CLArgumentsCorrect = False
            raise ValueError()
    except ValueError:
        print ("Error: second command line argument (modus) should be train or test")
    try:
        inputFileName = sys.argv[3]
        if not os.path.isfile(inputFileName):
            CLArgumentsCorrect = False
            raise ValueError()
    except ValueError:
        print ("Error: third command line argument (input file) should be an existing file")
    outputFileName = sys.argv[4]
    featureIndexFileName = ''
    if len(sys.argv) == 6:
        featureIndexFileName = sys.argv[5]
    if not CLArgumentsCorrect:
        sys.exit("Process terminated due to invalid command line argument(s)")
    try:
        featureFile = os.path.join(workingDirectory, 'features.txt')
        if not os.path.isfile(featureFile):
            raise ValueError()
    except ValueError:
        print ("Error: working directory should contain a file features.txt")
        sys.exit("Process terminated due to missing feature file")
    try:
        preproDirectory = os.path.join(workingDirectory, 'prepro')
        if not os.path.isdir(preproDirectory):
            raise ValueError()
    except ValueError:
        print( "Error: working directory should contain a subdirectory prepro")
        sys.exit("Process terminated due to missing prepro directory in working directory")


    #instanceObjectList = processed_dataset
    instanceObjectList = createInstanceObjectList(processed_dataset)

    if modus == 'train':
        createTrainingLists(instanceObjectList)

    createFeatureVectors(instanceObjectList, outputFileName, featureIndexFileName)


if __name__ == "__main__" :
    main()
