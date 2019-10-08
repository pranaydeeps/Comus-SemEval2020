#!usr/bin/env python
# encoding: utf-8

"""
extractFeatureVectors.py

Created by Marjan Van de Kauter on 2015-01-12.
Copyright (c) 2015 LT3. All rights reserved.

Created in the framework of the SentiFM project (Sentiment mining for Financial Markets)
Based on the code created for SemEval-2014 Task 9: Sentiment Analysis in Twitter
Creates feature vectors for sentiment polarity classification based on a given input file

Takes 4 or 5 arguments:

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
* Lemma ngram features
* Lemma + PoS tag ngram features
* Character ngram features
* PMI token ngram features
* Flooding features
* Capitalization features
* Punctuation features
* Hashtag features
* Sentiment lexicon and emoticon features
* Part-of-speech tag features
* Named Entity features
* Dependency relation features

All feature vectors are written to the given output file path
"""


import sys
import os
import codecs
import subprocess
from Instance import Instance
from Token import Token
from FeatureSelection import FeatureSelection


workingDirectory = ''
featureFile = ''
preproDirectory = ''


def createInstanceObjectList(inputFileName):
    '''
    Creates a list of Instance objects from the input file, which contain all the (linguistic) information
    needed to extract the features for sentiment polarity classification
    '''
    print ('Reading instances...')
    instanceObjects = []
    with codecs.open(inputFileName, 'r', 'utf8') as inputFile:
    	for line in inputFile:
            content, label = line.strip().split('\t')
            instanceObject = Instance(content, label)
            for i, token in enumerate(content.split()):
                instanceObject.tokenDictionary[i+1] = Token(token)
            if FeatureSelection.getInstance(featureFile).normalizeInstances:
                instanceObject.tokenDictionary = instanceObject.normalizeTokens()
            instanceObjects.append(instanceObject)
    return instanceObjects



def createTrainingLists(instanceObjectList):
    '''
    Extracts lists of token ngrams, character ngrams, lemma ngrams, lemma+PoStag ngrams and dependency relations 
    from the training data and stores them in text files to be used to create the token ngram, character ngram, 
    lemma ngram, lemma+PoStag ngram, dependency relation and PMI ngram features
    '''
    print ('Creating training dictionaries...')
    tokenNgramFeatureRange = FeatureSelection.getInstance(featureFile).tokenNgramFeatureRange	
    characterNgramFeatureRange = FeatureSelection.getInstance(featureFile).characterNgramFeatureRange
    PMILexicons = FeatureSelection.getInstance(featureFile).PMILexicons
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
        if 'nrc' in PMILexicons:
            for ngramList in instanceObject.getTokenNgrams(range(1,3)):
                for ngram in ngramList:
                    trainingTokenUniBigrams.append(ngram)
        if 'se' in PMILexicons:
            for ngramList in instanceObject.getTokenNgrams(range(1,2)):
                for ngram in ngramList:
                    trainingTokenUnigrams.append(ngram)
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
    if 'nrc' in PMILexicons:
        tokenUniBigramTrainingListFileName = os.path.join(workingDirectory, 'uniBigramList.txt')
        if os.path.isfile(tokenUniBigramTrainingListFileName):
            open(tokenUniBigramTrainingListFileName, 'w').close()
        tokenUniBigramTrainingListFile = codecs.open(tokenUniBigramTrainingListFileName, 'a', 'utf8')
        for ngram in trainingTokenUniBigrams:
            tokenUniBigramTrainingListFile.write(ngram)
            tokenUniBigramTrainingListFile.write('\n')
        tokenUniBigramTrainingListFile.close()
    if 'se' in PMILexicons:
        tokenUnigramTrainingListFileName = os.path.join(workingDirectory, 'unigramList.txt')
        if os.path.isfile(tokenUnigramTrainingListFileName):
            open(tokenUnigramTrainingListFileName, 'w').close()
        tokenUnigramTrainingListFile = codecs.open(tokenUnigramTrainingListFileName, 'a', 'utf8')
        for ngram in trainingTokenUnigrams:
            tokenUnigramTrainingListFile.write(ngram)
            tokenUnigramTrainingListFile.write('\n')
        tokenUnigramTrainingListFile.close()
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
    if os.path.isfile(outputFileName):
        open(outputFileName, 'w').close()
    outputFile = codecs.open(outputFileName, 'a', 'utf8')
    if featureIndexFileName:
        if os.path.isfile(featureIndexFileName):
            open(featureIndexFileName, 'w').close()
        featureIndexFile = open(featureIndexFileName, 'a')
    for instanceNr, instance in enumerate(instanceObjectList):
        sentimentLabel = instance.label
        if sentimentLabel == '-':
            sentimentLabel = '1'
        if sentimentLabel == '0':
            sentimentLabel = '2'
        if sentimentLabel == '+':
            sentimentLabel = '3'
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
        sentimentLexicons = FeatureSelection.getInstance(featureFile).sentimentLexicons
        lexiconFeatureTypes = FeatureSelection.getInstance(featureFile).lexiconFeatureTypes
        includeHashtokenLexiconFeatures = FeatureSelection.getInstance(featureFile).includeHashtokenLexiconFeatures
        if sentimentLexicons:
            for lexicon in sentimentLexicons:
                if 'postokens' in lexiconFeatureTypes and lexicon not in ["duoman","patternman"]:
                    nrPosTokens = instance.countPositiveTokens(lexicon, False)
                    if nrPosTokens > 0:
                        featureDictionary[featureIndex+1] = nrPosTokens
                    featureIndex+=1
                    if instanceNr == 0:
                        featureIndexInfo = '%s-nrPosToken\t%d\n' % (lexicon, featureIndex)
                        if featureIndexFileName:
                            featureIndexFile.write(featureIndexInfo)
                if 'negtokens' in lexiconFeatureTypes:
                    nrNegTokens = instance.countNegativeTokens(lexicon, False)
                    if nrNegTokens > 0:
	                    featureDictionary[featureIndex+1] = nrNegTokens
                    featureIndex+=1
                    if instanceNr == 0:
                        featureIndexInfo = '%s-nrNegTokens\t%d\n' % (lexicon, featureIndex)
                        if featureIndexFileName:
                            featureIndexFile.write(featureIndexInfo)
                if 'neuttokens' in lexiconFeatureTypes and lexicon not in ["pattern"]:
                    nrNeutTokens = instance.countNeutralTokens(lexicon, False)
                    if nrNeutTokens > 0:
                        featureDictionary[featureIndex+1] = nrNeutTokens
                    featureIndex+=1
                    if instanceNr == 0:
                        featureIndexInfo = '%s-nrNeutTokens\t%d\n' % (lexicon, featureIndex)
                        if featureIndexFileName:
                            featureIndexFile.write(featureIndexInfo)
                if 'overallvalue' in lexiconFeatureTypes:
                    overallValue = instance.calculatePolarityValue(lexicon, False)
                    if overallValue != 0:
                        featureDictionary[featureIndex+1] = overallValue
                    featureIndex+=1
                    if instanceNr == 0:
                        featureIndexInfo = '%s-overallValue\t%d\n' % (lexicon, featureIndex)
                        if featureIndexFileName:
                            featureIndexFile.write(featureIndexInfo)
            if includeHashtokenLexiconFeatures:
                for lexicon in sentimentLexicons:
                    if 'postokens' in lexiconFeatureTypes:
                        nrPosTokens = instance.countPositiveTokens(lexicon, True)
                        if nrPosTokens > 0:
                            featureDictionary[featureIndex+1] = nrPosTokens
                        featureIndex+=1
                        if instanceNr == 0:
                            featureIndexInfo = '%s-nrPosTokens-hashtokens\t%d\n' % (lexicon, featureIndex)
                            if featureIndexFileName:
                                featureIndexFile.write(featureIndexInfo)
                    if 'negtokens' in lexiconFeatureTypes:
                        nrNegTokens = instance.countNegativeTokens(lexicon, True)
                        if nrNegTokens > 0:
                            featureDictionary[featureIndex+1] = nrNegTokens
                        featureIndex+=1
                        if instanceNr == 0:
                            featureIndexInfo = '%s-nrNegTokens-hashtokens\t%d\n' % (lexicon, featureIndex)
                            if featureIndexFileName:
                                featureIndexFile.write(featureIndexInfo)
                    if 'neuttokens' in lexiconFeatureTypes:
                        nrNeutTokens = instance.countNeutralTokens(lexicon, True)
                        if nrNeutTokens > 0:
                            featureDictionary[featureIndex+1] = nrNeutTokens
                        featureIndex+=1
                        if instanceNr == 0:
                            featureIndexInfo = '%s-nrNeutTokens-hashtokens\t%d\n' % (lexicon, featureIndex)
                            if featureIndexFileName:
                                featureIndexFile.write(featureIndexInfo)
                    if 'overallvalue' in lexiconFeatureTypes:
                        overallValue = instance.calculatePolarityValue(lexicon, True)
                        if overallValue != 0:
                            featureDictionary[featureIndex+1] = overallValue
                        featureIndex+=1
                        if instanceNr == 0:
                            featureIndexInfo = '%s-overallValue-hashtokens\t%d\n' % (lexicon, featureIndex)
                            if featureIndexFileName:
                                featureIndexFile.write(featureIndexInfo)
        PMILexicons = FeatureSelection.getInstance(featureFile).PMILexicons
        for lexicon in PMILexicons:
            PMINgramFeatureDictionary, numberOfPMIFeatures = instance.createPMINgramFeatures(lexicon, workingDirectory)
            for index in PMINgramFeatureDictionary:
                featureDictionary[featureIndex+index] = PMINgramFeatureDictionary[index]
            featureIndex+=numberOfPMIFeatures
            if instanceNr == 0:
                featureIndexInfo = '%s-PMIFeatures\t%d-%d\n' % (lexicon, featureIndex+1-numberOfPMIFeatures, featureIndex)
                if featureIndexFileName:
                    featureIndexFile.write(featureIndexInfo)
        featureVector = sentimentLabel + " " + " ".join("%d:%f" % (key, featureDictionary[key]) for key in sorted(featureDictionary.keys()))
        outputFile.write(featureVector)
        outputFile.write('\n')
    outputFile.close()
    if featureIndexFileName:
        featureIndexFile.close()


def main():

    global workingDirectory
    global featureFile
    global preproDirectory

    # checks if the given command line arguments are valid
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

    instanceObjectList = createInstanceObjectList(inputFileName)

    if modus == 'train':
        createTrainingLists(instanceObjectList)

    createFeatureVectors(instanceObjectList, outputFileName, featureIndexFileName)


if __name__ == "__main__" :
    main()
