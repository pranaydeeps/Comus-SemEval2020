#!usr/bin/env python
# encoding: utf-8

"""
FeatureSelection.py

Created by Marjan Van de Kauter on 2015-01-12.
Copyright (c) 2015 LT3. All rights reserved.

Created in the framework of the SentiFM project (Sentiment mining for Financial Markets)
Based on the code created for SemEval-2014 Task 9: Sentiment Analysis in Twitter
This class is used to create a FeatureSelection object which indicates the features that will be extracted
"""


import codecs


class FeatureSelection:


    INSTANCE = None


    def __init__(self, featureSelectionFileName):
        '''FeatureSelection constructor'''
        if self.INSTANCE is not None:
            raise ValueError("An instantiation already exists")
        self.normalizeInstances = False
        self.tokenNgramFeatureRange = range(0, 1)
        #self.lemmaNgramFeatureRange = range(0, 1)
        #self.lemmaPosUnigrams = False
        self.characterNgramFeatureRange = range(0, 1)
        self.countFloodedTokens = False
        self.countFloodedPunctuationTokens = False
        self.punctuationLastToken = False
        self.countCapitalizedTokens = False
        self.countHashtags = False
        self.sentimentLexicons = []
        self.lexiconFeatureTypes = []
        self.includeHashtokenLexiconFeatures = False
        self.modifierTypes = []
        self.PoSFeatureTypes = []
        self.NEFeatureTypes = []
        self.dependencyFeatureTypes = []
        self.PMILexicons = []
        with codecs.open(featureSelectionFileName, 'r') as featureSelectionFile:
            for line in featureSelectionFile:
                if line.strip() and not line.startswith('#'):
                    variable, value = line.strip().split('\t')
                    variable = variable.strip()
                    value = value.strip().lower()
                    if variable == 'normalizeInstances':
                        if value == 'true':
                            self.normalizeInstances = True
                    elif variable == 'tokenNgramFeatureRange':
                        lowerBound, upperBound = value.split('-')
                        self.tokenNgramFeatureRange = range(int(lowerBound), 
                            int(upperBound)+1)
                    #elif variable == 'lemmaNgramFeatureRange':
                    #    lowerBound, upperBound = value.split('-')
                    #    self.lemmaNgramFeatureRange = range(int(lowerBound),
                    #        int(upperBound)+1)
                    #elif variable == 'lemmaPosUnigrams':
                    #    if value == 'true':
                    #        self.lemmaPosUnigrams = True
                    elif variable == 'characterNgramFeatureRange':
                        lowerBound, upperBound = value.split('-')
                        self.characterNgramFeatureRange = range(int(lowerBound), 
                            int(upperBound)+1)
                    elif variable == 'countFloodedTokens':
                        if value == 'true':
                            self.countFloodedTokens = True
                    elif variable == 'countFloodedPunctuationTokens':
                        if value == 'true':
                            self.countFloodedPunctuationTokens = True
                    elif variable == 'punctuationLastToken':
                        if value == 'true':
                            self.punctuationLastToken = True
                    elif variable == 'countCapitalizedTokens':
                        if value == 'true':
                            self.countCapitalizedTokens = True
                    elif variable == 'countHashtags':
                        if value == 'true':
                            self.countHashtags = True
                    elif variable == 'sentimentLexicons':
                        if value != 'none':
                            for lexicon in value.split():
                                self.sentimentLexicons.append(lexicon)
                    elif variable == 'lexiconFeatureTypes':
                        if value != 'none':
                            for type in value.split():
                                self.lexiconFeatureTypes.append(type)
                    elif variable == 'includeHashtokenLexiconFeatures':
                        if value == 'true':
                            self.includeHashtokenLexiconFeatures = True
                    elif variable == 'modifierTypes':
                        if value != 'none':
                            for type in value.split():
                                self.modifierTypes.append(type)
                    elif variable == 'PoSFeatureTypes':
                        if value != 'none':
                            for type in value.split():
                                self.PoSFeatureTypes.append(type)
                    elif variable == 'NEFeatureTypes':
                        if value != 'none':
                            for type in value.split():
                                self.NEFeatureTypes.append(type)
                    elif variable == 'dependencyFeatureTypes':
                        if value != 'none':
                            for type in value.split():
                                self.dependencyFeatureTypes.append(type)
                    elif variable == 'PMILexicons':
                        if value != 'none':
                            for lexicon in value.split():
                                self.PMILexicons.append(lexicon)


    @classmethod
    def getInstance(cls, featureFileName):
        if cls.INSTANCE is None:
            cls.INSTANCE = FeatureSelection(featureFileName)
        return cls.INSTANCE
