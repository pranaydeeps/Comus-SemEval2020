#!usr/bin/env/python
# encoding: utf-8

"""
TrainingDictionary.py

Created by Marjan Van de Kauter on 2015-01-12.
Copyright (c) 2015 LT3. All rights reserved.

Created in the framework of the SentiFM project (Sentiment mining for Financial Markets)
Based on the code created for SemEval-2014 Task 9: Sentiment Analysis in Twitter
The following classes are used to create objects holding training dictionaries 
of token ngrams, character ngrams or dependency relations, which are necessary 
to extract token ngram, character ngram, PMI ngram and dependency relation features from a given instance
"""


import codecs
import os
from PMINgram import PMINgram


class TokenNgramDictionary:


    INSTANCE = None


    def __init__(self, workingDirectory):
        '''TokenNgramDictionary constructor'''
        if self.INSTANCE is not None:
            raise ValueError("An instantiation already exists")
        self.numberOfTokenNgramFeatures = {}
        self.tokenNgramDictionary = self.createTokenNgramDictionary(workingDirectory)


    def createTokenNgramDictionary(self, workingDirectory):
        '''Creates a dictionary of token ngrams from the training data'''
        tokenNgramDictionary = {}
        tokenNgramTrainingListFileName = os.path.join(workingDirectory, 'tokenNgramList.txt')
        with codecs.open(tokenNgramTrainingListFileName, 'r', 'utf8') as tokenNgramTrainingListFile:
            for i, line in enumerate(tokenNgramTrainingListFile):
                ngram = line.strip().split('\t')[0]
                tokenNgramDictionary[ngram] = i+1
                if len(ngram.split()) not in self.numberOfTokenNgramFeatures:
                    self.numberOfTokenNgramFeatures[len(ngram.split())] = 1
                else:
                    self.numberOfTokenNgramFeatures[len(ngram.split())]+=1
        return tokenNgramDictionary


    def getTokenNgramIndex(self, ngram):
        '''Returns the dictionary index of a token ngram if it is present in the dictionary'''
        if ngram in self.tokenNgramDictionary:
            return self.tokenNgramDictionary[ngram]


    def getNumberOfTokenNgramFeatures(self):
        '''Returns the total number of token ngram features for each n in the feature range (for feature indexing purposes)'''
        return self.numberOfTokenNgramFeatures


    @classmethod
    def getInstance(cls, workingDirectory):
        if cls.INSTANCE is None:
            cls.INSTANCE = TokenNgramDictionary(workingDirectory)
        return cls.INSTANCE


class CharacterNgramDictionary:


    INSTANCE = None


    def __init__(self, workingDirectory):
        '''CharacterNgramDictionary constructor'''
        if self.INSTANCE is not None:
            raise ValueError("An instantiation already exists")
        self.numberOfCharacterNgramFeaturesDictionary = {}
        self.characterNgramDictionary = self.createCharacterNgramDictionary(workingDirectory)


    def createCharacterNgramDictionary(self, workingDirectory):
        '''Creates a dictionary of character ngrams from the training data'''
        characterNgramDictionary = {}
        charNgramTrainingListFileName = os.path.join(workingDirectory, 'characterNgramList.txt')
        with codecs.open(charNgramTrainingListFileName, 'r', 'utf8') as charNgramTrainingListFile:
            for i, line in enumerate(charNgramTrainingListFile):
                ngram = line.strip().split('\t')[0]
                characterNgramDictionary[ngram] = i+1
                if len(ngram) not in self.numberOfCharacterNgramFeaturesDictionary:
                    self.numberOfCharacterNgramFeaturesDictionary[len(ngram)] = 1
                else:
                    self.numberOfCharacterNgramFeaturesDictionary[len(ngram)]+=1
        return characterNgramDictionary


    def getCharacterNgramIndex(self, ngram):
        '''Returns the dictionary index of a character ngram if it is present in the dictionary'''
        if ngram in self.characterNgramDictionary:
            return self.characterNgramDictionary[ngram]


    def getNumberOfCharacterNgramFeatures(self):
        '''Returns the number of character ngram features for each n in the feature range (for feature indexing purposes)'''
        return self.numberOfCharacterNgramFeaturesDictionary


    @classmethod
    def getInstance(cls, workingDirectory):
        if cls.INSTANCE is None:
            cls.INSTANCE = CharacterNgramDictionary(workingDirectory)
        return cls.INSTANCE


class DependencyRelationDictionary:


    INSTANCE = None


    def __init__(self, workingDirectory):
        '''DependencyRelationDictionary constructor'''
        if self.INSTANCE is not None:
            raise ValueError("An instantiation already exists")
        self.dependencyRelationDictionary = self.createDependencyRelationDictionary(workingDirectory)


    def createDependencyRelationDictionary(self, workingDirectory):
        '''Creates a dictionary of dependency relations from the training data'''
        dependencyRelationDictionary = {}
        dependencyRelationTrainingListFileName = os.path.join(workingDirectory, 'dependencyRelationList.txt')
        indexNr = 1
        with codecs.open(dependencyRelationTrainingListFileName, 'r', 'utf8') as depRelationTrainingListFile:
            for i, line in enumerate(depRelationTrainingListFile):
                if line.startswith('#'):
                    featureType = line.strip()[1:]
                    dependencyRelationDictionary[featureType] = {}
                else:
                    dependencyRelationDictionary[featureType][line.strip().split('\t')[0]] = indexNr
                    indexNr+=1
        return dependencyRelationDictionary


    def getRelationIndex(self, relation, relationType):
        '''Returns the dictionary index of a dependency relation if it is present in the dictionary'''
        if relation in self.dependencyRelationDictionary[relationType]:
            return self.dependencyRelationDictionary[relationType][relation]


    def getNumberOfDependencyRelationFeatures(self):
        '''Returns the number of dependency relation features for each dependency feature type (for feature indexing purposes)'''
        numbersOfDependencyRelations = {}
        for featureType in self.dependencyRelationDictionary:
            numbersOfDependencyRelations[featureType] = len(self.dependencyRelationDictionary[featureType])
        return numbersOfDependencyRelations


    @classmethod
    def getInstance(cls, workingDirectory):
        if cls.INSTANCE is None:
            cls.INSTANCE = DependencyRelationDictionary(workingDirectory)
        return cls.INSTANCE


class PMI_NRCDictionary:


    INSTANCE = None


    def __init__(self, workingDirectory):
        '''PMI_NRCDictionary constructor'''
        if self.INSTANCE is not None:
            raise ValueError("An instantiation already exists")
        self.PMI_NRCDictionary, self.numberOfFeatures = self.createPMI_NRCDictionary(workingDirectory)


    def createPMI_NRCDictionary(self, workingDirectory):
        '''
        Creates a dictionary of unigrams and bigrams with their corresponding PMI scores as calculated using NRC
        [Based on code by Els Lefever]
        '''
        PMIUniBigramDictionary = {}
        UniBigramDictionary = {}
        uniBigramTrainingListFileName = os.path.join(workingDirectory, 'uniBigramList.txt')
        PMIUnigramFileName = '/var/data/lt3/proj/SemEval2014_T9/pipeline/lib/lexicons/unigrams-pmilexicon.txt'
        PMIBigramFileName = '/var/data/lt3/proj/SemEval2014_T9/pipeline/lib/lexicons/bigrams-pmilexicon.txt'
        featureIndex = 1
        with codecs.open(uniBigramTrainingListFileName, 'r', 'utf8') as uniBigramTrainingListFile:
            for line in uniBigramTrainingListFile:
                UniBigramDictionary[line.strip()] = featureIndex
                featureIndex+=1
        with codecs.open(PMIUnigramFileName, 'r', 'utf8') as PMIUnigramFile:
            for line in PMIUnigramFile:
                fields = line.strip().split('\t')
                if fields[1] != 0:
                    if fields[0] in UniBigramDictionary:
                        PMIUniBigramDictionary[fields[0]] = PMINgram(fields[0], UniBigramDictionary[fields[0]], float(fields[1]))
        with codecs.open(PMIBigramFileName, 'r', 'utf8') as PMIBigramFile:
            for line in PMIBigramFile:
                fields = line.strip().split('\t')
                if fields[1] != 0:
                    if fields[0] in UniBigramDictionary:
                        PMIUniBigramDictionary[fields[0]] = PMINgram(fields[0], UniBigramDictionary[fields[0]], float(fields[1]))
        return PMIUniBigramDictionary, len(UniBigramDictionary)


    def getPMINgram(self, ngram):
        '''Returns the PMI Ngram object for an ngram if it present in the dictionary'''
        if ngram in self.PMI_NRCDictionary:
            return self.PMI_NRCDictionary[ngram]		


    def getNumberOfPMI_NRCFeatures(self):
        '''Returns the number of PMI NRC ngram features (for feature indexing purposes)'''
        return self.numberOfFeatures


    @classmethod
    def getInstance(cls, workingDirectory):
        if cls.INSTANCE is None:
            cls.INSTANCE = PMI_NRCDictionary(workingDirectory)
        return cls.INSTANCE


class PMI_SemEvalDictionary:


    INSTANCE = None


    def __init__(self, workingDirectory):
        '''PMI_SemEvalDictionary constructor'''
        if self.INSTANCE is not None:
            raise ValueError("An instantiation already exists")
        self.PMI_SEDictionary, self.numberOfFeatures = self.createPMI_SEDictionary(workingDirectory)


    def createPMI_SEDictionary(self, workingDirectory):
        '''
        Creates a dictionary of unigrams and their corresponding PMI scores as calculated on the SemEval training data
        [Based on code by Els Lefever]
        '''
        PMIUnigramDictionary = {}
        UnigramDictionary = {}
        PMILexiconFileName = '/var/data/lt3/proj/SemEval2014_T9/pipeline/lib/lexicons/PMIvalues_score'
        unigramTrainingListFileName = os.path.join(workingDirectory, 'unigramList.txt')
        with codecs.open(unigramTrainingListFileName, 'r', 'utf8') as unigramTrainingListFile:
            for i, line in enumerate(unigramTrainingListFile):
                UnigramDictionary[line.strip()] = i+1
        with codecs.open(PMILexiconFileName, 'r', 'utf8') as PMILexiconFile:
            for line in PMILexiconFile:
                fields = line.strip().split('\t')
                if fields[1] != 0:
                    if fields[0] in UnigramDictionary:
                        PMIUnigramDictionary[fields[0]] = PMINgram(fields[0], UnigramDictionary[fields[0]], float(fields[1]))
        return PMIUnigramDictionary, len(UnigramDictionary)


    def getPMINgram(self, ngram):
        '''Returns the PMI Ngram object for an ngram if it present in the dictionary'''
        if ngram in self.PMI_SEDictionary:
            return self.PMI_SEDictionary[ngram]   


    def getNumberOfPMI_SemEvalFeatures(self):
        '''Returns the number of PMI SemEval ngram features (for feature indexing purposes)'''
        return self.numberOfFeatures


    @classmethod
    def getInstance(cls, workingDirectory):
        if cls.INSTANCE is None:
            cls.INSTANCE = PMI_SemEvalDictionary(workingDirectory)
        return cls.INSTANCE
