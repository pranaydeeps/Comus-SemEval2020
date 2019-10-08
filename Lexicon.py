#!usr/bin/env python
# encoding: utf-8

"""
Lexicon.py

Created by Marjan Van de Kauter on 2015-01-12.
Copyright (c) 2015 LT3. All rights reserved.

Created in the framework of the SentiFM project (Sentiment mining for Financial Markets)
Based on the code created for SemEval-2014 Task 9: Sentiment Analysis in Twitter
The following classes are used to create Lexicon objects in which sentiment words (incl. emoticons)
from a certain lexicon can be stored with their corresponding polarity values (and PoS tags, if available)
[Based on code by Orphee De Clercq]
"""


from __future__ import division
import codecs
from bs4 import BeautifulSoup
from SentimentWord import SentimentWord


class Lexicon:


    INSTANCE = None


    def __init__(self):
        '''Lexicon constructor'''
        if self.INSTANCE is not None:
            raise ValueError("An instantiation already exists")
        self.sentimentWordDictionary = {}


    def getPolarityValue(self, tokenObject, hashtokenOnly):
        '''Returns the sentiment polarity value of a token if it is present in the sentiment lexicon'''
        token = tokenObject.content.lower()
        if hashtokenOnly:
            token = token.strip('#')
        PoSTag = tokenObject.PoSTag
        lemma = tokenObject.lemma.lower()
        isNegated = tokenObject.isNegated
        isDiminished = tokenObject.isDiminished
        isIntensified = tokenObject.isIntensified
        if token in self.sentimentWordDictionary:
            matchingPoSTag = True
            if self.sentimentWordDictionary[token].PoSTag:
                if self.sentimentWordDictionary[token].PoSTag != PoSTag:
                    matchingPoSTag = False
            if matchingPoSTag:
                polarityValue = self.sentimentWordDictionary[token].sentimentPolarityValue
                if isNegated:
                    polarityValue = -polarityValue
                if isDiminished:
                    polarityValue = polarityValue/2
                if isIntensified:
                    polarityValue = polarityValue*2
                return polarityValue
        elif lemma in self.sentimentWordDictionary:
            matchingPoSTag = True
            if self.sentimentWordDictionary[lemma].PoSTag:
                if self.sentimentWordDictionary[lemma].PoSTag != PoSTag:
                    matchingPoSTag = False
            if matchingPoSTag:
                polarityValue = self.sentimentWordDictionary[lemma].sentimentPolarityValue
                if isNegated:
                    polarityValue = -polarityValue
                if isDiminished:
                    polarityValue = polarityValue/2
                if isIntensified:
                    polarityValue = polarityValue*2
                return polarityValue


    @classmethod
    def getInstance(cls):
        if cls.INSTANCE is None:
            cls.INSTANCE = Lexicon()
        return cls.INSTANCE
	

class PatternLexicon(Lexicon):


    INSTANCE = None


    def __init__(self):
        '''PatternLexicon constructor'''
        if self.INSTANCE is not None:
            raise ValueError("An instantiation already exists")
        self.sentimentWordDictionary = self.createPatternSentimentWordDictionary()


    def createPatternSentimentWordDictionary(self):
        '''
        Creates a dictionary of sentiment words and their corresponding polarity values 
        from the Pattern lexicon
        '''
        sentimentWordPreliminaryDictionary = {}
        lexiconFileName = './lib/lexicons/pattern_nl-sentiment.xml'
        lexicon = BeautifulSoup(open(lexiconFileName), "xml")
        for word in lexicon.find_all('word'):
            wordForm = word['form']
            PoSTag = word['pos']
            if PoSTag == 'VB':
                PoSTag = 'WW'
            if PoSTag == 'NN':
                PoSTag = 'N'
            if PoSTag == 'FW':
                PoSTag = 'N'
            if PoSTag == 'UH':
                PoSTag = 'TSW'
            if PoSTag == 'JJ':
                PoSTag = 'ADJ'
            if PoSTag == 'RB':
                PoSTag == 'BW'
            polarityValue = float(word['polarity'])
            if (wordForm, PoSTag) not in sentimentWordPreliminaryDictionary:
                sentimentWordPreliminaryDictionary[(wordForm, PoSTag)] = []
            sentimentWordPreliminaryDictionary[(wordForm, PoSTag)].append(polarityValue)
        sentimentWordDictionary = {}
        for (wordForm, PoSTag) in sentimentWordPreliminaryDictionary:
            polarityValue = 0
            for value in sentimentWordPreliminaryDictionary[(wordForm, PoSTag)]:
                polarityValue += value
            polarityValue = polarityValue / len(sentimentWordPreliminaryDictionary[(wordForm, PoSTag)])
            if polarityValue >= 0.01 or polarityValue <=-0.01:
                sentimentWordDictionary[wordForm] = SentimentWord(wordForm, PoSTag, polarityValue)
        return sentimentWordDictionary


    @classmethod
    def getInstance(cls):
        if cls.INSTANCE is None:
            cls.INSTANCE = PatternLexicon()
        return cls.INSTANCE


class PatternManLexicon(Lexicon):


    INSTANCE = None


    def __init__(self):
        '''PatternManLexicon constructor'''
        if self.INSTANCE is not None:
            raise ValueError("An instantiation already exists")
        self.sentimentWordDictionary = self.createPatternManSentimentWordDictionary()


    def createPatternManSentimentWordDictionary(self):
        '''
        Creates a dictionary of sentiment words and their corresponding polarity values
        from the Pattern manual lexicon
        '''
        sentimentWordPreliminaryDictionary = {}
        lexiconFileName = './lib/lexicons/pattern_nl-sentiment.xml'
        lexicon = BeautifulSoup(open(lexiconFileName), "xml")
        for word in lexicon.find_all('word'):
            wordForm = word['form']
            PoSTag = word['pos']
            confidence = float(word['confidence'])
            if PoSTag == 'VB':
                PoSTag = 'WW'
            if PoSTag == 'NN':
                PoSTag = 'N'
            if PoSTag == 'FW':
                PoSTag = 'N'
            if PoSTag == 'UH':
                PoSTag = 'TSW'
            if PoSTag == 'JJ':
                PoSTag = 'ADJ'
            if PoSTag == 'RB':
                PoSTag == 'BW'
            polarityValue = float(word['polarity'])
            if confidence == 1:
                if (wordForm, PoSTag) not in sentimentWordPreliminaryDictionary:
                    sentimentWordPreliminaryDictionary[(wordForm, PoSTag)] = []
                sentimentWordPreliminaryDictionary[(wordForm, PoSTag)].append(polarityValue)
        sentimentWordDictionary = {}
        for (wordForm, PoSTag) in sentimentWordPreliminaryDictionary:
            polarityValue = 0
            for value in sentimentWordPreliminaryDictionary[(wordForm, PoSTag)]:
                polarityValue += value
            polarityValue = polarityValue / len(sentimentWordPreliminaryDictionary[(wordForm, PoSTag)])
            if polarityValue >= 0.01 or polarityValue <=-0.01:
                sentimentWordDictionary[wordForm] = SentimentWord(wordForm, PoSTag, polarityValue)
        return sentimentWordDictionary


    @classmethod
    def getInstance(cls):
        if cls.INSTANCE is None:
            cls.INSTANCE = PatternManLexicon()
        return cls.INSTANCE


class DuomanLexicon(Lexicon):


    INSTANCE = None


    def __init__(self):
        '''DuomanLexicon constructor'''
        if self.INSTANCE is not None:
            raise ValueError("An instantiation already exists")
        self.sentimentWordDictionary = self.createDuomanSentimentWordDictionary()


    def createDuomanSentimentWordDictionary(self):
        '''
        Creates a dictionary of sentiment words and their corresponding polarity values 
        from the Duoman lexicon
        '''
        sentimentWordDictionary = {}
        lexiconFileName = './lib/lexicons/full_lexicon.txt'
        with codecs.open(lexiconFileName, 'r', 'utf8') as lexiconFile:		
            for line in lexiconFile:
                wordPos, polarityValue = line.strip().split('\t')
                wordForm = wordPos[:wordPos.rfind(' ')]
                polarityValue = float(polarityValue)
                PoSTag = "n"
                if polarityValue >= 0.01 or polarityValue <= -0.01:
                    sentimentWordDictionary[wordForm] = SentimentWord(wordForm, PoSTag, polarityValue)
        return sentimentWordDictionary

    @classmethod
    def getInstance(cls):
        if cls.INSTANCE is None:
            cls.INSTANCE = DuomanLexicon()
        return cls.INSTANCE


class DuomanManLexicon(Lexicon):


    INSTANCE = None


    def __init__(self):
        '''DuomanManLexicon constructor'''
        if self.INSTANCE is not None:
            raise ValueError("An instantiation already exists")
        self.sentimentWordDictionary = self.createDuomanManSentimentWordDictionary()


    def createDuomanManSentimentWordDictionary(self):
        '''
        Creates a dictionary of sentiment words and their corresponding polarity values
        from the Duoman manual lexicon
        '''
        sentimentWordDictionary = {}
        lexiconFileName = './lib/lexicons/assessments_both_agreed.txt'
        with codecs.open(lexiconFileName, 'r', 'utf8') as lexiconFile:
            for line in lexiconFile:
                wordPos, polarityValue = line.strip().split('\t')
                wordForm = wordPos[:wordPos.rfind(' ')]
                PoSTag = wordPos[wordPos.rfind(' ')+1:]
                if PoSTag == 'a':
                    PoSTag = 'ADJ'
                if PoSTag == 'n':
                    PoSTag = 'N'
                if PoSTag == 'v':
                    PoSTag = 'WW'
                if PoSTag == 'b':
                    PoSTag = 'BW'
                polarityValue = float(polarityValue.replace(' ', ''))/2
                if polarityValue >= 0.01 or polarityValue <= -0.01:
                    sentimentWordDictionary[wordForm] = SentimentWord(wordForm, PoSTag, polarityValue)
        return sentimentWordDictionary

    @classmethod
    def getInstance(cls):
        if cls.INSTANCE is None:
            cls.INSTANCE = DuomanManLexicon()
        return cls.INSTANCE


class BounceLexicon(Lexicon):


    INSTANCE = None


    def __init__(self):
        '''BounceLexicon constructor'''
        if self.INSTANCE is not None:
            raise ValueError("An instantiation already exists")
        self.sentimentWordDictionary = self.createBounceEmoticonDictionary()


    def createBounceEmoticonDictionary(self):
        '''
        Creates a dictionary of emoticons and their corresponding polarity values
        from the Bounce emoticon lexicon
        '''
        emoticonDictionary = {}
        lexiconFileName = './lib/lexicons/emoticon_polarity.tsv'
        with codecs.open(lexiconFileName, 'r', 'utf8') as lexiconFile:
            for line in lexiconFile:
                fields = line.strip().split("\t")
                emoticon = fields[0]
                polarityValue = int(fields[1])
                emoticonDictionary[emoticon] = SentimentWord(emoticon, "", polarityValue)
        return emoticonDictionary


    def getPolarityValue(self, tokenObject, hashtokenOnly):
        '''Returns the sentiment polarity value of a token if it is present in the sentiment lexicon'''
        token = tokenObject.content
        if hashtokenOnly:
            token = token.strip('#')
        PoSTag = tokenObject.PoSTag
        if token in self.sentimentWordDictionary:
            matchingPoSTag = True
            if self.sentimentWordDictionary[token].PoSTag:
                if self.sentimentWordDictionary[token].PoSTag != PoSTag:
                    matchingPoSTag = False
            if matchingPoSTag:
                return self.sentimentWordDictionary[token].sentimentPolarityValue


    @classmethod
    def getInstance(cls):
        if cls.INSTANCE is None:
            cls.INSTANCE = BounceLexicon()
        return cls.INSTANCE
