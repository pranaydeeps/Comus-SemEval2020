#!usr/bin/env python
# encoding: utf-8

"""
Normalizer.py

Created by Marjan Van de Kauter on 2014-03-11.
Copyright (c) 2014 LT3. All rights reserved.

Created in the framework of SemEval-2014 Task 9: Sentiment Analysis in Twitter
The Normalizer class is used to normalize a token
(i.e. replace URL's, mentions, @-replies, abbreviations, etc.)
[Based on the normalization script of Cynthia Van Hee]
"""


import codecs
import re
from Token import Token


class Normalizer:


    INSTANCE = None


    def __init__(self):
        '''Normalizer constructor'''
        if self.INSTANCE is not None:
            raise ValueError("An instantiation already exists")
        self.abbreviationDictionary = self.createAbbreviationDictionary()

			
    def createAbbreviationDictionary(self):
        '''
        Creates a dictionary of abbreviations and their corresponding full word forms 
        from the abbreviation list file
        '''
        abbreviationDictionary = {}
        abbrListFileName = '/var/data/lt3/proj/SemEval2014_T9/pipeline/lib/lexicons/abbreviationList.txt'
        with codecs.open(abbrListFileName, 'r', 'utf8') as abbreviationListFile:
            for line in abbreviationListFile:
                abbreviation, fullWordForm = line.strip().split('\t')
                abbreviationDictionary[abbreviation] = fullWordForm
        return abbreviationDictionary


    def normalizeToken(self, token):
        '''
        Replaces URLs, mentions and abbreviations in a token 
        and creates a list of normalized Token objects
        '''
        normalizedTokenObjects = []
        # URLs are replaced
        if re.findall(r"http:/\S+", token.content.lower()) or re.findall(r"www\.\S+", token.content.lower()):
            normalizedTokenObjects.append(Token("http://someurl"))
        # mentions are replaced
        elif re.findall(r"\@\S+", token.content.lower()):
            normalizedTokenObjects.append(Token("@someuser"))
        # abbreviations are replaced
        #elif token.content.lower() in self.abbreviationDictionary:
        #    for normalizedToken in self.abbreviationDictionary[token.content.lower()].split():
        #        normalizedTokenObjects.append(Token(normalizedToken))
        else:
            normalizedTokenObjects.append(Token(token.content))
        return normalizedTokenObjects


    @classmethod
    def getInstance(cls):
        if cls.INSTANCE is None:
            cls.INSTANCE = Normalizer()
        return cls.INSTANCE
