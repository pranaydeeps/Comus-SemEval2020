#!usr/bin/env python
# encoding: utf-8

"""
Token.py

Created by Marjan Van de Kauter on 2015-01-12.
Copyright (c) 2015 LT3. All rights reserved.

Created in the framework of the SentiFM project (Sentiment mining for Financial Markets)
Based on the code created for SemEval-2014 Task 9: Sentiment Analysis in Twitter
Token objects are used to store all token-level (linguistic) information necessary 
to extract features for sentiment polarity classification from the instance in which the tokens occur
"""


import re


class Token:


    def __init__(self, content):
        '''Token constructor'''
        self.content = content
        self.PoSTag = ''
        self.PoSProbability = 0
        self.lemma = ''
        self.lemmaProbability = 0
        self.isNegated = False
        self.isDiminished = False
        self.isIntensified = False
        self.NEtag = 'O'
        self.NEProbability = 0


    def containsFlooding(self):
        '''
        Checks whether the token contains flooding i.e. characters repeated more than two times 
        (e.g. 'soooo')
        [Based on code by Cynthia Van Hee]
        Els: removed findall(ur")
        '''
        return re.findall("([^0-9?!.])(\1{2,})", self.content.lower())


    def isCapitalized(self):
        '''
        Checks whether all characters in the token are in uppercase
        [Based on code by Cynthia Van Hee]
        '''
        return self.content.isupper()


    def containsPunctuation(self):
        '''
        Checks whether the token contains an exclamation or question mark
        [Based on code by Cynthia Van Hee]
        '''
        return "?" in self.content or "!" in self.content


    def containsFloodedPunctuation(self):
        '''
        Checks whether the token contains a sequence of exclamation marks, 
        question marks or both exclamation and question marks
        [Based on code by Cynthia Van Hee]
        '''
        return re.findall("([?!])([?!]{1,})", self.content)


    def getCharacterNgrams(self, n):
        '''
        Extracts all character ngrams from a token		
        [Based on code by Cynthia Van Hee]
        '''
        characterNgrams = []
        lowercasedToken = self.content.lower()
        if len(lowercasedToken) >= n:
            if n == 1:
                for character in lowercasedToken:
                    characterNgrams.append(character)
            else:
                for i in range(len(lowercasedToken)-n+1):
                    ngram = "".join(token for token in lowercasedToken[i:i+n])
                    characterNgrams.append(ngram)
        characterNgrams = list(set(characterNgrams))
        return characterNgrams


    def setPoSTag(self, PoSTag):
        '''
        Sets the PoS tag of the token object
        (and simultaneously maps it to one of the main categories of the CGN tagset)
        '''
        if PoSTag.startswith('ADJ'):
            self.PoSTag = 'ADJ'
        elif PoSTag.startswith('BW'):
            self.PoSTag = 'BW'
        elif PoSTag.startswith('LET'):
            self.PoSTag = 'LET'
        elif PoSTag.startswith('LID'):
            self.PoSTag = 'LID'
        elif PoSTag.startswith('N(soort'):
            self.PoSTag = 'N'
        elif PoSTag.startswith('N(eigen'):
            self.PoSTag = 'NP'
        elif PoSTag.startswith('SPEC(deeleigen'):
            self.PoSTag = 'NP'
        elif PoSTag.startswith('SPEC(afk'):
            self.PoSTag = 'AFK'
        elif PoSTag.startswith('SPEC(afgebr'):
            self.PoSTag = 'AFGEBR'
        elif PoSTag.startswith('SPEC(vreemd'):
            self.PoSTag = 'VREEMD'
        elif PoSTag.startswith('SPEC(enof'):
            self.PoSTag = 'ENOF'
        elif PoSTag.startswith('SPEC(meta'):
            self.PoSTag = 'META'
        elif PoSTag.startswith('SPEC(symb'):
            self.PoSTag = 'SYMB'
        elif PoSTag.startswith('TSW'):
            self.PoSTag = 'TSW'
        elif PoSTag.startswith('TW'):
            self.PoSTag = 'TW'
        elif PoSTag.startswith('VG'):
            self.PoSTag = 'VG'
        elif PoSTag.startswith('VNW'):
            self.PoSTag = 'VNW'
        elif PoSTag.startswith('VZ'):
            self.PoSTag = 'VZ'
        elif PoSTag.startswith('WW'):
            self.PoSTag = 'WW'
