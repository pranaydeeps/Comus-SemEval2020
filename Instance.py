#!usr/bin/env python
# encoding: utf-8

"""
Instance.py

Created by Marjan Van de Kauter on 2015-01-12.
Copyright (c) 2015 LT3. All rights reserved.

Created in the framework of the SentiFM project (Sentiment mining for Financial Markets)
Based on the code created for SemEval-2014 Task 9: Sentiment Analysis in Twitter
Instance objects are used to store the (linguistic) properties of instances necessary 
to extract features for sentiment polarity classification from these instances
"""


from __future__ import division
import codecs
from TrainingDictionary import TokenNgramDictionary,  \
    CharacterNgramDictionary, DependencyRelationDictionary, PMI_NRCDictionary, PMI_SemEvalDictionary
from Lexicon import DuomanLexicon, DuomanManLexicon, PatternLexicon, PatternManLexicon, BounceLexicon
from Normalizer import Normalizer


class Instance:


    def __init__(self, content, label):
        '''Instance constructor'''	
        self.content = content
        self.label = label
        self.tokenDictionary = {}
        self.dependencyRelations = []


    def normalizeTokens(self):
        '''Normalizes each token in the instance and stores the normalized tokens in the token dictionary'''
        normalizedTokenDictionary = {}
        normalizedTokenId = 1
        for tokenObject in self.tokenDictionary.values():
            normalizedTokenObjects = Normalizer.getInstance().normalizeToken(tokenObject)
            for normalizedTokenObject in normalizedTokenObjects:
                normalizedTokenDictionary[normalizedTokenId] = normalizedTokenObject
                normalizedTokenId+=1
        return normalizedTokenDictionary


    def getTokenizedInstance(self):
        '''Produces a tokenized string of the instance'''        
        return " ".join(token.content for token in self.tokenDictionary.values())


    def detectNegation(self):
        '''Detects negators in the instance and sets the value of the isNegated attribute of the following 4 tokens to True'''
        for tokenNr in self.tokenDictionary:
            if self.tokenDictionary[tokenNr].content.lower() in ["niet", "geen", "nooit", "noch", "zonder", "nergens", "niets", "niemand"]:
                if (tokenNr+1) in self.tokenDictionary:
                    self.tokenDictionary[tokenNr+1].isNegated = True
                if (tokenNr+2) in self.tokenDictionary:
                    self.tokenDictionary[tokenNr+2].isNegated = True
                if (tokenNr+3) in self.tokenDictionary:
                    self.tokenDictionary[tokenNr+3].isNegated = True
                if (tokenNr+4) in self.tokenDictionary:
                    self.tokenDictionary[tokenNr+4].isNegated = True


    def detectDiminishers(self):
        '''Detects diminishers in the instance and sets the value of the isDiminished attribute of the following 4 tokens to True'''
        for tokenNr in self.tokenDictionary:
            if self.tokenDictionary[tokenNr].content.lower() in ['bijna']: #lijst aan te vullen (vertalen uit EN)
                if (tokenNr+1) in self.tokenDictionary:
                    self.tokenDictionary[tokenNr+1].isDiminished = True
                if (tokenNr+2) in self.tokenDictionary:
                    self.tokenDictionary[tokenNr+2].isDiminished = True
                if (tokenNr+3) in self.tokenDictionary:
                    self.tokenDictionary[tokenNr+3].isDiminished = True
                if (tokenNr+4) in self.tokenDictionary:
                    self.tokenDictionary[tokenNr+4].isDiminished = True


    def detectIntensifiers(self):
        '''Detects intensifiers in the instance and sets the value of the isIntensified attribute of the following 4 tokens to True'''
        for tokenNr in self.tokenDictionary:
            if self.tokenDictionary[tokenNr].content.lower() in ['heel']: #lijst aan te vullen (vertalen uit EN)
                if (tokenNr+1) in self.tokenDictionary:
                    self.tokenDictionary[tokenNr+1].isIntensified = True
                if (tokenNr+2) in self.tokenDictionary:
                    self.tokenDictionary[tokenNr+2].isIntensified = True
                if (tokenNr+3) in self.tokenDictionary:
                    self.tokenDictionary[tokenNr+3].isIntensified = True
                if (tokenNr+4) in self.tokenDictionary:
                    self.tokenDictionary[tokenNr+4].isIntensified = True


    def countFloodedTokens(self):
        '''Counts the number of tokens in the instance which contain flooding'''	
        counter = 0
        for token in self.tokenDictionary.values():
            if token.containsFlooding():
                counter+=1
        if len(self.tokenDictionary) > 0:
            counter = counter/len(self.tokenDictionary)
        return counter


    def countCapitalizedTokens(self):
        '''Counts the number of capitalized tokens in the instance'''		
        counter = 0
        for token in self.tokenDictionary.values():
            if token.isCapitalized():
                counter+=1
        if len(self.tokenDictionary) > 0:
            counter = counter/len(self.tokenDictionary)
        return counter


    def countTokensWithFloodedPunctuation(self):
        '''Counts the number of tokens in the instance which contain flooded punctuation'''
        counter = 0
        for token in self.tokenDictionary.values():
            if token.containsFloodedPunctuation():
                counter+=1
        if len(self.tokenDictionary) > 0:
            counter = counter/len(self.tokenDictionary)
        return counter


    def getPunctuationLastToken(self):
        '''Checks whether the last token of the instance contains an exclamation or question mark'''
        if self.tokenDictionary:
            return self.tokenDictionary[len(self.tokenDictionary)].containsPunctuation()
        return False

	
    def countHashtags(self):
        '''
        Counts the number of hashtags in the instance
        [Based on code by Orphee De Clercq]
        '''
        counter = 0
        for token in self.tokenDictionary.values():
            if token.content.startswith('#'):
                counter+=1
        if len(self.tokenDictionary) > 0:
            counter = counter/len(self.tokenDictionary)
        return counter


    def getTokenPolarityValue(self, token, hashtokenOnly, lexicon):
        '''
        Returns the sentiment polarity value of a token if it is present in the sentiment lexicon
        (in case of hashtags: returns the polarity value for the token with or without hashtag symbol)
        '''
        if lexicon == 'duoman':
            return DuomanLexicon.getInstance().getPolarityValue(token, hashtokenOnly)
        if lexicon == 'duomanman':
            return DuomanManLexicon.getInstance().getPolarityValue(token, hashtokenOnly)
        elif lexicon == 'pattern':
            return PatternLexicon.getInstance().getPolarityValue(token, hashtokenOnly)
        elif lexicon == 'patternman':
            return PatternManLexicon.getInstance().getPolarityValue(token, hashtokenOnly)
        elif lexicon == 'bounce':
            return BounceLexicon.getInstance().getPolarityValue(token, hashtokenOnly)

   
    def countPositiveTokens(self, lexicon, hashtokensOnly):
        '''
        Counts the number of positive words in the instance 
        (taking into account all tokens or only hashtag tokens)
        [Based on code by Orphee De Clercq]
        '''
        counter = 0
        for token in self.tokenDictionary.values():
            if hashtokensOnly:
                if token.content.startswith('#'):
                    tokenPolarityValue = self.getTokenPolarityValue(token, hashtokensOnly, lexicon)
                    if tokenPolarityValue != None and tokenPolarityValue > 0:
                        counter+=1
            else:
                tokenPolarityValue = self.getTokenPolarityValue(token, hashtokensOnly, lexicon)
                if tokenPolarityValue != None and tokenPolarityValue > 0:
                    counter+=1
        if not hashtokensOnly and lexicon not in ['bounce', 'se']:
            if len(self.tokenDictionary) > 0:
                counter = counter/len(self.tokenDictionary)
        return counter

		
    def countNegativeTokens(self, lexicon, hashtokensOnly):
        '''
        Counts the number of negative words in the instance 
        (taking into account all tokens or only hashtag tokens)
        [Based on code by Orphee De Clercq]
        '''
        counter = 0
        for token in self.tokenDictionary.values():
            if hashtokensOnly:
                if token.content.startswith('#'):
                    tokenPolarityValue = self.getTokenPolarityValue(token, hashtokensOnly, lexicon)
                    if tokenPolarityValue != None and tokenPolarityValue < 0:
                        counter+=1
            else:
                tokenPolarityValue = self.getTokenPolarityValue(token,hashtokensOnly, lexicon)
                if tokenPolarityValue != None and tokenPolarityValue < 0:
                    counter+=1
        if not hashtokensOnly and lexicon not in ['bounce', 'se']:
            if len(self.tokenDictionary) > 0:
                counter = counter/len(self.tokenDictionary)
        return counter


    def countNeutralTokens(self, lexicon, hashtokensOnly):
        '''
        Counts the number of neutral words in the instance 
        (taking into account all tokens or only hashtag tokens)
        [Based on code by Orphee De Clercq]
        '''
        counter = 0
        for token in self.tokenDictionary.values():
            if hashtokensOnly:
                if token.content.startswith('#'):
                    tokenPolarityValue = self.getTokenPolarityValue(token, hashtokensOnly, lexicon)
                    if tokenPolarityValue != None and tokenPolarityValue == 0:
                        counter+=1
            else:
                tokenPolarityValue = self.getTokenPolarityValue(token, hashtokensOnly, lexicon)
                if tokenPolarityValue != None and tokenPolarityValue == 0:
                    counter+=1
        if not hashtokensOnly and lexicon not in ['bounce', 'se']:
            if len(self.tokenDictionary) > 0:
                counter = counter/len(self.tokenDictionary)
        return counter


    def calculatePolarityValue(self, lexicon, hashtokensOnly):
        '''
        Calculates a sentiment polarity value for the full instance 
        (taking into account all tokens or only hashtag tokens)
        [Based on code by Orphee De Clercq]
        '''
        polarityValue = 0
        for token in self.tokenDictionary.values():
            if hashtokensOnly:
                if token.content.startswith('#'):
                    tokenPolarityValue = self.getTokenPolarityValue(token, hashtokensOnly, lexicon)
                    if tokenPolarityValue != None:
                        polarityValue+=tokenPolarityValue
            else:
                tokenPolarityValue = self.getTokenPolarityValue(token, hashtokensOnly, lexicon)
                if tokenPolarityValue != None:
                    polarityValue+=tokenPolarityValue
        return polarityValue


    def getPoSFeatureIndex(self, PoSTag):
        '''
        Returns the index for a Part-of-Speech feature 
        using a dictionary of all possible Part-of-Speech tags
        '''
        PoSDictionary = {'ADJ': 1, 'BW': 2, 'LET': 3, 'LID': 4, 'N': 5, 'NP': 6, 'TSW': 7, 'TW': 8, 'VG': 9,
            'VNW': 10, 'VZ': 11, 'WW': 12, 'AFGEBR': 13, 'AFK': 14, 'ENOF': 15, 'META': 16, 'SYMB': 17, 'VREEMD': 18}
        return PoSDictionary[PoSTag]


    def createPoSBinaryFeatures(self):
        '''
        Creates binary Part-of-Speech features for an instance, 
        which indicate whether a certain PoS tag occurs in the instance or not
        Returns sparse features, together with the total number of features 
        (for feature indexing purposes)
        '''
        PoSBinaryFeatures = {}
        for tokenObject in self.tokenDictionary.values():
            PoSTagFeatureIndex = self.getPoSFeatureIndex(tokenObject.PoSTag)
            if PoSTagFeatureIndex not in PoSBinaryFeatures:
                PoSBinaryFeatures[PoSTagFeatureIndex] = True
        return PoSBinaryFeatures, 18
				

    def createPoSTernaryFeatures(self):
        '''
        Creates ternary Part-of-Speech features for an instance, 
        which indicate whether a certain PoS tag occurs 0, 1 or more times in the instance
        Returns sparse features, together with the total number of features 
        (for feature indexing purposes)
        '''
        PoSTernaryFeatures = {}
        for tokenObject in self.tokenDictionary.values():
            PoSTagFeatureIndex = self.getPoSFeatureIndex(tokenObject.PoSTag)
            if PoSTagFeatureIndex not in PoSTernaryFeatures:
                PoSTernaryFeatures[PoSTagFeatureIndex] = 1
            elif PoSTernaryFeatures[PoSTagFeatureIndex] == 1:
                PoSTernaryFeatures[PoSTagFeatureIndex] = 2
        return PoSTernaryFeatures, 18


    def createPoSAbsoluteFeatures(self):
        '''
        Creates absolute Part-of-Speech features for an instance, 
        which indicate how many times a certain PoS tag occurs in the instance
        Returns sparse features, together with the total number of features 
        (for feature indexing purposes)
        '''
        PoSAbsoluteFeatures = {}
        for tokenObject in self.tokenDictionary.values():
            PoSTagFeatureIndex = self.getPoSFeatureIndex(tokenObject.PoSTag)
            if PoSTagFeatureIndex not in PoSAbsoluteFeatures:
                PoSAbsoluteFeatures[PoSTagFeatureIndex] = 1
            else:
                PoSAbsoluteFeatures[PoSTagFeatureIndex]+=1
        return PoSAbsoluteFeatures, 18


    def createPoSFrequencyFeatures(self):
        '''
        Creates frequency Part-of-Speech features for an instance, 
        which indicate the frequency of a certain PoS tag in the instance
        Returns sparse features, together with the total number of features 
        (for feature indexing purposes)
        '''
        PoSFrequencyFeatures = {}
        for tokenObject in self.tokenDictionary.values():
            PoSTagFeatureIndex = self.getPoSFeatureIndex(tokenObject.PoSTag)
            if PoSTagFeatureIndex not in PoSFrequencyFeatures:
                PoSFrequencyFeatures[PoSTagFeatureIndex] = 1
            else:
                PoSFrequencyFeatures[PoSTagFeatureIndex]+=1
        for index in PoSFrequencyFeatures:
            PoSFrequencyFeatures[index] = PoSFrequencyFeatures[index]/len(self.tokenDictionary)
        return PoSFrequencyFeatures, 18


    def createNEBinaryFeature(self):
        '''
        Creates a binary named entity feature for the given instance, 
        which indicates whether or not the instance contains named entities
        '''
        for tokenObject in self.tokenDictionary.values():
            if tokenObject.NEtag != 'O':
                return True
        return False


    def createNEAbsoluteFeature(self):
        '''
        Creates an absolute named entity feature for the given instance, 
        which indicates how many named entities the instance contains
        '''
        counter = 0
        for tokenObject in self.tokenDictionary.values():
            if tokenObject.NEtag.startswith('B'):
                counter+=1
        return counter


    def createNETokenAbsoluteFeature(self):
        '''
        Creates an absolute named entity token feature for the given instance, 
        which indicates how many tokens are part of a named entity
        '''
        counter = 0
        for tokenObject in self.tokenDictionary.values():
            if tokenObject.NEtag != 'O':
                counter+=1
        return counter


    def createNETokenFrequencyFeature(self):
        '''
        Creates a named entity token frequency feature for the given instance, 
        which indicates the frequency of named entity tokens in that instance
        '''
        if len(self.tokenDictionary) > 0:
            return self.createNETokenAbsoluteFeature()/len(self.tokenDictionary)
        return 0


    def getDependencyRelations(self, featureTypes):
        '''
        Extracts all (unique) dependency relations from an instance
        '''
        dependencyRelations = {}
        for featureType in featureTypes:
            dependencyRelations[featureType] = []
        for relation in self.dependencyRelations:
            type = relation[0]
            headNr = relation[1]
            modifierNr = relation[2]
            if headNr == 0:
                head = 'ROOT'
            if modifierNr == 0:
                modifier = 'ROOT'
            if 'h-bo' in featureTypes:
                if headNr != 0:
                    head = self.tokenDictionary[headNr].PoSTag
                if modifierNr != 0:
                    modifier = self.tokenDictionary[modifierNr].content.lower()
                relationString = type + '(' + head + ', ' + modifier + ')'
                dependencyRelations['h-bo'].append(relationString)
            if 'm-bo' in featureTypes:
                if headNr != 0:
                    head = self.tokenDictionary[headNr].content.lower()
                if modifierNr != 0:
                    modifier = self.tokenDictionary[modifierNr].PoSTag
                relationString = type + '(' + head + ', ' + modifier + ')'
                dependencyRelations['m-bo'].append(relationString)
            if 'hm-bo' in featureTypes:
                if headNr != 0:
                    head = self.tokenDictionary[headNr].PoSTag
                if modifierNr != 0:
                    modifier = self.tokenDictionary[modifierNr].PoSTag
                relationString = type + '(' + head + ', ' + modifier + ')'
                dependencyRelations['hm-bo'].append(relationString)
            if 'hm-lex' in featureTypes:
                if headNr != 0:
                    head = self.tokenDictionary[headNr].content.lower()
                if modifierNr != 0:
                    modifier = self.tokenDictionary[modifierNr].content.lower()
                relationString = type + '(' + head + ', ' + modifier + ')'
                dependencyRelations['hm-lex'].append(relationString)
        for type in dependencyRelations:
            dependencyRelations[type] = list(set(dependencyRelations[type]))
        return dependencyRelations


    def createDependencyFeatures(self, featureTypes, workingDirectory):
        '''
        Creates dependency relation features for an instance
        Returns sparse features, together with the total number of features for each dependency feature type
        (for feature indexing purposes)
        '''
        dependencyFeatureIndexList = []
        numbersOfDependencyFeatures = DependencyRelationDictionary.getInstance(workingDirectory).getNumberOfDependencyRelationFeatures()
        dependencyRelationDictionary = self.getDependencyRelations(featureTypes)
        for relationType in dependencyRelationDictionary:
            for relation in dependencyRelationDictionary[relationType]:
                relationIndex = DependencyRelationDictionary.getInstance(workingDirectory).getRelationIndex(relation, relationType)
                if relationIndex:
                    dependencyFeatureIndexList.append(relationIndex)
        dependencyFeatureIndexList.sort()
        return (dependencyFeatureIndexList, numbersOfDependencyFeatures)		


    def getTokenNgrams(self, featureRange):
        '''
        Extracts all token ngrams from an instance
        [Based on code by Cynthia Van Hee]
        '''
        tokenNgrams = []
        lowercasedTokens = []
        for token in self.tokenDictionary.values():
            lowercasedTokens.append(token.content.lower())
        for i in featureRange:
            ngrams = []
            if len(lowercasedTokens) >= i:
                if i == 1:
                    for token in lowercasedTokens:
                        ngrams.append(token)
                else:
                    for j in range(len(lowercasedTokens)-i+1):
                        ngram = " ".join(token for token in lowercasedTokens[j:j+i])
                        ngrams.append(ngram)
            ngrams = list(set(ngrams))
            tokenNgrams.append(ngrams)
        return tokenNgrams

	
    def createTokenNgramFeatures(self, featureRange, workingDirectory):
        '''
        Creates token ngram features for an instance
        Returns sparse features, together with the total number of features for each n in the feature range
        (for feature indexing purposes)
        [Based on code by Cynthia Van Hee]
        '''
        tokenNgramFeatureIndexList = []
        totalNumberOfFeatures = TokenNgramDictionary.getInstance(workingDirectory).getNumberOfTokenNgramFeatures()
        for ngramList in self.getTokenNgrams(featureRange):
            for ngram in ngramList:
                ngramIndex = TokenNgramDictionary.getInstance(workingDirectory).getTokenNgramIndex(ngram)
                if ngramIndex:
                    tokenNgramFeatureIndexList.append(ngramIndex)
        tokenNgramFeatureIndexList.sort()
        return (tokenNgramFeatureIndexList, totalNumberOfFeatures)


    def getAllCharacterNgrams(self, featureRange):
        '''
        Extracts character ngrams from each token in an instance		
        '''
        allCharacterNgrams = [[] for _ in featureRange]
        for token in self.tokenDictionary.values():
            for i, n in enumerate(featureRange):
                for ngram in token.getCharacterNgrams(n):
                    allCharacterNgrams[i].append(ngram)
        allCharacterNgrams = [list(set(ngramList)) for ngramList in allCharacterNgrams]
        return allCharacterNgrams


    def createCharacterNgramFeatures(self, featureRange, workingDirectory):
        '''
        Creates character ngram features for an instance
        Returns sparse features, together with the total number of features for each n in the feature range
        (for feature indexing purposes)
        [Based on code by Cynthia Van Hee]
        '''
        charNgramFeatureIndexList = []
        totalNumberOfFeatures = CharacterNgramDictionary.getInstance(workingDirectory).getNumberOfCharacterNgramFeatures()
        for ngramList in self.getAllCharacterNgrams(featureRange):
            for ngram in ngramList:
                ngramIndex = CharacterNgramDictionary.getInstance(workingDirectory).getCharacterNgramIndex(ngram)
                if ngramIndex:
                    charNgramFeatureIndexList.append(ngramIndex)
        charNgramFeatureIndexList.sort()
        return (charNgramFeatureIndexList, totalNumberOfFeatures)


    def createPMINgramFeatures(self, dictionary, workingDirectory):
        '''
        Creates PMI ngram features for an instance
        Returns sparse features, together with the total number of features
        (for feature indexing purposes)
        '''
        if dictionary == 'nrc':
            ngramLists = self.getTokenNgrams(range(1, 3))			
            totalNumberOfFeatures = PMI_NRCDictionary.getInstance(workingDirectory).getNumberOfPMI_NRCFeatures()
            PMINgramFeatureDictionary = {}
            for ngramList in ngramLists:
                for ngram in ngramList:
                    PMINgramObject = PMI_NRCDictionary.getInstance(workingDirectory).getPMINgram(ngram)
                    if PMINgramObject:
                        PMINgramFeatureDictionary[PMINgramObject.index] = PMINgramObject.PMIscore
        elif dictionary == 'se':
            ngramLists = self.getTokenNgrams(range(1, 2))
            totalNumberOfFeatures = PMI_SemEvalDictionary.getInstance(workingDirectory).getNumberOfPMI_SemEvalFeatures()
            PMINgramFeatureDictionary = {}
            for ngramList in ngramLists:
                for ngram in ngramList:
                    PMINgramObject = PMI_SemEvalDictionary.getInstance(workingDirectory).getPMINgram(ngram)
                    if PMINgramObject:
                        PMINgramFeatureDictionary[PMINgramObject.index] = PMINgramObject.PMIscore
        return PMINgramFeatureDictionary, totalNumberOfFeatures
