#!usr/bin/env python
# encoding: utf-8

"""
SentimentWord.py

Created by Marjan Van de Kauter on 2015-01-12.
Copyright (c) 2015 LT3. All rights reserved.

Created in the framework of the SentiFM project (Sentiment mining for Financial Markets)
Based on the code created for SemEval-2014 Task 9: Sentiment Analysis in Twitter
SentimentWord objects are used to store sentiment words (incl. emoticons) from sentiment lexicons 
with their corresponding sentiment polarity value and PoSTag (if available)
"""


class SentimentWord:


    def __init__(self, wordForm, PoSTag, sentimentPolarityValue):
        '''SentimentWord constructor'''
        self.wordForm = wordForm
        self.PoSTag = PoSTag
        self.sentimentPolarityValue = sentimentPolarityValue
