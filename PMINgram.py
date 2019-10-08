#!usr/bin/env python
# encoding: utf-8

"""
PMINgram.py

Created by Marjan Van de Kauter on 2015-01-12.
Copyright (c) 2015 LT3. All rights reserved.

Created in the framework of the SentiFM project (Sentiment mining for Financial Markets)
Based on the code created for SemEval-2014 Task 9: Sentiment Analysis in Twitter
PMINgram objects are used to store ngrams with their corresponding index and PMI score
"""


class PMINgram:


    def __init__(self, content, index, PMIscore):
        '''PMINgram constructor'''
        self.content = content
        self.index = index
        self.PMIscore = PMIscore
