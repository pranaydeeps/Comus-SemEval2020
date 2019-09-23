from nltk.tokenize import word_tokenize
import codecs
import string
import re
import math
import csv


def prep_task1(data_list):
    task1_list = []
    for item in data_list:
        if len(item)==9:
            temp_list = []
            temp_list.append(item[0])
            temp_list.append(item[2])
            if 'positive' in item[8]:
                temp_list.append(1)
            if 'neutral' in item[8]:
                temp_list.append(0)
            if 'negative' in item[8]:
                temp_list.append(2)
            task1_list.append(temp_list)
    return task1_list

def lex_classifier(pos_words, neg_words, neutral_words, datalist):
    correct = 0
    incorrect = 0
    recognized = 0
    total_words = 0
    
    pos_correct = 0
    neg_correct = 0
    pos_incorrect = 0
    neg_incorrect = 0
    neutral_correct = 0
    neutral_incorrect = 0

    print('Iterating through dataset')
    for item in datalist:
        line = item[1].lower()
        linemath = 0
        actualmath = 0
        data = line.strip('\n')
        if data:
            all_words = word_tokenize(data)
            for word in all_words:
                if word in pos_words:
                    linemath+=1
                    recognized+=1
                if word in neg_words:
                    linemath-=1
                    recognized+=1
                if word in neutral_words:
                    recognized+=1

            total_words += len(all_words)
            if linemath!=0:
                linemath = linemath/abs(linemath)
            if item[2] == 1:
                if linemath == 1:
                    correct+=1
                    pos_correct+=1
                else:
                    pos_incorrect+=1       
                    incorrect+=1
            if item[2] == 2:
                if linemath == -1:
                    correct+=1
                    neg_correct+=1
                else:
                    neg_incorrect+=1
                    incorrect+=1
            if item[2] == 0:
                if linemath == 0:
                    correct+=1
                    neutral_correct+=1
                else:
                    neutral_incorrect+=1
                    incorrect+=1

    total = correct + incorrect
    acc = correct/float(total)
    print('### LEXICON CLASSIFICATION PROCESSING SUMMARY ###')
    print('\n ')
    print('Classification Accuracy: {}/{} = {}'.format(correct, total, acc))
    print('Words Recognized from Lexicon: {}/{} = {}'.format(recognized, total_words, recognized/float(total_words)))
    print('##############################################')
    print('\n')
    print("Classwise Accuracies:")
    print("Positive: {}/{} = {}".format(pos_correct, (pos_correct+pos_incorrect), pos_correct/float(pos_correct+pos_incorrect)))
    print("Negative: {}/{} = {}".format(neg_correct, (neg_correct+neg_incorrect), neg_correct/float(neg_correct+neg_incorrect)))
    print("Neutral: {}/{} = {}".format(neutral_correct, (neutral_correct+neutral_incorrect), neutral_correct/float(neutral_correct+neutral_incorrect)))

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
            # if processed[1] == 'positive' or processed[1] == 'negative':
            #     if processed[2] == '0' and pos_check == 1:
            #         neutral_words.append(processed[0])
            #         pos_check = 0
            #     if processed[2] == '0':
            #         pos_check = 1

    pos_words = list(set(pos_words))
    neg_words = list(set(neg_words))
    neutral_words = list(set(neutral_words))            
    
    return pos_words, neg_words, neutral_words

with open('data/data_7000.csv', 'r') as f:
    reader = csv.reader(f)
    data_list = list(reader)

print('Read Data From CSV...')
pos_words, neg_words, neutral_words = load_nrc_emotion_lex('data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')
print('NRC Emotion Lexicon Loaded...')
processed_dataset = prep_task1(data_list)
print('Prepared Dataset...')
lex_classifier(pos_words, neg_words, neutral_words, processed_dataset)