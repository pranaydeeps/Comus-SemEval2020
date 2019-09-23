# Comus-SemEval2020

## Memotion Analysis

### About

1000 Tagged Images(Memes)

Task 1: Classify into Positive or Negative.

Task 2: Identify Type of Humour (Sarcastic, Offensive, Motivational, Other)

Task 3: Identify the Scale of the type of humour expressed

Update: 6499 TAGGED MEMES, BUT ONLY 1000 PROVIDED, HAVE TO DOWNLOAD REST


### Scripts

* utils.py - General Functions
⋅⋅* prep_task1(): Create Data for Task 1. Format [img_name, text, sentiment]
⋅⋅* load_nrc_emotion_lex(): Loads the NRC Emotion Lexicon and returns [positive_words, negative_words, neutral_words]
⋅⋅* lex_classifier(): Uses the NRC Emotion Lexicon to perform Sentiment Analysis on a set of texts and outputs classification scores for each class.

### To-Do

* Train a Supervised Model Baseline using various features
* Create a dictionary of the words in Memes and attempt to create meaningful 'memebeddings'