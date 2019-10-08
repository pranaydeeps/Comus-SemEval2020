# Comus-SemEval2020

## Memotion Analysis

### About

1000 Tagged Images(Memes)

Task 1: Classify into Positive or Negative.

Task 2: Identify Type of Humour (Sarcastic, Offensive, Motivational, Other)

Task 3: Identify the Scale of the type of humour expressed

Update: 6499 TAGGED MEMES, BUT ONLY 1000 PROVIDED, HAVE TO DOWNLOAD REST

### Requirements

* Python-OpenCV - (incl. cv-contrib) for SIFT Features
* scrapy - For the imgflip Scraper
* PyTorch - For the Open Images Label Extractor
* NLTK - For the punkt tokenizer 

### Tools

* imgflip_scraper - Scrapes ImgFlip.com for Meme Templates. Downloaded templates are stored in data/TEMPLATES. To scrape, go to the folder and:

```python

scrapy crawl templates_spider

```

* Pittacium - Tool for Extracting Object Labels from a given image. Requires pytorch model file, which can't be uploaded to the repository. E-Mail me if needed, and I can send it over. Model Converted from Tensorflow to PyTorch. To use, go to the folder and:

```python

from infer_torch import predict
predict(image_path)

```
### Scripts

* utils.py - General Functions
  * prep_task1(): Create Data for Task 1. Format [img_name, text, sentiment]
  * load_nrc_emotion_lex(): Loads the NRC Emotion Lexicon and returns [positive_words, negative_words, neutral_words]
  * lex_classifier(): Uses the NRC Emotion Lexicon to perform Sentiment Analysis on a set of texts and outputs classification scores for each class.

* find_similarities_between_images.py - Uses SIFT-Based Features to look for a template for a particular meme, out of the 800 templates scraped from ImgFlip

### To-Do

* Train a Supervised Model Baseline using various features
* Create a dictionary of the words in Memes and attempt to create meaningful 'memebeddings'

### Integration Sentiment Features English

python utils_els.py ./WorkingDir/ train ./WorkingDir/test.csv data_test_out.txt feature_index.txt

