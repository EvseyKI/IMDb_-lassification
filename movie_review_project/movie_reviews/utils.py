import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from string import punctuation
from gensim.parsing.preprocessing import STOPWORDS
import pandas as pd

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

lemmatizer = WordNetLemmatizer()
stop_words = STOPWORDS | set(stopwords.words('english'))
negation_words = {"not", "no", "never", "neither", "nor", "but", "however", "although"}
stop_words = stop_words - negation_words

contractions_df = pd.read_csv('movie_reviews/models/contractions.csv')
contractions = dict(zip(contractions_df['Contraction'], contractions_df['Meaning']))

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def expand_contractions(text):
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in contractions.keys()) + r')\b')
    return pattern.sub(lambda x: contractions[x.group()], text)

def process_text(text):
    text = expand_contractions(text)
    if "<" in text and ">" in text:
        text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'\d+', '', text)
    text_without_punc = re.sub(f'[{re.escape(punctuation)}]', '', text)
    tokens = word_tokenize(text_without_punc.lower())
    pos_tags = pos_tag(tokens)
    clean_tokens = [
        lemmatizer.lemmatize(word, pos=get_wordnet_pos(pos) or wordnet.NOUN)
        for word, pos in pos_tags
        if word not in stop_words
    ]
    
    return clean_tokens
