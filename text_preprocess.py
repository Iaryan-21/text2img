import re
import torch
from torch.utils.data import Dataset, DataLoader
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    tokens = text.split()
    words = [w for w in tokens if w not in stopwords.words("english")]
    stemmed = [PorterStemmer().stem(w) for w in words]
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    return lemmed


vocab = {'<PAD>': 0, '<UNK>': 1}
def build_vocab(texts):
    for text in texts:
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = len(vocab)


def text_to_sequence(text):
    tokens = tokenize(text)
    return [vocab.get(token, vocab['<UNK>']) for token in tokens]

# texts = [
#     "A beautiful, sunny day! Isn't it wonderful?",
#     "The rain in Spain stays mainly in the plain."
# ]

# build_vocab(texts)

# text_sequence = text_to_sequence(texts[0])
# print(f"Tokens: {tokenize(texts[0])}")
# print(f"Text sequence: {text_sequence}")

# # Print the vocabulary
# print(f"Vocabulary: {vocab}")
