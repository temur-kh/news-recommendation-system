from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
from nltk import word_tokenize
russian_stopwords = stopwords.words("russian")


class Singleton(type):
    _instances = {}

    def __call__(self, *args, **kwargs):
        if self not in self._instances:
            self._instances[self] = super(Singleton, self).__call__(*args, **kwargs)
        return self._instances[self]


class MyStemObject(Mystem, metaclass=Singleton):
    pass


class Vectorizer(TfidfVectorizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = self

    def __call__(self, sample):
        return self.transform(sample)


# проводит токенизацию и лемматизацию текста
def preprocess_text(text):
    tokens = word_tokenize(text)
    text = " ".join(tokens)
    mystem = MyStemObject()
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords
              and token != " "
              and token.strip() not in punctuation]

    return tokens


# параметры для tfidf векторизатора
def get_tfidf_params():
    params = {
        'tokenizer': preprocess_text,
        'stop_words': stopwords.words("russian"),
        'ngram_range': (1, 3),
        'min_df': 3
    }
    return params
