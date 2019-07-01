from deeppavlov.models.embedders.fasttext_embedder import FasttextEmbedder
from deeppavlov.models.embedders.tfidf_weighted_embedder import TfidfWeightedEmbedder
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
import pandas as pd
from .helpers import Vectorizer, get_tfidf_params
from .user import User
from random import randint


class NewsAdviser:
    """
    Система рекомендации новостей
    """
    def __init__(self, emb_path, feature='title'):
        self.emb_path = emb_path
        self.feature = feature
        self.data_df = None
        self.tfidf = Vectorizer(**get_tfidf_params())
        self.fasttext_embedder = None
        self.fasttext_tfidf = None
        self.dimension = 300
        rbp = RandomBinaryProjections('rbp', 2)
        self.engine = Engine(self.dimension, lshashes=[rbp])
        pass

    # сохраняет вектора в engine
    def _store_vectors(self, df=None):
        if df is None:
            df = self.data_df
        for index, row in df.iterrows():
            tfidf_vec = self.fasttext_tfidf([[row[self.feature]]])[0]
            vector = tfidf_vec.reshape((self.dimension,))
            self.engine.store_vector(vector, index)

    # создает модель
    def build(self, data_df: pd.DataFrame, verbose=False):
        if verbose:
            print('Start Building Model...')
        self.data_df = data_df
        self.tfidf.fit([i for i in data_df[self.feature]])
        if verbose:
            print('Have fitted TF-IDF Vectorizer.')
        self.fasttext_embedder = FasttextEmbedder(self.emb_path)
        if verbose:
            print('Loaded FastText Model.')
        self.fasttext_tfidf = TfidfWeightedEmbedder(embedder=self.fasttext_embedder, vectorizer=self.tfidf)
        self._store_vectors()
        if verbose:
            print('Have stored all vectors.')
            print('Building Finished.')
        # TODO: pickle model and load it next time

    # позволяет получить соседей по индексу из датасета
    def _get_samples(self, neighbors):
        indices = []
        for neighbor in neighbors:
            vec, index, res = neighbor
            indices.append(index)
        samples = self.data_df.loc[indices, :]
        return samples

    # запрос по одному предложению
    def _query(self, request):
        tfidf_vec = self.fasttext_tfidf([[request]])[0]
        vector = tfidf_vec.reshape((self.dimension,))
        neighbors = self.engine.neighbours(vector)
        samples = self._get_samples(neighbors)
        return samples

    # рекомендует определенное количество статей с рандомным выбором статей из истории юзера для запросов
    def recommend_news(self, user: User, amount=10, verbose=False):
        recommendations = []
        history = user.history
        for i in range(amount):
            # выберим рандомную запись из истории пользователя
            rand_n = randint(0, len(history) - 1)
            request = history[rand_n][self.feature]
            if verbose:
                print('Finding recommendation for the article: {}'.format(request))
            samples = self._query(request)

            # выберим одну рандомную статью, чтобы добавить в рекомендательную выборку
            news = (samples.sample(n=1)).iloc[0]
            if verbose:
                print('Recommend to user {} an article: {}'.format(user.id, news[self.feature]))
            recommendations.append(news)
        return recommendations
    #
    # def add_news_to_dataset(self, news_df: pd.DataFrame):
    #     self.data_df.append(news_df, ignore_index=True)
    #     self.tfidf.fit(news_df)                         # нужно пофиксить фитинг новых данных в tfidf
    #     self.fasttext_tfidf = TfidfWeightedEmbedder(embedder=self.fasttext_embedder, vectorizer=self.tfidf)
    #     self._store_vectors(news_df)


