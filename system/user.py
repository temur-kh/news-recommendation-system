from random import randint


class User:
    """
    Класс пользователя
    """
    def __init__(self, id, history_capacity=10):
        self.id = id
        self.history = []
        self.history_capacity = history_capacity

    # функция для инициализации начальной истории
    def init_history(self, data_df):
        if data_df.shape[0] > 10:
            raise ValueError('The data_df size is bigger than history_capacity')
        for index, row in data_df.iterrows():
            self.history.append(row)

    # функция для просмотра статьи и сохранения в истории юзера
    def watch_news(self, news, verbose=False):
        if verbose:
            print('The user {} watched the news: {}'.format(self.id, news['title']))
        self.history.append(news)
        if len(self.history) > self.history_capacity:
            del self.history[:1]

    # функция, позволяющая выбрать рандомную статью из выборки для дальнейшего просмотра
    @staticmethod
    def select_news(news_list):
        # select one random news to watch (this is just imitation of user selecting news)
        rand_n = randint(0, len(news_list) - 1)
        return news_list[rand_n]
