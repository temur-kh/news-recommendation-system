from system import *
import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
emb_path = "ft_native_300_ru_wiki_lenta_lemmatize.bin"
path_to_dataset = 'data/'


def get_filenames(path):
    filenames = [path+pos_json for pos_json in os.listdir(path) if pos_json.endswith('.json')]
    return filenames


def read_dataset(filenames):
    files = []
    for file_name in filenames:
        with open(file_name) as file:
            file_dict = json.load(file)
            files.append(file_dict)
    return files


def get_features(files):
    data = [{
        'title': file['title'],
        } for file in files]
    return data


def dataset_to_df(files):
    df = pd.DataFrame(files)
    return df


# функция, объединяющая все вышеперечисленное
def get_files_df(path, size=None):
    filenames = get_filenames(path)
    if size is None:
        size = len(filenames)
    files = read_dataset(filenames[:size])
    dicts = get_features(files)
    df = dataset_to_df(dicts)
    return df


def test():
    print('Running the testing....')
    print('\n', '=' * 100, '\n')
    # считаем датасет и разделим его на две части
    data_df = get_files_df(path_to_dataset, size=10000)
    train_df, test_df = train_test_split(data_df, test_size=0.01, random_state=19)

    user = User(id=123)
    user_history = test_df.sample(n=10, random_state=19)
    user.init_history(user_history)

    system = NewsAdviser(emb_path=emb_path)
    system.build(train_df, verbose=True)

    print('\n', '=' * 100, '\n')

    news_list = system.recommend_news(user, verbose=True)

    news = user.select_news(news_list)
    user.watch_news(news, verbose=True)

    print('\n', '='*100, '\n')

    news = user.select_news(news_list)
    user.watch_news(news, verbose=True)

    _ = system.recommend_news(user, verbose=True)

    print('\n', '=' * 100, '\n')

    print('Test is finished successfully!')


if __name__ == '__main__':
    test()
