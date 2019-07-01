# Система рекомендации новостей

Коротко по файлам, содержащимся в директории:

1. Report.pdf - краткий репорт по исследованию
2. Data-Analysis.ipynb - ноутбук для анализа датасета
3. diagrams/ - папка, содержащая диаграммы созданные в ноутбуке Data-Analysis.ipynb. Я их сохранил, так как изображения с Plotly не сохранились в выводах ячеек
4. News-Recommendation-System.ipynb - ноутбук для создания модели системы рекомендационной системы
5. test_system.py - скрипт тестирующий реализованную систему
6. system/ - папка, содержащая Python файлы рабочей системы

Я советую просматривать всё в вышеуказанном порядке.

Также, если вы хотите запустить ноутбуки и скрипты, вам придется загрузить датасет и закинуть все json файлы в папку data/ текущей директории.
Ссылка дана в репорте.

Плюс к этому вам необходимо будет загрузить в текущую директорию FastText модель.

Ссылка на модель: http://files.deeppavlov.ai/embeddings/ft_native_300_ru_wiki_lenta_lemmatize/ft_native_300_ru_wiki_lenta_lemmatize.bin


В остальном вам достаточно иметь установленную Anaconda и DeepPavlov (pip install deeppavlov).