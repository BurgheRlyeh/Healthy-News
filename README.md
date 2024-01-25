# Healthy News

Проект по курсу "Основы машинного обучения", СПбПУ, 2024

## Описание

С помощью двух NLP моделей, а также сторонних библиотек Python, была реализована программа-анализатор текста, способная распознавать в нем именнованные сущности (персоны, компании, события, географические и геополитические сущности), а также определять окрас текста (негативный/позитивный).

### Входные данные

При запуске программы она спрашивает, в каком формате формате ввести текст. Возможны следующие варианты:
1. Из консоли
2. Из файла
3. Из фото
4. Из видеофайла
5. Из аудиофайла

### Основная работа

После получения текста, он переводится на английский язык и передается в модель, определяющую тональность текста. В случае, если модель приходит к выводу, что текст имеет позитивный окрас, программа его печатает, в ином случае пользователя предупреждают о возможном вреде и выводят именные сущности, встречаемые в тексте, в порядке частоты.

## Модели

### Окрас текста

Простая модель, реализованная с помощью NLP библиотеки spaCy. Бинарно классифицирует предложения. Использует [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) с отзывами на фильмы.

Accuracy (Ali) ~ 0.65
Accuracy (IMDB) ~ 0.71

### Именные сущности

Модель, обучаемая вручную, использующая алгоритмы LSTM (Long short-term memory) и оптимизатор Адам для градиентного спуска, а также используются библиотеки PyTorch и fastText. В качестве датасетов для модели выступают ['cc.en.300.bin'](https://fasttext.cc/docs/en/crawl-vectors.html) - отображение английских слов в вектора, и [ner_dataset](https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus/data) с именованными сущностями, также на английском языке.

Accuracy ~ 0.85

## Требования

Требования к пакетам смотреть в [requirements.txt](requirements.txt).

