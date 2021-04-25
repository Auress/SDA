### Итоговый проект курса "Спортивный анализ данных. Платформа Kaggle"
### -- Автор: Шенк Евгений Станиславович

Стек:
ML: pandas, numpy, lightgbm, xgboost, catboost, sklearn, scipy

Соривнование GeekBrains Competitive Data Analysis: https://www.kaggle.com/c/geekbrains-competitive-data-analysis/data  

Задача: Предсказать шанс (от 0 до 1) дефолта клиентов (из датасета test.csv) по информации из представленых датасетов.
Метрика "ROC-AUC". 

Файлы датасета:  
train.csv - пары "заявка - целевая переменная", для этой выборки нужно собрать признаки и обучить модель;
test.csv - пары "заявки - прогнозное значение", для этой выборки нужно собрать признаки и построить прогнозы;
bki.csv - данные БКИ о предыдущих кредитах клиента;
client_profile.csv - клиентский профиль, некоторые знания, которые есть у компании о клиенте;
payments.csv - история платежей клиента;
applications_history.csv - история предыдущих заявок клиента.

Выполнение:
1. Проведен exploratory data analysis для данных датасетов (распределение признаков, пропуски, выбросы и т.д.).
    Представлены в файлах:
    - EDA_applications_history_data.IPYNB
    - EDA_bki_data.IPYNB
    - EDA_client_profile_data.IPYNB
    - EDA_payments_data.IPYNB
2. Выбрана схема валидации: StratifiedKFold(n_splits=7, shuffle=True)
3. Сгенерированы признаки на основе признаков из клиентского профиля и статистик из других источников сгруппированных по полю 'APPLICATION_NUMBER'
4. Обучена модель LightGBM и с помощью нескольких раундов выбора признаков по Permutation importance выбран набор самых существенных и сохранен (top_feat.json).
5. Обучены модели LightGBM, XGBoost, CatBoost на выбраных признаках:
    - подобраны гиперпараметры для моделей с помощью GridSearchC и ручного перебора
    - обучены модели по критерию early_stopping
    - обучены модели со средним количеством деревьев из прошлого пункта (для стэкинга)
    - обучена модель поверх предсказаний предыдущих (xgb.XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=2))
6. Для датасета для предсказаний в результате сделаны два предсказания (можно выбрать на kaggle):
    - усреднение предсказаний модели для стэкинга
    - усреднение предсказаний модели только XGBoost

Проект содержит:  
1. CW_model.IPYNB - Основной нотбук, с загрузкой данных, генерацией признаков, обучением моделей и предсказаниями.
2. Файлы exploratory data analysis:
    - EDA_applications_history_data.IPYNB
    - EDA_bki_data.IPYNB
    - EDA_client_profile_data.IPYNB
    - EDA_payments_data.IPYNB
3. src/utils.py с функциями:
    - show_feature_importances - Отображение важности признаков.
    - do_train_test_split - Разделение на train, valid и test датасеты.
    - make_cross_validation - Кросс-валидация.
    - create_numerical_aggs - Построение агрегаций для числовых признаков.
4. папка models - содержит файлы .json: список важных признаков и гипер-параметры моделей
5. README.md - файл с информацией
