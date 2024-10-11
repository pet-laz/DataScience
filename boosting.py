# [0]     test-logloss:0.31470    train-logloss:0.29773
# [1]     test-logloss:0.27868    train-logloss:0.25908
# [2]     test-logloss:0.27898    train-logloss:0.24361
# [3]     test-logloss:0.27163    train-logloss:0.23343
# [4]     test-logloss:0.25738    train-logloss:0.22304
# [5]     test-logloss:0.24147    train-logloss:0.21063
# [6]     test-logloss:0.23687    train-logloss:0.20400
# [7]     test-logloss:0.23644    train-logloss:0.19658
# [8]     test-logloss:0.22932    train-logloss:0.18955
# [9]     test-logloss:0.22597    train-logloss:0.18258
# [10]    test-logloss:0.21775    train-logloss:0.17771
# [11]    test-logloss:0.21105    train-logloss:0.17260
# [12]    test-logloss:0.20293    train-logloss:0.16686
# [13]    test-logloss:0.20741    train-logloss:0.16294
# [14]    test-logloss:0.20079    train-logloss:0.15942
# [15]    test-logloss:0.21082    train-logloss:0.15664
# [16]    test-logloss:0.20795    train-logloss:0.15373
# [17]    test-logloss:0.20512    train-logloss:0.15123
# [18]    test-logloss:0.19664    train-logloss:0.14718
# [19]    test-logloss:0.19899    train-logloss:0.14551
# [20]    test-logloss:0.20340    train-logloss:0.14319
# [21]    test-logloss:0.19903    train-logloss:0.14070
# [22]    test-logloss:0.19720    train-logloss:0.13852
# [23]    test-logloss:0.20169    train-logloss:0.13621
# [24]    test-logloss:0.19871    train-logloss:0.13348
# [25]    test-logloss:0.19684    train-logloss:0.13193
# [26]    test-logloss:0.19518    train-logloss:0.13050
# [27]    test-logloss:0.19464    train-logloss:0.12908
# [28]    test-logloss:0.19788    train-logloss:0.12811
# [29]    test-logloss:0.19478    train-logloss:0.12627

# Этот вывод представляет собой результат обучения модели с использованием алгоритма градиентного бустинга XGBoost. 
# Каждая строка соответствует отдельной итерации (дереву) обучения модели,
# а значения "test-logloss" и "train-logloss" представляют значения функции потерь (logloss)
# на тестовой и обучающей выборках соответственно. 
 
# - "test-logloss" - значение функции потерь на тестовой выборке. 
# Это показатель того, насколько хорошо модель обобщает данные, которые она не видела в процессе обучения. 
# - "train-logloss" - значение функции потерь на обучающей выборке. 
# Это показатель того, насколько хорошо модель подстраивается под данные обучающей выборки. 
 
# Цель обучения модели - минимизировать значение функции потерь как на тестовой, так и на обучающей выборке. 
# Уменьшение значения функции потерь на обучающей выборке при каждой итерации может свидетельствовать о том, 
# что модель продолжает улучшаться и настраиваться под данные. 
# Однако, важно следить за значением функции потерь на тестовой выборке, 
# чтобы избежать переобучения модели.


def d1():
    import pandas as pd
    import numpy as np

    titanic = pd.read_csv("9.7_titanic.csv")
    # print(titanic.head())

    targets = titanic.Survived
    data = titanic.drop(columns="Survived")

    trees = [1] + list(range(10, 100, 10))
    # print(trees)

    from sklearn.ensemble import AdaBoostClassifier
    #AdaBoostClassifier - это метод адаптивного бустинга, который используется для построения ансамблей моделей машинного обучения.
    # Он работает путем комбинирования нескольких слабых учеников (например, деревьев решений) в сильный ученик. 
    # AdaBoostClassifier обучает каждую модель последовательно,
    # присваивая больший вес неправильно классифицированным образцам на каждой итерации,
    # что позволяет модели фокусироваться на тех образцах, которые сложно классифицировать.
    # В конечном итоге,
    # AdaBoostClassifier комбинирует предсказания всех моделей для получения итогового прогноза.
    from sklearn.model_selection import cross_val_score
    # Функция cross_val_score из библиотеки scikit-learn используется для выполнения кросс-валидации модели. 
    # Кросс-валидация - это метод оценки производительности модели, который позволяет оценить, 
    # насколько хорошо модель обобщает на новых данных. 
    # Функция cross_val_score разбивает данные на несколько фолдов, 
    # обучает модель на каждом фолде и оценивает ее производительность. 
    # Возвращаемое значение - это массив оценок производительности модели на каждом фолде. 
    # Это полезный инструмент для выбора наилучших гиперпараметров модели и оценки ее способности к обобщению.
    ada_scoring = []
    for tree in trees:
        ada = AdaBoostClassifier(n_estimators=tree, algorithm='SAMME')
        score = cross_val_score(ada, data, targets, scoring="roc_auc", cv=3)
        ada_scoring.append(score)
    ada_scoring = np.asmatrix(ada_scoring)
    # print(ada_scoring)


    #То же проделаем для GradientBoostingClassifier & XGBClassifier (вместо Ada)
    from sklearn.ensemble import GradientBoostingClassifier
    gbc_scoring = []
    for tree in trees:
        gbc = GradientBoostingClassifier(n_estimators=tree)
        score = cross_val_score(gbc, data, targets, scoring="roc_auc", cv=3)
        gbc_scoring.append(score)
    gbc_scoring = np.asmatrix(gbc_scoring)

    from xgboost import XGBClassifier
    xgb_scoring = []
    for tree in trees:
        xgb = XGBClassifier(n_estimators=tree)
        score = cross_val_score(xgb, data, targets, scoring="roc_auc", cv=3)
        xgb_scoring.append(score)
    xgb_scoring = np.asmatrix(xgb_scoring)

    import matplotlib.pyplot as plt

    plt.plot(trees, ada_scoring.mean(axis=1), label='AdaBoost')
    plt.plot(trees, gbc_scoring.mean(axis=1), label='GradientBoost')
    plt.plot(trees, xgb_scoring.mean(axis=1), label='XGBoost')
    plt.grid(True)
    plt.xlabel('trees')
    plt.ylabel('auc_score')
    plt.legend(loc='lower right')

    plt.show()


def dz1_p1():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import pandas as pd

    x, y = make_classification(n_samples=1000, n_features=15, n_informative=7, 
                            n_redundant=3, n_repeated=3, random_state=17)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=17)

    features = pd.DataFrame(x)
    targets = pd.Series(y)

    scat_mtrx = pd.plotting.scatter_matrix(features, c=targets, figsize=(25, 25), marker='o',
                                        hist_kwds={'bins': 20}, s=40, alpha=0.8)

    import xgboost as xgb

    dtrain = xgb.DMatrix(x_train, y_train)
    dtest = xgb.DMatrix(x_test, y_test)

    params = {'objective': 'binary:logistic',
            'max_depth': 3,
            'eta': 0.1}

    num_rounds = 60

    # xgb_model = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_rounds)

    evals = [(dtest, 'test'), (dtrain, 'train')]
    xgb_model = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_rounds, evals=evals)
    score = xgb.plot_importance(xgb_model)
    print(score)



def dz1_p2():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import xgboost as xgb

    x, y = make_classification(n_samples=1000, n_features=7, n_informative=3, n_redundant=3, 
                           n_classes=2, weights=[0.9, 0.1], random_state=20)

    # print(f'There are {sum(y)} positive instances')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=17)
    dtrain = xgb.DMatrix(x_train, y_train)
    dtest = xgb.DMatrix(x_test, y_test)

    features = pd.DataFrame(x)
    targets = pd.Series(y)

    scat_mtrx = pd.plotting.scatter_matrix(features, c=targets, figsize=(25, 25), marker='o',
                                        hist_kwds={'bins': 20}, s=40, alpha=0.8)

    params = {'objective': 'binary:logistic', 
          'max_depth': 1, 
          'verbosity': 0, 
          'eta': 1}

    num_rounds = 30

    evals = [(dtest, 'test'), (dtrain, 'train')]
    xgb_model = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_rounds, evals=evals)

    
    # С помощью метода train обучите модель.
    # С помощью метода predict получите предсказания для тестовых данных. 
    # Так как алгоритм возвращает вероятности, получите бинарную матрицу значений этих вероятностей, 
    # элементы которой при полученной вероятности > 0.5 равны True, а при вероятности <= 0.5 равны False. 
    # Выведите эту матрицу.

    xgb_prediction = xgb_model.predict(dtrain)
    # print(xgb_prediction)
    # # #### ДОБАВИЛ ОКРУГЯШКУ ВМЕСТО print(xgb_prediction) шобы было понятно че к чему (Zакоменчено)
    # import numpy as np
    # formatted_numbers = np.array([np.format_float_positional(num, precision=4, unique=False, fractional=False) for num in xgb_prediction])
    # print(formatted_numbers)

    # count_Trues = 0
    # for num in xgb_prediction:
    #     if num > 0.5: 
    #         count_Trues += 1
    # print(count_Trues) # count_Trues = 52 Да здравствует Санкт-Петербург, и это город наш. Я каждый свой новый куплет валю как никогда


    # bin_xgb_prediction = []
    # for num in xgb_prediction:
    #     if num > 0.5: 
    #         bin_xgb_prediction.append(True)
    #     else:
    #         bin_xgb_prediction.append(False)
    bin_xgb_prediction = [True if num > 0.5 else False for num in xgb_prediction]
    # print(bin_xgb_prediction)

    # Выведите матрицу ошибок, точность и полноту для полученных предсказаний.
    from sklearn.metrics import confusion_matrix, precision_score, recall_score
    xgb_confusion_matrix = confusion_matrix(y_train, bin_xgb_prediction)
    print(xgb_confusion_matrix)

    # Теперь зададим вручную веса для экземпляров классов.
    import numpy as np

    weights = np.zeros(len(y_train))
    weights[y_train == 0] = 1
    weights[y_train == 1] = 5

def main():
    dz1_p2()


if __name__=='__main__':
    main()