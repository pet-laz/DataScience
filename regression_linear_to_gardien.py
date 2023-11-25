# ЛИНЕЙНАЯ РЕГРЕССИЯ

def d1(): # график площади к цене
    from matplotlib import pyplot as plt
    plt.scatter([50,60,70,100], [10,30,40,50], 40, 'g', 'o', alpha=0.8)
    plt.show()


def d2(): # Реализация линейной регрессии (палки по точкам, которая типа по серединке)
    # выпишем матрицу
    import numpy as np

    X = np.array( [[1, 50], [1, 60], [1, 70], [1, 100]] ) #еденички - тривиальные переменные для каждого объека, 
    #второй столбик значение площади на которых надо обучиться
    #Вектор Y (цена)
    Y = np.array( [[10], [30], [40], [50]] )
    # начинаем хуячить формулу X**T * X
    X_T_X = (X.T).dot(X) # перемножение матрицы (T - транспонирование, dot - перемножение)
    # найдем обратную матрицу
    from numpy.linalg import inv
    X_T_X_inv = inv(X_T_X)
    # реализуем формулу ( ( (X ** T) * X ) ** -1) * ( X ** T ) * Y
    w = (X_T_X_inv.dot(X.T).dot(Y))


def d3(): 
    # тож самое но кратко
    import numpy as np
    from numpy.linalg import inv
    from matplotlib import pyplot as plt
    X = np.array( [[1, 50], [1, 60], [1, 70], [1, 100]] ) 
    Y = np.array( [[10], [30], [40], [50]] )
    w = inv((X.T).dot(X)).dot(X.T).dot(Y) # легендарная формула Линейной регресси на Python 

    # ебашим предсказание (как ванга)
    
    #задаем границы
    margin = 10 # граница
    X_min = X[:,1].min() - margin
    X_max = X[:,1].max() + margin

    # наблор точек для прямой 
    X_support = np.linspace(X_min, X_max, num=100) # num - кол-во точек
    # Гадаем
    Y_model = w[0][0] + w[1][0]*X_support # это реализация функции y = ax + b
    plt.xlim(X_min, X_max)
    plt.ylim(0, Y[:,0].max() + margin)
    # исходные точки
    plt.scatter(X[:,1], Y[:,0], 40, 'g', 'o')
    # гадание 
    plt.plot(X_support, Y_model)
    plt.show()
    


def ndprint(a, format_string ='{0:.2f}'):
        """Функция, которая распечатывает список в красивом виде"""
        return [format_string.format(v,i) for i,v in enumerate(a)]


def d4():
    # теперь мы делаем тоже самое, но используя не написанные наши формулы, а встроенные 
    from sklearn.datasets import fetch_openml
    import numpy as np
    boston_dataset = fetch_openml(name='boston', version=1, as_frame=False, parser='auto')
    
    features =  np.array(boston_dataset.data, dtype=float)
    y = np.array(boston_dataset.target)
    # print('Матрица Объекты X Фичи  (размерность): %s %s' % features.shape)
    # print('Целевая переменная y (размерность): %s' % y.shape)

    from numpy.linalg import inv
    w_analytic = np.linalg.inv(features.T.dot(features)).dot(features.T).dot(y)
    # print("Аналитически определённые коэффициенты \n%s" % ndprint(w_analytic))

    from sklearn.linear_model import LinearRegression

    # обучаем модель "из коробки"
    reg = LinearRegression().fit(features, y)
    print("Коэффициенты, вычисленные моделью sklearn \n%s" % ndprint(reg.coef_))


def d5():
    # ДЗШКА
    # Есть два набора точек - X_hw и Y_hw. В рамках домашней работы нужно:
    #  -визуализировать набор точек
    #  -найти коэффициенты регрессии w0,w1 по шагам, как в уроке
    #  -посчитать предсказание в виде y^=w0+w1x и визуализировать его вместе с точками X_hw и Y_hw

    import numpy as np
    from numpy.linalg import inv
    from matplotlib import pyplot as plt

    X_hw = np.array([[1, 50], [1, 60], [1, 70], [1, 100]])
    Y_hw = np.array([[10], [15], [40], [45]])

    # # визуализируем набор точек
    # plt.scatter(X_hw[:,1], Y_hw[:,0], 40, 'g', 'o', alpha=0.8)
    # plt.show()

    # находик кэфы регрессии
    w = inv((X_hw.T).dot(X_hw)).dot(X_hw.T).dot(Y_hw)
    
    # print(w[0][0], '\n', w[1][0])
    margin = 10
    X_hw_min = X_hw[:,1].min() - margin
    X_hw_max = X_hw[:,1].max() + margin

    X_support = np.linspace(X_hw_min, X_hw_max, num=100)
    Y_model = w[0][0] + w[1][0]*X_support
    plt.xlim(X_hw_min, X_hw_max)
    plt.ylim(0, Y_hw[:,0].max() + margin)
    # исходные точки
    plt.scatter(X_hw[:,1], Y_hw[:,0], 40, 'g', 'o')
    # гадание 
    plt.plot(X_support, Y_model)
    plt.show()

    
def d6():
    # прода d4, прогон по ошибка MAE и MSE (модуль и квадрат ошибок там все понятно)
    from sklearn.datasets import fetch_openml
    import numpy as np
    boston_dataset = fetch_openml(name='boston', version=1, as_frame=False, parser='auto')
    
    features =  np.array(boston_dataset.data, dtype=float)
    y = np.array(boston_dataset.target)

    from sklearn.linear_model import LinearRegression

    # обучаем модель "из коробки"
    reg = LinearRegression().fit(features, y)
    y_pred = reg.predict(features)
    y_true = y
    from sklearn.metrics import mean_absolute_error

    print("MAE = %s" % mean_absolute_error(
        reg.predict(features), y)
    )
    from sklearn.metrics import mean_squared_error

    mse = mean_squared_error(y_true, y_pred)

    print('MSE = %s' % mse)

    from sklearn.metrics import r2_score

    print("r2_score = %s" % r2_score(y_true, y_pred))


def d7():
    #У вас есть два набора точек – истинные значения y_true и предсказанные значения y_pred

    # для каждой точки из y_true постройте величину ошибки e=y−y^ – это называется остатки регрессии
    # возведите ошибки в квадрат e2
    # постройте график ошибок – зависимость e2 от e

    import numpy as np
    from matplotlib import pyplot as plt
    from sklearn.linear_model import LinearRegression

    y_pred = np.array([30.0, 25.03, 30.57, 28.61, 27.94, 25.26, 23.0, 19.54, 11.52, 18.92, 19.0, 21.59, 20.91, 19.55, 19.28, 19.3, 20.53, 16.91, 16.18, 18.41, 12.52, 17.67, 15.83, 13.81, 15.68, 13.39, 15.46, 14.71, 19.55, 20.88, 11.46, 18.06, 8.81, 14.28, 13.71, 23.81, 22.34, 23.11, 22.92, 31.36])
    y_true = np.array([24.0, 21.6, 34.7, 33.4, 36.2, 28.7, 22.9, 27.1, 16.5, 18.9, 15.0, 18.9, 21.7, 20.4, 18.2, 19.9, 23.1, 17.5, 20.2, 18.2, 13.6, 19.6, 15.2, 14.5, 15.6, 13.9, 16.6, 14.8, 18.4, 21.0, 12.7, 14.5, 13.2, 13.1, 13.5, 18.9, 20.0, 21.0, 24.7, 30.8])

    err = y_true - y_pred # в видосе надо в скобки и знак минус хотя к дз написано что все верно, но мб err = -(y_true - y_pred) хотя с хуя ли
    square_err = np.square(err)

    ids = np.argsort(err)

    plt.scatter(err[ids], square_err[ids])
    plt.show()


def d8():
    # ФОРМАЛИЗАЦИЯ, данные могут быть кривоватыми, условно 1-2-3-4-120-6, и крч чтобы это сгладить и построить более верную
    # модель линейной регрессии можно возвести массив наш в квадрат, либо взять крень (тут очевидно корень папизже)
    # Ну это базово, а на деле юзается z-score, он сам типо сглаживает.
    import numpy as np
    from matplotlib import pyplot as  plt

    x = np.linspace(1, 10, num=10).reshape(-1,1)
    y = [
        1.5,
        2.5,
        3,
        4.5,
        12,
        6.7,
        7,
        8.5,
        14,
        7
    ]
    # # вот у нас все криво
    # plt.scatter(x, y)
    # plt.show()

    # # у чим и смотрим, все еще криво на пизже
    # from sklearn.linear_model import LinearRegression
    # from sklearn.metrics import r2_score, mean_squared_error

    # reg = LinearRegression().fit(x, y)
    # y_pred = reg.predict(x)

    # print(mean_squared_error(y, y_pred))
    # y_transformed = np.log(y)

    # plt.scatter(x, y_transformed)
    # plt.show()

    # Посмотрим на z-score сглаживание
    from sklearn.preprocessing import StandardScaler

    raw_data = np.array([
        1.,  3.,  2.,  4.,  2., 10.,  2.,  5.,  2.,  2.,  1.,  7.,  5.,  2.,  5., 16., 10.,  3.,24.],
        dtype=np.float32
    )

    print("Сырой датасет: %s" % raw_data)

    transformed_data = StandardScaler().fit_transform(raw_data.reshape(-1, 1)).reshape(-1)
    print("z-transform датасет: %s" % transformed_data)
    # Еще есть min-max нормализация, там логично если формулу посмотреть, Arr = (x - x.min) / (x.max - x.min), все значения будут от 0 до 1
    # и это сгладит углы (хотя все еще реальная картина 0.111113, 0.111334, 0.111112, 1. Ну тут уже вопрос хули тут 1 делает)
    from sklearn.preprocessing import MinMaxScaler


    print("Сырой датасет: %s" % raw_data)

    transformed_data = MinMaxScaler().fit_transform(raw_data.reshape(-1, 1)).reshape(-1)

    print("Min-Max scale датасет: %s" % transformed_data)


def d9():
    # ДЗШКА
    # Даны точки  x  и значения в этих точках  y . 
    # Нормализуйте  y  с помощью z-score и постройте график зависимости нормализованных значений от  x . 
    # Для графика используйте .scatter(x,y_tansformed)
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from matplotlib import pyplot as plt

    x = np.linspace(1,10,num=10)
    y = np.array(
        [1.,  3.,  4.,  2., 10.,  5.,  5.,  2.,  5., 10.],
        dtype=np.float32
    )
    transformed_data = StandardScaler().fit_transform(y.reshape(-1, 1)).reshape(-1)
    plt.scatter(x, transformed_data, c = 'g')
    # plt.scatter(x, y, c = 'b')
    plt.scatter(10, 10, alpha=0) # для удержания графика на y = 10

    plt.show()


def d10():
    # нелинейная линейность (Полиномиальная регрессия)
    from matplotlib import pyplot as plt
    import numpy as np
    import pandas as pd

    def generate_degrees(source_data: list, degree: int):
        """Функция, которая принимает на вход одномерный массив, а возвращает n-мерный
        
        Для каждой степени от 1 до degree возводим x в эту степень
        """
        return np.array([
            source_data**n for n in range(1, degree + 1)  
        ]).T
    

    def train_polynomial(degree, data):
        from sklearn.metrics import mean_squared_error
        """Генерим данные, тренируем модель
        
        дополнительно рисуем график
        """
        
        X = generate_degrees(data['x_train'], degree)

        model = LinearRegression().fit(X, data['y_train'])
        y_pred = model.predict(X)

        error = mean_squared_error(data['y_train'], y_pred)
        print("Степень полинома %d Ошибка %.3f" % (degree, error))

        plt.scatter(data['x_train'], data['y_train'], 40, 'g', 'o', alpha=0.8, label='data')
        plt.plot(data['x_train'], y_pred)
        # plt.show()
        return error
    

    data = pd.read_csv('3.10_non_linear.csv', sep=',')
    plt.scatter(data.x_train, data.y_train, 40, 'b', 'o')
    # print(data['x_train'])
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(data[['x_train']], data.y_train)
    y_hat = reg.predict(data[['x_train']])
    # plt.plot(data.x_train, y_hat)
    
    # degree = 4
    # X = generate_degrees(data['x_train'], degree)
    # print(X)
    # plt.show()
    predict_best = [1, 1.]
    for degree in range(1, 25 + 1):
        error = train_polynomial(degree, data)
        if error < predict_best[1]:
            predict_best = [degree, error]
                            
    print(predict_best)
    




def d11():
    # реализуем функции fit и predict своими руками (как в начале самом)
    from matplotlib import pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler

    houses = pd.read_csv("1.4_houses_train.csv")
    houses_2 = pd.read_csv("1.4_houses_valid.csv")
    reg = LinearRegression().fit(houses[['dim_1']], houses['price'])
    reg2 = LinearRegression().fit(houses_2[['dim_1']], houses_2['price'])
    error = r2_score(reg2.predict(houses_2[['dim_1']]), reg.predict(houses_2[['dim_1']]))
    r2_score_no_help = error
    
    transformed_1_dim_1 = StandardScaler().fit_transform(houses[['dim_1']])
    transformed_2_dim_1 = StandardScaler().fit_transform(houses_2[['dim_1']])
    reg_tr = LinearRegression().fit(transformed_1_dim_1, houses['price'])
    reg2_tr = LinearRegression().fit(transformed_2_dim_1, houses_2['price'])
    error = r2_score(reg2_tr.predict(houses_2[['dim_1']]), reg_tr.predict(houses_2[['dim_1']]))
    r2_score_z_preobr = error
    print(r2_score_no_help, r2_score_z_preobr)


def d12():
    from matplotlib import pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    # from sklearn.linear_model import LinearRegression
    # from sklearn.metrics import r2_score
    # from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.metrics import mean_squared_error

    def generate_degrees(source_data: list, degree: int):
        """Функция, которая принимает на вход одномерный массив, а возвращает n-мерный
        Для каждой степени от 1 до  degree возводим x в эту степень
        """
        return np.array([
            source_data**n for n in range(1, degree + 1)  
        ]).T
    
    data = pd.read_csv('3.10_non_linear.csv', sep=',')

    def test_educate(degree):
        # degree = 8
        X = generate_degrees(data['x_train'], degree)
        y = data.y_train.values

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=10)
        model = Ridge(alpha=0.5).fit(X_train, y_train) # меняй alpha для сглаживания переобучения
        y_pred = model.predict(X_valid)
        y_pred_train = model.predict(X_train)
        print("Качество на валидации: %.3f" % mean_squared_error(y_valid, y_pred))
        print("Качество на обучении: %.3f" % mean_squared_error(y_train, y_pred_train))

    test_educate(8)

def generate_degrees(source_data: list, degree: int):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from numpy.linalg import norm
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    
    """Функция, которая принимает на вход одномерный массив, а возвращает n-мерный
    Для каждой степени от 1 до  degree возводим x в эту степень
    """
    return np.array([
        source_data**n for n in range(1, degree + 1)  
    ]).T


def train_polynomial(degree, data):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from numpy.linalg import norm
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    
    """Генерим данные, тренируем модель  
    дополнительно рисуем график
    """
    X = generate_degrees(data['x_train'], degree)
    y = data.y_train.values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=10)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    y_pred_train = model.predict(X_train)
    error_valid = mean_squared_error(y_valid, y_pred)
    error_train = mean_squared_error(y_train, y_pred_train)
    print(
        "Степень полинома %d\nОшибка на валидации %.3f\nОшибка на обучении %.3f" %
        (degree, error_valid, error_train)
    )
    order_test = np.argsort(X_valid[:,0])
    plt.scatter(X_valid[:,0][order_test], y_valid[order_test], 40, 'r', 'o', alpha=0.8)
    print("Норма вектора весов \t||w|| = %.2f" % (norm(model.coef_)))
    # визуализируем решение
    x_linspace = np.linspace(data['x_train'].min(), data['x_train'].max(), num=100)
    y_linspace = model.predict(generate_degrees(x_linspace, degree))
    plt.plot(x_linspace, y_linspace)
    return error_valid, error_train, norm(model.coef_)


def d13():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from numpy.linalg import norm
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    
    data = pd.read_csv('3.10_non_linear.csv', sep=',')
    # основной график
    plt.scatter(data.x_train, data.y_train, 40, 'g', 'o', alpha=0.8)
    plt.show()


    

    degrees = []
    valid_errors = []
    train_errors = []
    w_norm = []

    degree = 3

    error_valid, error_train, coef_norm = train_polynomial(degree, data)

    degrees.append(degree)
    valid_errors.append(error_valid)
    train_errors.append(error_train)
    w_norm.append(coef_norm)


def d14():
    import numpy as np
    from numpy.linalg import inv
    from matplotlib import pyplot as plt
    X = np.array( [[1, 50], [1, 60], [1, 70], [1, 100]] ) 
    Y = np.array( [[10], [30], [40], [50]] )
    alpha_ = 0
    # w = inv((X.T).dot(X) + alpha_*np.eye(len(X)//2)).dot(X.T).dot(Y) # легендарная формула Линейной регресси на Python 
    # y_p
    w = inv((X.T).dot(X) + alpha_*np.eye(len(X)//2)).dot(X.T).dot(Y)

    # ебашим предсказание (как ванга)
    
    #задаем границы
    margin = 10 # граница
    X_min = X[:,1].min() - margin
    X_max = X[:,1].max() + margin

    # наблор точек для прямой 
    X_support = np.linspace(X_min, X_max, num=100) # num - кол-во точек
    # Гадаем
    Y_model = w[0][0] + w[1][0]*X_support # это реализация функции y = ax + b
    plt.xlim(X_min, X_max)
    plt.ylim(0, Y[:,0].max() + margin)
    # исходные точки
    plt.scatter(X[:,1], Y[:,0], 40, 'g', 'o')
    # гадание 
    plt.plot(X_support, Y_model)
    plt.show()

def d15():
    import numpy as np
    from numpy.linalg import inv
    from matplotlib import pyplot as plt
    from sklearn.linear_model import Ridge
    import pandas as pd
    from numpy.linalg import norm
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    #plt.scatter(data['x_train'], data['y_train'], 40, 'g', 'o', alpha=0.8, label='data')
    data = pd.read_csv('3.10_non_linear.csv', sep=',')

    model_ridge = Ridge(alpha=0.01)
    model_linear = Ridge(alpha=0.0)
    degree = 10

    X = generate_degrees(data['x_train'], degree)
    y = data['y_train']
    # обучаем линейную регрессию с  регуляризацией
    model_ridge.fit(X, y)
    model_linear.fit(X, y)

    x_linspace = np.linspace(data['x_train'].min(), data['x_train'].max(), num=100)

    y_linspace_linear = model_linear.predict(generate_degrees(x_linspace, degree))
    y_linspace_ridge = model_ridge.predict(generate_degrees(x_linspace, degree))

    plt.plot(x_linspace, y_linspace_linear)
    plt.plot(x_linspace, y_linspace_ridge)
    plt.show()
    print("Норма вектора весов Ridge \t||w|| = %.2f" % (norm(model_ridge.coef_)))
    print("Норма вектора весов Linear \t||w|| = %.2f" % (norm(model_linear.coef_)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # print(X_train.shape, X_test.shape)
    alphas = np.arange(0.1, 1, 0.01)

    best_alpha = alphas[0]
    best_rmse = np.infty

    for alpha in alphas:
        model_ridge = Ridge(alpha=alpha)
        # обучаем линейную регрессию с  регуляризацией
        model_ridge.fit(X_train, y_train)
        y_pred = model_ridge.predict(X_test)
        error = mean_squared_error(y_test, y_pred)
        if error < best_rmse:
            best_rmse = error
            best_alpha = alpha
        print("alpha =%.2f Ошибка %.5f" % (alpha, error))
    print('\n-------\nЛучшая модель aplpha=%.2f с ошибкой RMSE=%.5f\n-------' % (best_alpha, best_rmse))
    

def d16():
    import numpy as np
    from numpy.linalg import inv
    from matplotlib import pyplot as plt
    from sklearn.linear_model import Ridge
    import pandas as pd
    from numpy.linalg import norm
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import SGDRegressor
    from scipy.spatial import distance

    data = pd.read_csv('3.10_non_linear.csv', sep=',')
    X = data['x_train'].values.reshape(-1, 1)
    y = data['y_train']
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    sgd_regressor = SGDRegressor(learning_rate='constant', eta0=0.009, fit_intercept=True, random_state=42)
    w_current, epsilon, weight_evolution, rmse_evolution = np.random.random(2), 0.001, [], []
    for step in range(800):
        sgd_regressor = sgd_regressor.partial_fit(X_train, y_train)
        weight_evolution.append(distance.euclidean(w_current, sgd_regressor.coef_))
        if weight_evolution[-1] < epsilon:
            print('STOP ITERRATION OM STEP %d' % step);break
        rmse_evolution.append(
            mean_squared_error(y_valid, sgd_regressor.predict(X_valid))
        )
        w_current = sgd_regressor.coef_.copy()
    # plt.plot(range(step), rmse_evolution)
    # plt.show()
    x_linspace = np.linspace(data['x_train'].min(), data['x_train'].max(), num=100)

    y_linspace= sgd_regressor.predict(x_linspace.reshape(-1,1))

    plt.plot(x_linspace, y_linspace)
    plt.scatter(data.x_train, data.y_train, 40, 'g', 'o', alpha=0.8, label='data')

    plt.show()


def d17():
    import numpy as np
    from numpy.linalg import inv
    from matplotlib import pyplot as plt
    from sklearn.linear_model import Ridge
    import pandas as pd
    from numpy.linalg import norm
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import SGDRegressor
    from scipy.spatial import distance
    from sklearn.datasets import fetch_openml
    boston_dataset = pd.read_csv('3.10_non_linear.csv', sep=',')
    X = boston_dataset['x_train'].values.reshape(-1, 1)
    y = boston_dataset.y_train
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=10)

    sgd_regressor = SGDRegressor(learning_rate='constant', eta0=0.009, fit_intercept=True, random_state=42)
    w_current, epsilon, weight_evolution, rmse_evolution = np.random.random(13), 0.001, [], []
    for step in range(800):
        sgd_regressor = sgd_regressor.partial_fit(X_train, y_train)
        weight_evolution.append(distance.euclidean(w_current, sgd_regressor.coef_))
        if weight_evolution[-1] < epsilon:
            print('STOP ITERRATION OM STEP %d' % step);break
        rmse_evolution.append(
            mean_squared_error(y_valid, sgd_regressor.predict(X_valid))
        )
        w_current = sgd_regressor.coef_.copy()
    plt.plot(range(step), rmse_evolution)
    plt.show()  
    # X = X.apply(pd.to_numeric, errors='coerce')
    # x_linspace = np.linspace(X.min(), X.max(), num=100)

    # y_linspace= sgd_regressor.predict(x_linspace.reshape(-1,1))

    # plt.plot(x_linspace, y_linspace)
    # plt.scatter(X, y, 40, 'g', 'o', alpha=0.8, label='data')

    # plt.show()


def d18():
    import numpy as np
    from numpy.linalg import inv
    from matplotlib import pyplot as plt
    from sklearn.linear_model import Ridge
    import pandas as pd
    from numpy.linalg import norm
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import SGDRegressor
    from scipy.spatial import distance

    data = pd.read_csv('3.10_non_linear.csv', sep=',')
    X = data['x_train'].values.reshape(-1, 1)
    y = data['y_train']



def main():

    d18()


if __name__ == '__main__':
    main()