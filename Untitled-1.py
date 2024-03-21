def d1():
    import numpy as np
    from matplotlib import pyplot as plt
    import pickle

    with open('8.8_eigen.pkl', 'rb') as f:
        X = pickle.load(f)

    
    # plt.plot(X[:,0], X[:,1], 'x')
    # plt.axis('equal')
    # plt.show()
        
    # print(X.shape)

    from sklearn.decomposition import PCA

    pca = PCA(n_components=1).fit(X)

    X_pca = pca.transform(X)

    # print(X_pca[:10])

    X_new = pca.inverse_transform(X_pca)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(X_new[:,0], X_new[:,1], 'x')
    plt.subplot(212)
    plt.plot(X[:,0], X[:,1], 'o')
    plt.show()


def d2():
    import numpy as np
    from matplotlib import pyplot as plt
    import pandas as pd
    from sklearn.decomposition import PCA


    # мое решение
    data = pd.read_csv('8.8_client_segmentation.csv')
    X = np.column_stack((data.call_diff.values, data.traffic_diff.values))
    pca = PCA(n_components=1).fit(X)
    X_pca = pca.transform(X)
    X_new = pca.inverse_transform(X_pca)
    plt.plot(data.call_diff.values, data.traffic_diff.values, 'o')
    plt.plot(X_new[:,0], X_new[:,1], 'x')
    plt.show()

    # # Интернет решение из овтетов
    # data = pd.read_csv('8.8_client_segmentation.csv')
    # X = data[['call_diff', 'traffic_diff']].values
    # y = data.customes_class.values
    # pca = PCA(n_components=1).fit(X)
    # X_pca = pca.transform(X)
    # X_new = pca.inverse_transform(X_pca)
    # plt.plot(data.call_diff, data.traffic_diff, 'o')
    # plt.plot(X_new[:,0], X_new[:,1], 'x')
    # plt.show()


def d3():
    import numpy as np
    from matplotlib import pyplot as plt
    import pandas as pd
    from sklearn.decomposition import PCA

    data = pd.read_csv('8.8_client_segmentation.csv')
    X = np.column_stack((data.call_diff.values, data.sms_diff.values, data.traffic_diff.values))
    y = data.customes_class.values
    pca = PCA(n_components=2).fit(X)
    X_pca = pca.transform(X)
    colors = ['r', 'g', 'b']  # Цвета для каждого класса
    markers = ['$У$', '$Й$', '$Х$']
    for i, cls in enumerate(np.unique(y)):
        plt.scatter(X_pca[y == cls, 0], X_pca[y == cls, 1], marker=markers[i], c=colors[i], label=cls)
    plt.show()


def main():
    d3()


if __name__ == '__main__':
    main()

    # data = pd.read_csv('8.8_client_segmentation.csv')
    # X = data[['call_diff', 'traffic_diff']].values
    # y = data.customes_class.values
    # pca = PCA(n_components=1).fit(X)
    # X_pca = pca.transform(X)
    # X_new = pca.inverse_transform(X_pca)
    # plt.plot(X_new[:,0], X_new[:,1], 'x')
    # plt.plot(data.call_diff, data.traffic_diff, 'o')
    # plt.show()