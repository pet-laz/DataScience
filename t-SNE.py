def d1():

    import pandas as pd

    df_source = pd.read_csv('8.8_client_segmentation.csv')

    # print(df_source.head())

    X = df_source[['call_diff', 'sms_diff', 'traffic_diff']].values
    y = df_source.customes_class.values

    from sklearn.manifold import TSNE

    tsne_transformer = TSNE(n_components=2)

    x_tsne = tsne_transformer.fit_transform(X)

    # print(x_tsne[:10])

    colors = ['bo', 'gx', 'ro']
    num_labels = 3
    # И нарисуем получившиеся точки в нашем новом пространстве
    from matplotlib import pyplot as plt
    for name, label, color in [('class_%d' % i, i, colors[i]) for i in range(num_labels)]:
        plt.plot(x_tsne[y == label, 0], x_tsne[y == label, 1], color, label=label)
    plt.legend(loc=0)
    plt.show()


def dz1():
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score

    def eval_model(input_x, input_y):
        """Обучаем и валидируем модель"""
        X_train, X_test, y_train, y_test = train_test_split(
            input_x, input_y, test_size=.3, stratify=y, random_state=42
        )
        # Для примера возьмём неглубокое дерево решений
        clf = DecisionTreeClassifier(max_depth=2, random_state=42)  
        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_test)
        acc_score = accuracy_score(y_test, preds.argmax(axis=1))
        print('Accuracy: %.5f' % acc_score)
        
    df_source = pd.read_csv('8.8_client_segmentation.csv')
    X = df_source[['call_diff','sms_diff','traffic_diff']].values
    y = df_source.customes_class.values

    eval_model(X, y)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2).fit(X)
    X_pca = pca.transform(X)

    eval_model(X_pca, y)

    from sklearn.decomposition import TruncatedSVD

    svd = TruncatedSVD(n_components=2).fit(X)
    X_svd = svd.transform(X)
    eval_model(X_svd, y)

    from sklearn.manifold import TSNE

    X_tsne = TSNE(n_components=2).fit_transform(X)

    eval_model(X_tsne, y)


def main():
    dz1()


if __name__ == '__main__':
    main()