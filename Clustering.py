# КЛАСТЕРИЗАЦИЯ

def d1():
    from scipy.spatial.distance import euclidean

    c1 = [1.0, 1.5]
    c2 = [-1.0, -0.5]
    dist = euclidean(c1,c2)
    # расстояние между точками c1 и c2
    print(dist)


def d2():
    import pickle
    import numpy as np
    with open('7.10.clustering.pkl', 'rb') as file:
        data_clustering = pickle.load(file)
    # print(data_clustering)
    X = np.array(data_clustering['X'])
    Y = np.array(data_clustering['Y'])

    # # визуальчик
    # from matplotlib import pyplot as plt
    # plt.scatter(X[:,0], X[:,1], c = Y)
    # plt.show()
    
    from sklearn.cluster import KMeans
    kmeans_model = KMeans(n_clusters=2, random_state=42)
    kmeans_model.fit(X)
    from matplotlib import pyplot as plt
    plt.scatter(X[:,0], X[:,1], c = kmeans_model.labels_)
    plt.show()


def d3(n): 
    # ДЗШКА
    def first():
        import numpy as np

        clust = np.array([
        [-0.5972191612445694, -0.5919098916910581],
        [-1.5838020751869848, 0.4743393635868491],
        [-1.892642118066139, -1.2770390481464395],
        [-1.021761443519372, -0.38446059106320013],
        [-0.628271339507516, -2.308149346281125],
        [-0.7180915776856387, 1.1805550909358404],
        [-1.543143767548152, -1.4163791359687334],
        [0.022103701018375554, -1.1279478858865397],
        [-0.7776518874305778, -0.4157532453316538],
        [-1.2073577296008344, -1.1308523658604184]
        ])

        centroid = np.array([-0.9774245525274352, -0.032635425821084516])

        from scipy.spatial.distance import euclidean

        distance_ = []
        for x, y in clust:
            distance_.append(euclidean([x,y], centroid))
        print(distance_)
    def second():
            import pickle
            import numpy as np
            with open('7.10.clustering.pkl', 'rb') as file:
                data_clustering = pickle.load(file)
            X = np.array(data_clustering['X'])
            Y = np.array(data_clustering['Y'])
            from sklearn.cluster import KMeans
            kmeans_model = KMeans(n_clusters=3, random_state=42)
            kmeans_model.fit(X)
            from matplotlib import pyplot as plt
            plt.scatter(X[:,0], X[:,1], c = kmeans_model.labels_)
            plt.show()
    def third():
        import pickle
        import numpy as np
        centroinds = []
        with open('7.10._clustering.pkl', 'rb') as file:
            data_clustering = pickle.load(file)
        X = np.array(data_clustering['X'])
        Y = np.array(data_clustering['Y'])
        from sklearn.cluster import KMeans
        for i in range(10):
            kmeans_model = KMeans(n_clusters=2, n_init=1, random_state=None, algorithm='full', max_iter=2)
            kmeans_model.fit(X)
            centroinds.append(kmeans_model.cluster_centers_)
        # print(centroinds[:,0])
        from matplotlib import pyplot as plt
        plt.scatter(X[:,0], X[:,1], c = Y)
        plt.scatter([point[0][0] for point in centroinds], [point[0][1] for point in centroinds],s = 40, c = 'blue')
        plt.show()
    functions = {1: first, 2: second, 3: third} 
    if n in functions: 
        functions[n]()


def d4():
    from scipy.spatial.distance import euclidean
    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.cluster import KMeans
    import pickle
    from matplotlib import pyplot as plt
    import numpy as np
    with open('7.10._clustering.pkl', 'rb') as file:
        data_clustering = pickle.load(file)
    X = np.array(data_clustering['X'])
    Y = np.array(data_clustering['Y'])

    metrics = []
    MAX_CLUSTERS = 7

    for cluster_num in range(1, MAX_CLUSTERS):
        kmeans_model = KMeans(n_clusters=cluster_num, random_state=99, n_init=10).fit(X)
        centroids, labels = kmeans_model.cluster_centers_, kmeans_model.labels_
        metric = 0
        for centroid_label in range(cluster_num):
            metric += euclidean_distances(X[labels==centroid_label], centroids[centroid_label,:].reshape(1, -1)).sum(axis=0)[0]
        print(f'число кластеров: {cluster_num}, метрика: {metric}')
        metrics.append(metric)

    # kmeans_model
    D = []
    for i in range(0, len(metrics)-1):
        d = abs(metrics[i+1]-metrics[i])/abs(metrics[i]-metrics[i-1])
        D.append(d)
    print("Лучшее число кластеров: %s" % (np.argmin(D)+1))

    plt.plot([i+1 for i in range(len(metrics))], metrics)
    plt.show()


def d5():
    from scipy.spatial.distance import euclidean
    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.cluster import KMeans
    import pickle
    from matplotlib import pyplot as plt
    import numpy as np
    with open('7.10._clustering.pkl', 'rb') as file:
        data_clustering = pickle.load(file)
    X = np.array(data_clustering['X'])
    Y = np.array(data_clustering['Y'])

    def random_centroids_selection(X, n, k):# инициализируем центры класеров
        result = []
        random_selection = np.random.randint(0, n, size=k)
        for oject_id in random_selection:
            result.append(X[oject_id, :])
        return result
        
    def eval_weight_evolution(centroids_objects_prev, centroids_objects, k): # вычисляем как сдвинулись центры за 1 шаг алгоритма
        result  = []
        for i in range(0, k):
            dist = euclidean(
                centroids_objects_prev[i], 
                centroids_objects[i]
            )
            result.append(dist)
        return result

    def eval_cluster_labels(X, centroids_objects): # Вычисляем метки кластеров
        cluster_distance = euclidean_distances(X, centroids_objects)
        cluster_labels = cluster_distance.argmin(axis=1)
        return cluster_labels
    
    def eval_centroids(X, k, cluster_labels): # Вычисялем центроиды кластеров
        result = []
        for i in range(k):
            new_centriod = X[cluster_labels==i].mean(axis=0)
            result.append(new_centriod)
        return result

    def k_means(X: np.array, k: int=2, eps: float=0.001, num_iteration: int=10): # алгоритм к-средних с параметрами
        # centroids, clusters_labels = None, None
        try:
            n, m = X.shape
        except ValueError:
            print('Передан некорректный объект X')
        centroid_objects = random_centroids_selection(X, n, k)
        centroid_objects_prev = [np.zeros(m) for i in range(k)]
        weight_evolution = eval_weight_evolution(centroid_objects_prev, centroid_objects, k)
        # print(weight_evolution)
        step = 0

        while step < num_iteration and sum(weight_evolution[i] > eps for i in range(k)) != 0:
            centroid_objects_prev = centroid_objects.copy()
            # Вычисляем метки кластеров
            cluster_labels = eval_cluster_labels(X, centroid_objects)
            # Вычисляем центроиды кластеров
            centroid_objects = eval_centroids(X, k, cluster_labels)
            # на сколько сместились кластера
            weight_evolution = eval_weight_evolution(centroid_objects_prev, centroid_objects, k)
            print(f'шаг: {step}, смещение кластеров: {weight_evolution}')
            step += 1
        return np.vstack(centroid_objects), cluster_labels
    
    centroids, clusters_labels = k_means(X, k=2, num_iteration=10)    
    plt.scatter(X[:, 0], X[:, 1], s=40, c=Y, marker='o', alpha=0.8, label='data')
    plt.plot(centroids[:, 0], centroids[:, 1], marker='+', mew=10, ms=20)
    plt.show()

    


def main():
    #d3(1)
    d5()


if __name__ == '__main__':
    main()