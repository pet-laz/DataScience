# разложение на сингулярное сложение
from sklearn.decomposition import TruncatedSVD

import numpy as np
from matplotlib import pyplot as plt
import pickle
def d1():

    with open('8.8_eigen.pkl', 'rb') as f:
        X = pickle.load(f)

    # plt.plot(X[:,0], X[:,1], 'x')
    # plt.axis('equal')
    # plt.show()

    svd_model = TruncatedSVD(n_components=1).fit(X)
    X_svd = svd_model.transform(X)
    X_svd_inverted = svd_model.inverse_transform(X_svd)

    plt.plot(X_svd_inverted[:,0], X_svd_inverted[:,1], 'x')
    plt.show()


def dz1():
    from numpy.linalg import svd, det

    A = np.array([[3,2,2],[2,3,-2]])

    U, S, VT = svd(A)
    # print(U)
    det_U = det(U)
    print(det_U)


def dz2(n):
    from matplotlib.pyplot import imshow
    import matplotlib.image as mpimg
    img = mpimg.imread('dorian_grey.png')
    # print(type(img),img.shape)
    # imshow(img)
    # plt.show()
    def rgb2gray(rgb): # на деле нихуя не gray, просто серым воспринимат приятнее, средний цвет - зеленый +- (на фотке его много)
        #''' Берётся среднее трёх цветов RGB'''
        tile = np.tile(np.c_[0.333, 0.333, 0.333], reps=(rgb.shape[0],rgb.shape[1],1))
        return np.sum(tile * rgb, axis=2)

    img_gray = rgb2gray(img) 
    # print(type(img_gray), img_gray.shape)
    # imshow(img_gray, cmap = "gray")
    # plt.show()
    svd_model = TruncatedSVD(n_components=n).fit(img_gray) # n_components - кол-во фичей, которое мы оставляем
    img_gray_svd = svd_model.transform(img_gray) 
    print(img_gray_svd.shape)
    img_gray_svd_restored = svd_model.inverse_transform(img_gray_svd)
    # print(type(img_gray_svd_restored),img_gray_svd_restored.shape)
    imshow(img_gray_svd_restored, cmap = "gray")
    plt.show()


def dz3(n): #PCA ( по рофлу попробывал +- одинаково робят)
    from sklearn.decomposition import PCA

    from matplotlib.pyplot import imshow
    import matplotlib.image as mpimg
    img = mpimg.imread('dorian_grey.png')
    # print(type(img),img.shape)
    # imshow(img)
    # plt.show()
    def rgb2gray(rgb): # на деле нихуя не gray, просто серым воспринимат приятнее, средний цвет - зеленый +- (на фотке его много)
        #''' Берётся среднее трёх цветов RGB'''
        tile = np.tile(np.c_[0.333, 0.333, 0.333], reps=(rgb.shape[0],rgb.shape[1],1))
        return np.sum(tile * rgb, axis=2)

    img_gray = rgb2gray(img) 
    # print(type(img_gray), img_gray.shape)
    # imshow(img_gray, cmap = "gray")
    # plt.show()
    svd_model = PCA(n_components=n).fit(img_gray) # n_components - кол-во фичей, которое мы оставляем
    img_gray_svd = svd_model.transform(img_gray) 
    print(img_gray_svd.shape)
    img_gray_svd_restored = svd_model.inverse_transform(img_gray_svd)
    # print(type(img_gray_svd_restored),img_gray_svd_restored.shape)
    imshow(img_gray_svd_restored, cmap = "gray")
    plt.show()


def main():
    n = 15
    dz2(n)


if __name__ == '__main__':
    main()
