from time import time
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA, KernelPCA
# from sklearn.linear_model import Perceptron
# from sklearn.model_selection import train_test_split


def get_data(filename):
    print('getting data: ', filename, flush=True)
    t0 = time()
    data = np.load(filename)
    X = data[:, 0:-1]
    y = data[:, -1]
    # print('time:', time() - t0)
    return X, y
# print(len(X))

# t0 = time()
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.1, random_state=44)
# print(time() - t0)


def get_two_data(filename1, filename2):
    print('getting data: ', filename1, filename2)
    t0 = time()
    data1 = np.load(filename1)
    data2 = np.load(filename2)
    X = np.hstack((data1[:, 0:-1], data2[:, 0:-1]))
    y = data1[:, -1]
    # print('time:', time() - t0)
    return X, y


def logreg(X, y):
    # print('logreg')
    t0 = time()
    LR = LogisticRegressionCV(cv=10, Cs=10, n_jobs=7)
    LR.fit(X, y)
    y_pred = LR.predict(X)
    np.savetxt('predictions', y_pred)
    sc = roc_auc_score(y, y_pred)
    # print('time: ', time() - t0)
    print(
        'score: ', sc
    )
    return sc


def reduce_dim(X, n_dim=5):
    print('reducing dimensionality '+str(n_dim))
    t0 = time()
    pca = PCA(n_components=n_dim)
    # pca = KernelPCA(n_components=5, kernel='linear')
    X = pca.fit_transform(X)
    # print(time() - t0)
    return X


def main():
    # logreg(*get_data('../data/fwhr_gender.npy'))
    # logreg(*get_data('../data/fwhr_gender_test.npy'))
    logreg(*get_data('../data/fwhr_gender_test_michal.npy'))
    logreg(*get_data('../data/ellipse_picked_test.npy'))
    # logreg(*get_data('../data/fwhr_gender_test_altfwhr.npy'))
    X, y = get_data('../data/ellipsetest.npy')
    # l = ['center_x', 'center_y', 'width', 'height', 'phi']
    # for ii in range(5):
    #     print(l[ii])
    #     logreg(X[:, ii].reshape(-1, 1), y)
    logreg(X, y)
    # logreg(X, y)
    logreg(reduce_dim(X, 2), y)
    print('only y coord of the ellipse')
    logreg(X[:, 1].reshape(-1, 1), y)

    X, y = get_data('../data/polyfit6_test.npy')
    logreg(X, y)
    logreg(reduce_dim(X, 5), y)

    X, y = get_data('../data/butterfly_catastrophe_test.npy')
    logreg(X, y)
    logreg(reduce_dim(X, 3), y)

    # logreg(*get_data('../data/eccentricity_test.npy'))
    # logreg(*get_data('../data/rectum_test.npy'))
    # logreg(*get_data('../data/dists_contour_test.npy'))
    logreg(*get_data('../data/face_contour_test.npy'))
    # logreg(*get_data('../data/dists_2_refs_contour_test.npy'))
    # logreg(*get_data('../data/lips_whr_gender.npy'))
    # logreg(*get_data('../data/random_gender.npy'))
    # logreg(*get_data('../data/face_triangle_to_circumcircle_ratio_gender.npy'))

    # logreg(*get_data('../data/distances_gender.npy'))
    # logreg(*get_data('../data/hand_picked_dists_gender.npy'))
    # logreg(*get_data('../data/dists_from_2_refs_gender.npy'))

    # ffs = ['dists_2_refs_lips',
    #        'dists_2_refs_lower_lip',
    #        'dists_2_refs_contour',
    #        'dists_2_refs_contour_top4',
    #        'dists_2_refs_nose',
    #        'dists_2_refs_nose_top4',
    #        'dists_2_refs_eyebrows',
    #        'dists_2_refs_eyebrows_corners',
    #        'dists_2_refs_eyes']

    # ffs = ['dists_lips',
    #        'dists_lower_lip',
    #        'dists_contour',
    #        'dists_contour_top4',
    #        'dists_nose',
    #        'dists_nose_top4',
    #        'dists_eyebrows',
    #        'dists_eyebrows_corners',
    #        'dists_eyes']

    # for f in ffs:
    #     logreg(*get_data('../data/'+f+'_gender.npy'))

    # logreg(*get_data('distances_normalized_gender.npy'))
    # logreg(*get_data('distances_normalized_by_eyes_gender.npy'))


if __name__ == '__main__':
    main()
