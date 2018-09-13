from time import time
import numpy as np
from get_features import get_features
import features_funcs


def main():
    # ffs = [features_funcs.dists_2_refs_lips,
    #        features_funcs.dists_2_refs_lower_lip,
    #        features_funcs.dists_2_refs_contour,
    #        features_funcs.dists_2_refs_contour_top4,
    #        features_funcs.dists_2_refs_nose,
    #        features_funcs.dists_2_refs_nose_top4,
    #        features_funcs.dists_2_refs_eyebrows,
    #        features_funcs.dists_2_refs_eyebrows_corners,
    #        features_funcs.dists_2_refs_eyes]
    # ffs = [features_funcs.dists_lips,
    #        features_funcs.dists_lower_lip,
    #        features_funcs.dists_contour,
    #        features_funcs.dists_contour_top4,
    #        features_funcs.dists_nose,
    #        features_funcs.dists_nose_top4,
    #        features_funcs.dists_eyebrows,
    #        features_funcs.dists_eyebrows_corners,
    #        features_funcs.dists_eyes]
    ffs = [features_funcs.distances_normalized_by_eyes]

    for feature_func in ffs:
        print('get labels', flush=True)
        t0 = time()
        labels = np.loadtxt(
            "../data/labels", delimiter="\n", dtype=np.str)[:: 2]
        print(time() - t0)

        print('get features (with gender)', flush=True)
        t0 = time()
        features = get_features('../data/temp.csv', feature_func,
                                labels, '../data/gender.csv')
        print(time() - t0)

        feature_length = 0
        for v in features.values():
            feature_length = len(v)
            break

        print('save as np array')
        t0 = time()
        arr = np.zeros((len(features), feature_length))
        for row, v in enumerate(features.values()):
            arr[row] = v
        np.save(feature_func.__name__+'_gender', arr)
        print(time() - t0)


if __name__ == '__main__':
    main()
