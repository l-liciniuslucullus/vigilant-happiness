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
    ffs = [features_funcs.fwhr]

    test_data = False

    for feature_func in ffs:
        print('get features (with gender)', flush=True)
        t0 = time()
        if test_data:
            features = get_features("../data/testdata.csv", feature_func,
                                    "../data/testdata_labels.txt",
                                    None, face_id_clean=True, limit_deg=180)
        else:
            features = get_features("../data/data.csv", feature_func,
                                    "../data/data_labels.txt",
                                    "../data/gender.csv", face_id_clean=False, limit_deg=1)
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
        if test_data:
            np.save(feature_func.__name__+'test', arr)
        else:
            np.save(feature_func.__name__, arr)
        print(time() - t0)


if __name__ == '__main__':
    main()
