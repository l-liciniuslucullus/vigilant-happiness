from time import time
import numpy as np
from logreg import logreg, get_data
from avg_face_diff import k_max_norms
from sklearn.decomposition import PCA


def filter_points(data, point_idx_to_take):
    col_idx_to_take = [x for idx in point_idx_to_take for x in (idx, idx+1)]
    return data[:, col_idx_to_take]


def filter_with_pca(data, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)


def get_max_norm_args():
    g0 = np.load('../data/g0.npy')
    g1 = np.load('../data/g1.npy')
    g = g0 - g1
    args, _ = k_max_norms(g, k=len(g))
    return args


def main():
    t0 = time()
    args = get_max_norm_args()

    # labels = np.loadtxt("../data/labels", delimiter="\n", dtype=np.str)[:: 2]
    # labels = {ii: ll[:-2] for ii, ll in enumerate(labels)}

    X, y = get_data('../data/pure_landmarks_gender.npy')

    scores = []

    for k in reversed(range(1, len(args)+1)):
        print(k, flush=True)
        # scores.append(logreg(filter_points(X, args[:k]), y))
        scores.append(logreg(filter_with_pca(X, k), y))

    np.save('log_reg_scores_pca', np.array(scores))

    print(
        time() - t0
    )


if __name__ == '__main__':
    main()
