import numpy as np
from get_features import get_features, get_gender
import matplotlib.pyplot as plt
from utils import unroll, normalize, normalize_by_eyes
from features_funcs import pure_landmarks


def split_genders(faces):
    g0 = [row[:-1] for row in faces.values() if row[-1] == 0]
    g1 = [row[:-1] for row in faces.values() if row[-1] == 1]
    return np.array(g0), np.array(g1)


def main():
    print('getting faces')
    g1, g2 = split_genders(
        get_features("../data/testdata.csv", pure_landmarks,
                     "../data/testdata_labels.txt",
                     None, face_id_clean=True, limit_deg=180)
    )

    # g1, g2 = split_genders(
    #     get_features("../data/data.csv", pure_landmarks,
    #                  "../data/data_labels.txt",
    #                  "../data/gender.csv", face_id_clean=False, limit_deg=1)
    # )

    # g1 = g1[:1000]
    # g2 = g2[:1000]
    print('averaging')
    g1 = [(x, y) for x, y in zip(np.mean(g1, 0)[::2], np.mean(g1, 0)[1::2])]
    g2 = [(x, y) for x, y in zip(np.mean(g2, 0)[::2], np.mean(g2, 0)[1::2])]
    g1 = np.array(g1)
    g2 = np.array(g2)
    # np.save('g0_2deg', g1)
    # np.save('g1_2deg', g2)
    plt.plot(g1[:, 0], g1[:, 1], '.')
    # plt.savefig('gender0.png')
    # plt.clf()
    plt.plot(g2[:, 0], g2[:, 1], '.')
    # plt.savefig('gender1.png')
    plt.show()


if __name__ == '__main__':
    main()
