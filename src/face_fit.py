import numpy as np
import pandas as pd

import utils as ut
from features_funcs import face_contour
from get_features import get_labelled_landmarks
from utils import facing_straight, labels_to_col_nr, normalize_by_eyes, unroll


def get_jawline():
    col_names = ut.labels_to_col_nr('../data/data_labels.txt')[1]
    data = pd.read_csv('../data/data.csv', header=None,
                       names=col_names, engine='c', nrows=1)

    jawline = data.filter(like='contour_', axis=1)
    return jawline


def get_jawline_normalized(file, labels_file, face_id_clean=False, limit_deg=1):
    features = {}
    col_nr, labels = labels_to_col_nr(labels_file)
    with open(file, 'r') as data_file:
        for row in data_file:
            data = row.split(',')
            face_id, labelled_data, ok = get_labelled_landmarks(
                data, col_nr, labels, face_id_clean=face_id_clean, limit_deg=limit_deg)
            if not ok:
                continue
            labelled_data = unroll(
                labelled_data, np.float(data[col_nr['headpose.roll_angle']]))
            labelled_data = normalize_by_eyes(labelled_data)
            features[face_id] = face_contour(labelled_data)
    return features


def save_jaws(file, labels_file, face_id_clean=False, limit_deg=1):
    print('get jaws')
    jaws = get_jawline_normalized(file, labels_file, face_id_clean, limit_deg)
    # feature_length = 0
    # for v in jaws.values():
    #     feature_length = len(v)
    #     break

    print('save as np array')
    arr = np.array((list(jaws.values())))
    print(arr.shape)
    # arr = np.zeros((len(jaws), feature_length))
    # for row, v in enumerate(jaws.values()):
    #     arr[row] = v
    np.save('jaws', arr)


def main():
    save_jaws('../data/data.csv', '../data/data_labels.txt', True, 1)


if __name__ == '__main__':
    main()
