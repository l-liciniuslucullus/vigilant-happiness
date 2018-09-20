# from get_features import get_labelled_landmarks

# with open('../data/testdata.csv') as file:
#     for row in file:
#         data = row.split(',')
#         id_, lp, b = get_labelled_landmarks(
#             data, '../data/testdata_labels.txt', face_id_clean=True)
#         print(id_, lp, b)


# # potentialy if changed labelled_data to a dict:
# # keys = [k for k in d.keys() if 'lip' in k]
# # lips = [landmarks[k] for k in set(landmarks).intersection(keys)]

import features_funcs as ff
import utils
import numpy as np
labels = utils.labels_to_col_nr('../data/testdata_labels.txt')
data = np.loadtxt('../data/testdata.csv', delimiter=',', dtype=np.object)

import get_features as gf

row = gf.get_labelled_landmarks(
    data[0], labels[0], labels[1], False, 180, True)
print(data[0, labels[0]['fwhr']])
ff.fwhr(row[1])
