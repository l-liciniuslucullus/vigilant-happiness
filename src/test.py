from get_features import get_labelled_landmarks

with open('../data/testdata.csv') as file:
    for row in file:
        data = row.split(',')
        id_, lp, b = get_labelled_landmarks(
            data, '../data/testdata_labels.txt', face_id_clean=True)
        print(id_, lp, b)


# keys = [k for k in d.keys() if 'lip' in k]
# lips = [landmarks[k] for k in set(landmarks).intersection(keys)]