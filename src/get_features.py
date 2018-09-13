import numpy as np


def labels_to_col_nr(labels_file):
    labels = {}
    with open(labels_file, 'r') as file:
        for row_nr, row_data in enumerate(file):
            labels[row_data[:-1]] = row_nr  # :-1 to remove \n
    return labels


def facing_straight(angles, limit=1):
    try:
        angles = np.asarray(angles, dtype=np.float)
    except ValueError:
        return False
    if np.any(np.abs(angles) > limit):
        return False
    return True


def get_labeled_landmarks(data_row=None, labels=None, face_straight=True, limit=1):
    x = data_row
    if x is None:
        x = np.loadtxt("../data/test_row", delimiter=",", dtype=np.object)
    if face_straight and not facing_straight(x[236:238], limit):
        return None, None, False
    face_id = x[1][11:-14]
    y = [int(o) for o in x[4:-61]]
    z = np.asarray([(p[1], -p[0]) for p in zip(y[::2], y[1::2])])
    if labels is None:
        labels = np.loadtxt("../data/all_labels",
                            delimiter="\n", dtype=np.str)[4:-61:2]
    labels = [l[:-2] for l in labels]
    labeled_data = list(zip(z, labels))
    return face_id, labeled_data, True


def get_features(file, feature_func, labels=None, gender_file=None):
    features = {}
    if gender_file:
        gender = get_gender(gender_file)
    with open(file, 'r') as data_file:
        for row in data_file:
            data = row.split(',')
            face_id, labeled_data, ok = get_labeled_landmarks(data, labels)
            if not ok or (gender_file and face_id not in gender):
                continue
            roll = np.float(data[238])
            # face_width = np.float(data[-5])
            # face_height = np.float(data[-2])
            # thats a hack it only works for 'normalaize' in feature.funcs.distances.normalized
            # features[face_id] = feature_func(labeled_data, roll, face_width, face_height)
            # and thath works only for normalized_by_eyes
            features[face_id] = feature_func(labeled_data, roll)

            # regular way:
            # features[face_id] = feature_func(labeled_data)

            if gender_file:
                features[face_id].append(gender[face_id])
    return features


def get_gender(file):
    genders = {}
    with open(file, 'r') as f:
        for row in f:
            data = row.split(',')
            face_id = data[0][1:-1]
            gender = int(data[1])
            genders[face_id] = gender
    return genders
