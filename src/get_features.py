import numpy as np
from utils import labels_to_col_nr, facing_straight, unroll, normalize_by_eyes


def get_labelled_landmarks(data_row, col_nr, labels, face_straight=True, limit_deg=1,
                           face_id_clean=False):
    if face_straight and not facing_straight(
            [data_row[col_nr['headpose.yaw_angle']],
             data_row[col_nr['headpose.pitch_angle']]
             ],
            limit_deg):
        return None, None, False
    face_id = data_row[col_nr['face_id']]
    if not face_id_clean:
        face_id = face_id[11:-14]
    coordinates = [int(o) for o in
                   data_row[col_nr['contour_chin.y']
                       :col_nr['right_eye_pupil.x']]
                   ]
    points = np.asarray([(p[1], -p[0]) for p in zip(coordinates[::2],
                                                    coordinates[1::2])])
    points_labels = [
        l[:-2] for l in labels[col_nr['contour_chin.y']:col_nr['right_eye_pupil.x']]]
    labelled_points = list(zip(points, points_labels))
    return face_id, labelled_points, True


def get_features(file, feature_func, labels_file, gender_file=None):
    features = {}
    col_nr, labels = labels_to_col_nr(labels_file)
    if gender_file:
        gender = get_gender(gender_file)
    with open(file, 'r') as data_file:
        for row in data_file:
            data = row.split(',')
            face_id, labelled_data, ok = get_labelled_landmarks(
                data, col_nr, labels)
            if not ok or (gender_file and face_id not in gender):
                continue
            labelled_data = unroll(labelled_data, np.float(data[col_nr['roll']]))
            labelled_data = normalize_by_eyes(labelled_data, labelled_data)
            features[face_id] = feature_func(labelled_data)
            if gender_file:
                features[face_id].append(gender[face_id])
    return features


def get_gender(file):
    genders = {}
    with open(file, 'r') as f:
        for row in f:
            data = row.split(',')
            face_id = data[0][1:-1]  # 1:-1 = remove ""
            gender = int(data[1])
            genders[face_id] = gender
    return genders
