import numpy as np
from utils import labels_to_col_nr, facing_straight


def get_labelled_landmarks(data_row, labels_file, face_straight=True, limit_deg=1,
                           face_id_clean=False):
    col_nr, labels = labels_to_col_nr(labels_file)
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
                   data_row[col_nr['contour_chin.y']:col_nr['right_eye_pupil.x']]
                   ]
    points = np.asarray([(p[1], -p[0]) for p in zip(coordinates[::2],
                                                    coordinates[1::2])])
    points_labels = [
        l[:-2] for l in labels[col_nr['contour_chin.y']:col_nr['right_eye_pupil.x']]]
    labelled_points = list(zip(points, points_labels))
    return face_id, labelled_points, True


def get_features(file, feature_func, labels=None, gender_file=None):
    features = {}
    if gender_file:
        gender = get_gender(gender_file)
    with open(file, 'r') as data_file:
        for row in data_file:
            data = row.split(',')
            face_id, labeled_data, ok = get_labelled_landmarks(data, labels)
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
