import numpy as np

def labels_to_col_nr(labels_file):
    # why not pandas? fuck pandas.
    labels_dict = {}
    labels = []
    with open(labels_file, 'r') as file:
        for row_nr, row_data in enumerate(file):
            labels_dict[row_data[:-1]] = row_nr  # :-1 to remove \n
            labels.append(row_data[:-1])
    return labels_dict, labels


def facing_straight(angles, limit=1):
    try:
        angles = np.asarray(angles, dtype=np.float)
    except ValueError:
        return False
    if np.any(np.abs(angles) > limit):
        return False
    return True


def unroll(marks, roll):
    roll = np.deg2rad(roll)
    roll = np.asarray(
        [
            [np.cos(roll), np.sin(roll)],
            [-np.sin(roll), np.cos(roll)]
        ]
    )
    return [m@roll for m in marks]


def normalize(marks, face_width, face_height):
    return [((m[0] - marks[0][0])/face_width, (m[1] - marks[0][1])/face_height)
            for m in marks]

def normalize_by_eyes(marks, labeled_data):
    eye_left = np.array([m[0][0] for m in labeled_data if 'left_eye_' in m[1]])
    eye_right = np.array([m[0][0]
                          for m in labeled_data if 'right_eye_' in m[1]])
    dist = np.mean(np.abs(eye_left - eye_right))
    markarray = np.array(marks)
    meanX = np.mean(markarray[:, 0])
    meanY = np.mean([m[0][1] for m in labeled_data if '_eye_' in m[1]])
    return [((m[0] - meanX)/dist, (m[1] - meanY)/dist)
            for m in marks]