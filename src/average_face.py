import numpy as np
from get_features import get_labelled_landmarks, get_gender
import matplotlib.pyplot as plt
from utils import unroll, normalize, normalize_by_eyes


def get_faces(file, labels=None, gender_file=None):
    features = {}
    if gender_file:
        gender = get_gender(gender_file)
    with open(file, 'r') as data_file:
        for row in data_file:
            data = row.split(',')
            face_id, labeled_data, ok = get_labelled_landmarks(
                data, labels, limit_deg=1)
            if not ok or (gender_file and face_id not in gender):
                continue
            marks = [m[0] for m in labeled_data]
            roll = np.float(data[238])
            marks = unroll(marks, roll)
            face_width = np.float(data[-5])
            face_height = np.float(data[-2])
            marks = normalize(marks, face_width, face_height)
            marks = [x for m in marks for x in m]
            features[face_id] = marks
            if gender_file:
                features[face_id].append(gender[face_id])
    return features


def split_genders(faces):
    g0 = [row[:-1] for row in faces.values() if row[-1] == 0]
    g1 = [row[:-1] for row in faces.values() if row[-1] == 1]
    return np.array(g0), np.array(g1)


def main():
    print('getting faces')
    g1, g2 = split_genders(
        get_faces('../data/temp.csv',
                  np.loadtxt("../data/labels", delimiter="\n",
                             dtype=np.str)[:: 2],
                  '../data/gender.csv')
    )
    # g1 = g1[:1000]
    # g2 = g2[:1000]
    print('averaging')
    g1 = [(x, y) for x, y in zip(np.mean(g1, 0)[::2], np.mean(g1, 0)[1::2])]
    g2 = [(x, y) for x, y in zip(np.mean(g2, 0)[::2], np.mean(g2, 0)[1::2])]
    g1 = np.array(g1)
    g2 = np.array(g2)
    np.save('g0_2deg', g1)
    np.save('g1_2deg', g2)
    # plt.plot(g1[:, 0], g1[:, 1], '.')
    # plt.savefig('gender0.png')
    # plt.clf()
    # plt.plot(g2[:, 0], g2[:, 1], '.')
    # plt.savefig('gender1.png')


if __name__ == '__main__':
    main()
