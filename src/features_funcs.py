from time import time
import numpy as np
from scipy.spatial.distance import cdist, euclidean
from numpy.linalg import norm
from get_features import get_features
from utils import unroll, normalize, normalize_by_eyes
import ellipses as el
from scipy.optimize import curve_fit as cf


def fwhr(landmarks):
    '''something wrong with this apparently'''
    # sides = [m[0] for m in landmarks if m[1] ==
    #          'contour_left3' or m[1] == 'contour_right3']

    # poimts farthest to right and left
    sides = np.array([m[0] for m in landmarks])
    sides = sides[sides[:, 0].argsort()]
    face_width = np.abs(sides[0, 0] - sides[-1, 0])

    eyes_and_lips = [m for m in landmarks if 'eye_top' in m[1]
                     or 'upper_lip_top' in m[1]]
    _, y1 = [p[0] for p in eyes_and_lips if 'left' in p[1]][0]
    _, y2 = [p[0] for p in eyes_and_lips if 'right' in p[1]][0]
    _, y0 = [p[0] for p in eyes_and_lips if 'lip' in p[1]][0]

    y_top = np.mean((y1, y2))

    face_height = np.abs(y_top - y0)

    return [face_width / face_height]

# face_width = euclidean(sides[0], sides[1])

# '''https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line'''
# eyes_and_lips = [m for m in landmarks if 'eye_top' in m[1]
#                  or 'upper_lip_top' in m[1]]
# x1, y1 = [p[0] for p in eyes_and_lips if 'left' in p[1]][0]
# x2, y2 = [p[0] for p in eyes_and_lips if 'right' in p[1]][0]
# x0, y0 = [p[0] for p in eyes_and_lips if 'lip' in p[1]][0]
# num = abs(
#     (y2-y1)*x0-(x2-x1)*y0+x2*y1-y2*x1
# )
# denum = ((y2-y1)**2+(x2-x1)**2)**0.5
# face_height = num / denum

# p1, p2 = np.array(sides[0]), np.array(sides[1])
# p3 = np.array([m[0]
#                for m in landmarks if m[1] == 'mouth_upper_lip_top'][0])
# d = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)

# print(face_width/face_height, face_width/d)

# x1, y1 = p1
# x2, y2 = p2
# x0, y0 = p3
# num = abs(
#     (y2-y1)*x0-(x2-x1)*y0+x2*y1-y2*x1
# )
# denum = ((y2-y1)**2+(x2-x1)**2)**0.5
# face_height = num / denum
# face_width = euclidean(p1, p2)

# print(face_width/face_height, face_width/d)
# exit(1)
# # [face_width / face_height]

# return [d]


def face_contour(landmarks):
    return list(np.array([m[0] for m in landmarks if 'contour_' in m[1]]).flatten())


def ellipse(landmarks):
    face = np.array([m[0] for m in landmarks if 'contour_' in m[1]])
    data = [face[:, 0], face[:, 1]]
    lsqe = el.LSqEllipse()
    lsqe.fit(data)
    center, width, height, phi = lsqe.parameters()
    return [center[0], center[1], width, height, phi]


def ellipse_picked(landmarks):
    x, y, w, h, phi = ellipse(landmarks)
    return [y, w, h]


def eccentricity(landmarks):
    x, y, w, h, phi = ellipse(landmarks)
    # assume width is major axis
    if w < h:
        w, h = h, w
    return [np.sqrt(1 - (h / w)**2)]


def rectum(landmarks):
    x, y, w, h, phi = ellipse(landmarks)
    # assume width is major axis
    if w < h:
        w, h = h, w
    return [h**2/w]


def _x6(x, a, b, c, d, e, f, g):
    return a*x**6 + b*x**5 + c*x**4 + d*x**3 + e*x**2 + f*x + g


def _butterfly(x, a, b, c, d, z):
    return x**6 + a*x**4 + b*x**3 + c*x**2 + d*x + z


def butterfly_catastrophe(landmarks):
    a = np.array([m[0] for m in landmarks if 'contour_' in m[1]])
    a = a[a[:, 0].argsort()]
    x = a[:, 0]
    y = a[:, 1]
    cutoff = 7
    popt, pcov = cf(_butterfly, x[cutoff:-cutoff], y[cutoff:-cutoff])
    return list(popt)


def polyfit6(landmarks):
    a = np.array([m[0] for m in landmarks if 'contour_' in m[1]])
    a = a[a[:, 0].argsort()]
    x = a[:, 0]
    y = a[:, 1]
    cutoff = 7
    popt, pcov = cf(_x6, x[cutoff:-cutoff], y[cutoff:-cutoff])
    return list(popt)


def face_triangle_area(landmarks):
    eyes_and_lips = [m for m in landmarks if 'eye_top' in m[1]
                     or 'upper_lip_top' in m[1]]
    '''https://en.wikipedia.org/wiki/Triangle#Using_coordinates'''
    x1, y1 = [p[0] for p in eyes_and_lips if 'left' in p[1]][0]
    x2, y2 = [p[0] for p in eyes_and_lips if 'right' in p[1]][0]
    x0, y0 = [p[0] for p in eyes_and_lips if 'lip' in p[1]][0]

    x1 -= x0
    x2 -= x0
    y1 -= y0
    y2 -= y0

    return [0.5*abs(x1*y2-x2*y1)]


# def delauney(points):
#     pass


def pure_landmarks(landmarks):
    return [np.array([c for m in landmarks for c in m[0]]).flatten()]


def hand_picked_points(landmarks):
    eyes = [m[0]
            for m in landmarks if 'left_eye_' in m[1] or 'right_eye_' in m[1]]
    lower_lip = [m[0] for m in landmarks if 'mouth_lower_lip' in m[1]]
    nose = [m[0] for m in landmarks if m[1] == 'nose_bridge1'
            or m[1] == 'nose_bridge2' or m[1] == 'nose_left_contour1'
            or m[1] == 'nose_right_contour1']
    contours = [m[0] for m in landmarks if m[1] == 'contour_left1'
                or m[1] == 'contour_left2' or m[1] == 'contour_left3'
                or m[1] == 'contour_left4' or m[1] == 'contour_right1'
                or m[1] == 'contour_right2' or m[1] == 'contour_right3'
                or m[1] == 'contour_right4']
    eyebrows = [m[0] for m in landmarks if m[1] == 'left_eyebrow_upper_right_corner'
                or m[1] == 'left_eyebrow_lower_right_corner'
                or m[1] == 'right_eyebrow_upper_left_corner'
                or m[1] == 'right_eyebrow_lower_left_corner']
    marks = eyes + lower_lip + nose + contours + eyebrows
    return marks


def hand_picked(landmarks):
    marks = hand_picked_points(landmarks)
    return [x for m in marks for x in m]


def hand_picked_dists(landmarks):
    marks = hand_picked_points(landmarks)
    p0 = np.array([marks[0]])
    dists = list(cdist(p0, marks[1:]).reshape(43,))
    return dists


'''one reference point distances'''


def distances(landmarks):
    landmarks = [m[0] for m in landmarks]
    points = landmarks[1:]
    p0 = np.array([landmarks[0]])
    return list(cdist(p0, points).reshape(105,))


def distances_normalized(landmarks, roll, width, height):
    marks = [m[0] for m in landmarks]
    marks = unroll(marks, roll)
    marks = normalize(marks, width, height)
    points = marks[1:]
    p0 = np.array([marks[0]])
    return list(cdist(p0, points).reshape(105,))


def distances_normalized_by_eyes(landmarks, roll):
    marks = [m[0] for m in landmarks]
    marks = unroll(marks, roll)
    marks = normalize_by_eyes(marks, landmarks)
    points = marks[1:]
    p0 = np.array([marks[0]])
    return list(cdist(p0, points).reshape(105,))


def dists_lips(landmarks):
    ref1 = 'contour_chin'
    ref1v = [m[0] for m in landmarks if m[1] == ref1]
    lips = [m[0] for m in landmarks if 'lip' in m[1]
            and m[1] != ref1]
    return list(cdist(
        np.array(ref1v), lips
    ).flatten())


def dists_lower_lip(landmarks):
    ref1 = 'contour_chin'
    ref1v = [m[0] for m in landmarks if m[1] == ref1]
    lips = [m[0] for m in landmarks if 'mouth_lower_lip_' in m[1]
            and m[1] != ref1]
    return list(cdist(
        np.array(ref1v), lips
    ).flatten())


def dists_nose(landmarks):
    ref1 = 'contour_chin'
    ref1v = [m[0] for m in landmarks if m[1] == ref1]
    nose = [m[0] for m in landmarks if 'nose_' in m[1]
            and m[1] != ref1]
    return list(cdist(
        np.array(ref1v), nose
    ).flatten())


def dists_nose_top4(landmarks):
    ref1 = 'contour_chin'
    ref1v = [m[0] for m in landmarks if m[1] == ref1]
    nose = [m[0] for m in landmarks if m[1] == 'nose_bridge1'
            or m[1] == 'nose_bridge2' or m[1] == 'nose_left_contour1'
            or m[1] == 'nose_right_contour1'
            and m[1] != ref1]
    return list(cdist(
        np.array(ref1v), nose
    ).flatten())


def dists_eyes(landmarks):
    ref1 = 'contour_chin'
    ref1v = [m[0] for m in landmarks if m[1] == ref1]
    eyes = [m[0]
            for m in landmarks if 'left_eye_' in m[1] or 'right_eye_' in m[1]
            and m[1] != ref1]
    return list(cdist(
        np.array(ref1v), eyes
    ).flatten())


def dists_eyebrows(landmarks):
    ref1 = 'contour_chin'
    ref1v = [m[0] for m in landmarks if m[1] == ref1]
    eyebrows = [m[0] for m in landmarks if 'eyebrow' in m[1]
                and m[1] != ref1]
    return list(cdist(
        np.array(ref1v), eyebrows
    ).flatten())


def dists_eyebrows_corners(landmarks):
    ref1 = 'contour_chin'
    ref1v = [m[0] for m in landmarks if m[1] == ref1]
    eyebrows = [m[0] for m in landmarks if m[1] == 'left_eyebrow_upper_right_corner'
                or m[1] == 'left_eyebrow_lower_right_corner'
                or m[1] == 'right_eyebrow_upper_left_corner'
                or m[1] == 'right_eyebrow_lower_left_corner'
                and m[1] != ref1]
    return list(cdist(
        np.array(ref1v), eyebrows
    ).flatten())


def dists_contour(landmarks):
    ref1 = 'contour_chin'
    ref1v = [m[0] for m in landmarks if m[1] == ref1]
    contour = [m[0]
               for m in landmarks if 'contour_' in m[1]
               and m[1] != ref1]
    return list(cdist(
        np.array(ref1v), contour
    ).flatten())


def dists_contour_top4(landmarks):
    ref1 = 'contour_chin'
    ref1v = [m[0] for m in landmarks if m[1] == ref1]
    contour = [m[0]
               for m in landmarks if m[1] == 'contour_left1'
               or m[1] == 'contour_left2'
               or m[1] == 'contour_left3'
               or m[1] == 'contour_left4'
               or m[1] == 'contour_right1'
               or m[1] == 'contour_right2'
               or m[1] == 'contour_right3'
               or m[1] == 'contour_right4'
               and m[1] != ref1]
    return list(cdist(
        np.array(ref1v), contour
    ).flatten())


'''two reference points distances'''


def get_refs(landmarks):
    ref1 = 'right_eyebrow_right_corner'
    ref2 = 'contour_chin'
    ref1v = [m[0] for m in landmarks if m[1] == ref1]
    ref2v = [m[0] for m in landmarks if m[1] == ref2]
    return ref1, ref2, np.vstack((ref1v, ref2v))


def dists_2_refs_all(landmarks):
    ref1, ref2, refs = get_refs(landmarks)
    marks = [m[0] for m in landmarks if m[1] != ref1 and m[1] != ref2]
    return list(cdist(refs, marks).flatten())


def dists_2_refs_lips(landmarks):
    ref1, ref2, refs = get_refs(landmarks)
    lips = [m[0] for m in landmarks if 'lip' in m[1]
            and m[1] != ref1 and m[1] != ref2]
    return list(cdist(
        refs, lips
    ).flatten())


def dists_2_refs_lower_lip(landmarks):
    ref1, ref2, refs = get_refs(landmarks)
    lips = [m[0] for m in landmarks if 'mouth_lower_lip_' in m[1]
            and m[1] != ref1 and m[1] != ref2]
    return list(cdist(
        refs, lips
    ).flatten())


def dists_2_refs_nose(landmarks):
    ref1, ref2, refs = get_refs(landmarks)
    nose = [m[0] for m in landmarks if 'nose_' in m[1]
            and m[1] != ref1 and m[1] != ref2]
    return list(cdist(
        refs, nose
    ).flatten())


def dists_2_refs_nose_top4(landmarks):
    ref1, ref2, refs = get_refs(landmarks)
    nose = [m[0] for m in landmarks if m[1] == 'nose_bridge1'
            or m[1] == 'nose_bridge2' or m[1] == 'nose_left_contour1'
            or m[1] == 'nose_right_contour1'
            and m[1] != ref1 and m[1] != ref2]
    return list(cdist(
        refs, nose
    ).flatten())


def dists_2_refs_eyes(landmarks):
    ref1, ref2, refs = get_refs(landmarks)
    eyes = [m[0]
            for m in landmarks if 'left_eye_' in m[1] or 'right_eye_' in m[1]
            and m[1] != ref1 and m[1] != ref2]
    return list(cdist(
        refs, eyes
    ).flatten())


def dists_2_refs_eyebrows(landmarks):
    ref1, ref2, refs = get_refs(landmarks)
    eyebrows = [m[0] for m in landmarks if 'eyebrow' in m[1]
                and m[1] != ref1 and m[1] != ref2]
    return list(cdist(
        refs, eyebrows
    ).flatten())


def dists_2_refs_eyebrows_corners(landmarks):
    ref1, ref2, refs = get_refs(landmarks)
    eyebrows = [m[0] for m in landmarks if m[1] == 'left_eyebrow_upper_right_corner'
                or m[1] == 'left_eyebrow_lower_right_corner'
                or m[1] == 'right_eyebrow_upper_left_corner'
                or m[1] == 'right_eyebrow_lower_left_corner'
                and m[1] != ref1 and m[1] != ref2]
    return list(cdist(
        refs, eyebrows
    ).flatten())


def dists_2_refs_contour(landmarks):
    ref1, ref2, refs = get_refs(landmarks)
    contour = [m[0]
               for m in landmarks if 'contour_' in m[1]
               and m[1] != ref1 and m[1] != ref2]
    return list(cdist(
        refs, contour
    ).flatten())


def dists_2_refs_contour_top4(landmarks):
    ref1, ref2, refs = get_refs(landmarks)
    contour = [m[0]
               for m in landmarks if m[1] == 'contour_left1'
               or m[1] == 'contour_left2'
               or m[1] == 'contour_left3'
               or m[1] == 'contour_left4'
               or m[1] == 'contour_right1'
               or m[1] == 'contour_right2'
               or m[1] == 'contour_right3'
               or m[1] == 'contour_right4'
               and m[1] != ref1 and m[1] != ref2]
    return list(cdist(
        refs, contour
    ).flatten())


def face_triangle_circumcircle_area(landmarks):
    eyes_and_lips = [m for m in landmarks if 'eye_top' in m[1]
                     or 'upper_lip_top' in m[1]]
    '''https://en.wikipedia.org/wiki/Circumscribed_circle#Cartesian_coordinates'''
    A = [p[0] for p in eyes_and_lips if 'left' in p[1]][0]
    B = [p[0] for p in eyes_and_lips if 'right' in p[1]][0]
    C = [p[0] for p in eyes_and_lips if 'lip' in p[1]][0]
    A2 = A[0]**2 + A[1]**2
    B2 = B[0]**2 + B[1]**2
    C2 = C[0]**2 + C[1]**2
    Sx = np.array(
        [
            [A2, A[1], 1],
            [B2, B[1], 1],
            [C2, C[1], 1]
        ]
    )
    Sx = 0.5*np.linalg.det(Sx)
    Sy = np.array(
        [
            [A[0], A2, 1],
            [B[0], B2, 1],
            [C[0], C2, 1]
        ]
    )
    Sy = 0.5*np.linalg.det(Sy)
    a = np.array(
        [
            [A[0], A[1], 1],
            [B[0], B[1], 1],
            [C[0], C[1], 1]
        ]
    )
    a = np.linalg.det(a)
    b = np.array(
        [
            [A[0], A[1], A2],
            [B[0], B[1], B2],
            [C[0], C[1], C2]
        ]
    )
    b = np.linalg.det(b)

    return [(b/a + (Sx**2 + Sy**2)/a**2)*np.pi]


def face_triangle_to_circumcircle_ratio(landmarks):
    return [face_triangle_area(landmarks)[0] / face_triangle_circumcircle_area(landmarks)[0]]


def lips_whr(landmarks):
    height = [x[0] for x in landmarks if x[1] == 'mouth_upper_lip_top'
              or x[1] == 'mouth_lower_lip_bottom']
    height = euclidean(height[0], height[1])
    width = [x[0] for x in landmarks if x[1] == 'mouth_right_corner'
             or x[1] == 'mouth_left_corner']
    width = euclidean(width[0], width[1])
    return [width/height]


def lips_dists(landmarks):
    lips = [x[0] for x in landmarks if 'lip' in x[1]]
    lips = cdist([lips[0]], lips[1:])
    lips.reshape(17,)
    return list(lips[0])


def brow_dists(landmarks):
    ret = [x[0] for x in landmarks if x[1] == 'right_eye_top']
    let = [x[0] for x in landmarks if x[1] == 'left_eye_top']
    left_brow = [x[0] for x in landmarks if 'left_eyebrow' in x[1]]
    right_brow = [x[0] for x in landmarks if 'right_eyebrow' in x[1]]
    return list(cdist(ret, right_brow).reshape(9,)) + \
        list(cdist(let, left_brow).reshape(9,))


def main():
    pass


if __name__ == '__main__':
    main()
