import numpy as np
import cv2

def calculate_projection_matrix(image, markers):
    
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    parameters = cv2.aruco.DetectorParameters_create()

    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
        image, dictionary, parameters=parameters)
    markerIds = [m[0] for m in markerIds]
    markerCorners = [m[0] for m in markerCorners]

    points2d = []
    points3d = []

    for markerId, marker in zip(markerIds, markerCorners):
        if markerId in markers:
            for j, corner in enumerate(marker):
                points2d.append(corner)
                points3d.append(markers[markerId][j])

    points2d = np.array(points2d)
    points3d = np.array(points3d)

    n = points2d.shape[0]
    data_matrix = np.zeros((2 * n, 11))
    coefficient_vector = np.zeros((2 * n, 1))

    for i in range(0, n):
        X = points3d[i, 0]
        Y = points3d[i, 1]
        Z = points3d[i, 2]
        u = points2d[i, 0]
        v = points2d[i, 1]

        data_matrix[(2 *
                     i), :] = [X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z]
        data_matrix[(2 * i) +
                    1, :] = [0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z]
        coefficient_vector[(2 * i)] = u
        coefficient_vector[(2 * i) + 1] = v

    M, residual, rank, s = np.linalg.lstsq(data_matrix, coefficient_vector, rcond=None)
    M = np.append(M, 1)
    M = np.reshape(M, (3, 4))

    return M, residual
