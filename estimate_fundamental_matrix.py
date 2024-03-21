import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

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


def normalize_coordinates(Points):
   
    n = Points.shape[0]
    u = np.copy(Points[:, 0])
    v = np.copy(Points[:, 1])

    # Calculate offset matrix
    c_u = np.mean(u)
    c_v = np.mean(v)

    offset_matrix = np.array([[1, 0, -c_u], [0, 1, -c_v], [0, 0, 1]])

    # Calculate scale matrix
    s = 1 / np.std([[u - c_u], [v - c_v]])

    scale_matrix = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]])

    # Calculate transformation matrix
    T = scale_matrix @ offset_matrix

    # Normalize points using transformation matrix
    for i in range(0, n):
        norm = T @ np.transpose([u[i], v[i], 1])
        u[i] = norm[0]
        v[i] = norm[1]

    return np.column_stack((u, v)), T


def estimate_fundamental_matrix_unnormalizedpoints(Points1, Points2):
  
    n = Points2.shape[0]

    u = np.copy(Points1[:, 0])
    v = np.copy(Points1[:, 1])
    u_prime = np.copy(Points2[:, 0])
    v_prime = np.copy(Points2[:, 1])

    # Create data matrix
    data_matrix = np.array([
        u_prime * u, u_prime * v, u_prime, v_prime * u, v_prime * v, v_prime,
        u, v,
        np.ones((n))
    ])
    data_matrix = np.transpose(data_matrix)

    # Get system matrix using svd
    U, S, Vh = np.linalg.svd(data_matrix)

    full_F = Vh[-1, :]
    full_F = np.reshape(full_F, (3, 3))
    U, S, Vh = np.linalg.svd(full_F)
    S[-1] = 0
    F_matrix = U @ np.diagflat(S) @ Vh

    residual = np.sum(np.square(data_matrix @ F_matrix.flatten()))
    dist = np.zeros((n))
    for j in range(0, n):
        homMatch1 = np.append(Points1[j, :], [1])
        homMatch2 = np.append(Points2[j, :], [1])
        dist[j] = np.abs(homMatch2 @ F_matrix @ np.transpose(homMatch1))
    residual = np.sum(np.square(dist))

    return F_matrix, residual

def estimate_fundamental_matrix(Points1, Points2):
    n = Points2.shape[0]

    Points1_norm, T1 = normalize_coordinates(Points1)
    Points2_norm, T2 = normalize_coordinates(Points2)

    u = np.copy(Points1_norm[:, 0])
    v = np.copy(Points1_norm[:, 1])
    u_prime = np.copy(Points2_norm[:, 0])
    v_prime = np.copy(Points2_norm[:, 1])

    # Create data matrix
    data_matrix = np.array([
        u_prime * u, u_prime * v, u_prime, v_prime * u, v_prime * v, v_prime,
        u, v,
        np.ones((n))
    ])
    data_matrix = np.transpose(data_matrix)
    U, S, Vh = np.linalg.svd(data_matrix)
    full_F = Vh[-1, :]
    full_F = np.reshape(full_F, (3, 3))
    U, S, Vh = np.linalg.svd(full_F)
    S[-1] = 0
    F_matrix_norm = U @ np.diagflat(S) @ Vh
    residual = np.sum(np.square(data_matrix @ F_matrix_norm.flatten()))
    F_matrix = np.transpose(T2) @ F_matrix_norm @ T1

    return F_matrix, residual

def ransac_fundamental_matrix(matches1, matches2, num_iters):
    
    num_pts = 8  # number of point correspondances per iteration
    threshold = 0.002  # threshold for distance metric of inliers
    n = matches1.shape[0]
    inlier_best_count = 0
    inlier_best = np.zeros((n))
    best_Fmatrix = np.zeros((3, 3))
    random.seed(0)  # seed the same as student template

    for i in range(0, num_iters):
        # Pick num_pts random pts from correspondances in [1, n]
        cur_rand = random.sample(range(n), num_pts)
        points1 = matches1[cur_rand, :]
        points2 = matches2[cur_rand, :]

        # Calculate the fundamental matrix using pt 2 work
        cur_F, residual = estimate_fundamental_matrix(points1, points2)

        # Use distance metric to find inliers. For a given correspondence x to
        # x', x'Fx = 0. So our metric refers to how far from zero our result
        # is. Store inliers
        dist = np.zeros((n))
        for j in range(0, n):
            homMatch1 = np.append(matches1[j, :], [1])
            homMatch2 = np.append(matches2[j, :], [1])
            dist[j] = np.abs(homMatch2 @ cur_F @ np.transpose(homMatch1))

        inliers = dist <= threshold
        inlier_count = np.sum(inliers)
        inlier_counts.append(inlier_count)

        inlier_residual = np.sum(np.square(dist[inliers]))
        inlier_residuals.append( inlier_residual )

        if (inlier_count > inlier_best_count):
            inlier_best_count = inlier_count
            inlier_best = inliers
            best_Fmatrix = cur_F
            best_inlier_residual = inlier_residual

    best_inliers1 = matches1[inlier_best, :]
    best_inliers2 = matches2[inlier_best, :]

    return best_Fmatrix, best_inliers1, best_inliers2, best_inlier_residual


def visualize_ransac():
    iterations = np.arange(len(inlier_counts))
    best_inlier_counts = np.maximum.accumulate(inlier_counts)
    best_inlier_residuals = np.minimum.accumulate(inlier_residuals)

    plt.figure(1, figsize = (8, 8))
    plt.subplot(211)
    plt.plot(iterations, inlier_counts, label='Current Inlier Count', color='red')
    plt.plot(iterations, best_inlier_counts, label='Best Inlier Count', color='blue')
    plt.xlabel("Iteration")
    plt.ylabel("Number of Inliers")
    plt.title('Current Inliers vs. Best Inliers per Iteration')
    plt.legend()

    plt.subplot(212)
    plt.plot(iterations, inlier_residuals, label='Current Inlier Residual', color='red')
    plt.plot(iterations, best_inlier_residuals, label='Best Inlier Residual', color='blue')
    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.title('Current Residual vs. Best Residual per Iteration')
    plt.legend()
    plt.show()