import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

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
