import numpy as np
import random
import cv2

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
