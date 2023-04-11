import numpy as np
import cv2

# Camera Parameters
# (K1/K2)*([R1 t1]/[R2 t2])average
divKRt_list = [[0.846691644105407, -0.040977459037432, 55.823319537537984],
               [0.020722060285037, 0.865278915125770, 27.655421360239277],
               [-0.000081064501160, -0.000045575501777, 1.030892142492117]]
div_KRt = np.array(divKRt_list)

# RGB Camera
dist_1 = (-0.080868630346796, 0.033251498828361, 0, 0)  # distortion coefficients
# Intrinsic Matrix
cameraMatrix_1 = np.array([[2.758911003546605e+02, 0, 3.153842663093544e+02],
                           [0, 2.768906021018125e+02, 1.688635944907660e+02],
                           [0, 0, 1]])

# Event Camera
dist_2 = (-0.209252096927732, 0.049369374356825, 0, 0)  # distortion coefficients
# Intrinsic Matrix
cameraMatrix_2 = np.array([[3.175437450932042e+02, 0, 3.129403375278937e+02],
                           [0, 3.182048698146490e+02, 1.543686033613551e+02],
                           [0, 0, 1]])

# used to translate points from 640x338 to 4096x2160
Rx = 4096 / 640
Ry = 2160 / 338


# function to format event camera coordinates according to calibration
# slice_arr = N x 4 array with each row in format: [epooch, x, y, polarity]
def format_points(slice_arr):
    # remove epoch and polarity from array
    coordinates = np.delete(slice_arr, [0, 3], 1)
    # Crop coordinates to match aspect ratio used in calibration
    coordinates = np.delete(coordinates, np.where(coordinates[:, 1] < 80)[0], axis=0)
    coordinates = np.delete(coordinates, np.where(coordinates[:, 1] >= 418)[0], axis=0)
    # Update remaining coordinates to match this
    for c in coordinates:
        c[1] -= 80
    # Undistort points according to intrinsic parameters
    unique_coords = np.unique(coordinates, axis=0).astype(np.float64)
    undistorted_coords = cv2.undistortPoints(unique_coords, cameraMatrix_2, dist_2, P=cameraMatrix_2)
    undistorted_coords = np.resize(undistorted_coords, unique_coords.shape)
    rounded_coords = np.round_(undistorted_coords).astype(int)
    return rounded_coords


# remove any out of frame points according to 640x338 resolution
# points = list of coordinates in format[[x1, y1, x2, y2], [x1, y1, x2, y2],...]
def remove_oof_points(points):
    new_points = []
    for p in points:
        if p[0] < 0 and p[2] >= 0:
            p[0] = 0
        if p[1] < 0 and p[3] >= 0:
            p[1] = 0
        if p[2] >= 640 and p[0] < 640:
            p[2] = 639
        if p[3] >= 338 and p[1] < 338:
            p[3] = 337
        if p[0] >= 0 and p[1] >= 0 and p[2] < 680 and p[3] < 338:
            new_points.append(p)
    return new_points


# translate points from event camera image coordinates to RGB camera image coordinates
# points = list of 4 bounding box coordinates in event camera image plane
# format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
# returns bounding box coordinates in RGB camera image plane
# format: [x1, y1, x2, y2]
def translate(points):
    xmin = 640
    ymin = 338
    xmax = 0
    ymax = 0
    for p in points:
        xy1 = np.array([[p[0]], [p[1]], [1]])  # add z dimension to coordinates
        calc_points = np.matmul(div_KRt, xy1)  # multiply by camera intrinsic & extrinsic matrices
        # normalise result by dividing by z coordinate
        calc_point_norm = (round(calc_points[0][0] / calc_points[2][0]), round(calc_points[1][0] / calc_points[2][0]))
        # find and update minimum and maximum x, y box coordinates
        if calc_point_norm[0] < xmin:
            xmin = calc_point_norm[0]
        elif calc_point_norm[0] > xmax:
            xmax = calc_point_norm[0]
        if calc_point_norm[1] < ymin:
            ymin = calc_point_norm[1]
        elif calc_point_norm[1] > ymax:
            ymax = calc_point_norm[1]
    return [xmin, ymin, xmax, ymax]


# function to distort points using intrinsic matrix & distortion coefficients
# points = 4x2 array of undistorted points [x1,y1; x2,y2; x3,x3; x4;y4]
# returns 4x2 array of distorted points
def distort_points(undistorted_points, K, dist_coeffs):
    # Add ones to undistorted points
    undistorted_points = np.hstack((undistorted_points, np.ones((len(undistorted_points), 1))))

    # Normalize undistorted points
    normalized_points = np.linalg.inv(K) @ undistorted_points.T
    r_squared = np.sum(normalized_points[:2, :] ** 2, axis=0)

    # Compute distorted points
    distorted_points = normalized_points[:2, :] * (
            1 + dist_coeffs[0] * r_squared + dist_coeffs[1] * r_squared ** 2) \
                       + np.vstack((
        2 * dist_coeffs[2] * normalized_points[0, :] * normalized_points[1, :] + dist_coeffs[
            3] * (r_squared + 2 * normalized_points[0, :] ** 2),
        dist_coeffs[2] * (r_squared + 2 * normalized_points[1, :] ** 2) + 2 * dist_coeffs[
            3] * normalized_points[0, :] * normalized_points[1, :]))

    # Unnormalize distorted points
    distorted_points = K @ np.vstack((distorted_points, np.ones((1, len(undistorted_points)))))

    # Extract x and y coordinates of distorted points
    x = distorted_points[0, :]
    y = distorted_points[1, :]
    return np.array([[x[0], y[0]], [x[1], y[1]], [x[2], y[2]], [x[3], y[3]]])


# convert points from 640x338 undistorted image to original 4096x2160 image with lens distortion
# p = list of box coordinates in format [x1, y1, x2, y2]
def convert_back(p):
    points = np.array([[p[0], p[1]], [p[0], p[3]], [p[2], p[1]], [p[2], p[3]]])  # convert to array
    newp = distort_points(points, cameraMatrix_1, dist_1)  # distort points
    # calculate minimum & maximum x,y box coordinates
    points = [min(newp[:, 0]), min(newp[:, 1]), max(newp[:, 0]), max(newp[:, 1])]
    rounded = [round(p) for p in points]
    # multiply by resolution ratio to convert to original resolution
    return [rounded[0] * Rx, rounded[1] * Ry, rounded[2] * Rx, rounded[3] * Ry]
