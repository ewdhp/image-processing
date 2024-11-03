import cv2
import numpy as np

def rectify_stereo_images(left_image, right_image, camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, R, T):
    """
    Rectify stereo images to align them on the same horizontal plane.

    Parameters:
    left_image (numpy.ndarray): Left input image.
    right_image (numpy.ndarray): Right input image.
    camera_matrix_left (numpy.ndarray): Camera matrix for the left camera.
    dist_coeffs_left (numpy.ndarray): Distortion coefficients for the left camera.
    camera_matrix_right (numpy.ndarray): Camera matrix for the right camera.
    dist_coeffs_right (numpy.ndarray): Distortion coefficients for the right camera.
    R (numpy.ndarray): Rotation matrix between the coordinate systems of the first and the second camera.
    T (numpy.ndarray): Translation vector between the coordinate systems of the first and the second camera.

    Returns:
    tuple: Rectified left and right images.
    """
    # Image size
    image_size = (left_image.shape[1], left_image.shape[0])

    # Compute the rectification transforms for both cameras
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, image_size, R, T)

    # Compute the rectification maps
    map1_left, map2_left = cv2.initUndistortRectifyMap(camera_matrix_left, dist_coeffs_left, R1, P1, image_size, cv2.CV_16SC2)
    map1_right, map2_right = cv2.initUndistortRectifyMap(camera_matrix_right, dist_coeffs_right, R2, P2, image_size, cv2.CV_16SC2)

    # Apply the rectification maps to the images
    rectified_left = cv2.remap(left_image, map1_left, map2_left, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(right_image, map1_right, map2_right, cv2.INTER_LINEAR)

    return rectified_left, rectified_right

# Example usage
left_image_path = 'left_image.jpg'
right_image_path = 'right_image.jpg'

# Load the images
left_image = cv2.imread(left_image_path)
right_image = cv2.imread(right_image_path)

# Camera parameters (example values, you should use your actual calibration data)
camera_matrix_left = np.array([[700, 0, 320], [0, 700, 240], [0, 0, 1]])
dist_coeffs_left = np.zeros(5)
camera_matrix_right = np.array([[700, 0, 320], [0, 700, 240], [0, 0, 1]])
dist_coeffs_right = np.zeros(5)
R = np.eye(3)  # Example rotation matrix
T = np.array([0.1, 0, 0])  # Example translation vector

# Rectify the images
rectified_left, rectified_right = rectify_stereo_images(left_image, right_image, camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, R, T)

# Display the rectified images
cv2.imshow('Rectified Left Image', rectified_left)
cv2.imshow('Rectified Right Image', rectified_right)
cv2.waitKey(0)
cv2.destroyAllWindows()