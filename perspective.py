import cv2
import numpy as np

def perspective_transform(image, src_points, dst_points):
    """
    Apply perspective transformation to the input image.

    Parameters:
    image (numpy.ndarray): Input image.
    src_points (numpy.ndarray): Source points for the transformation.
    dst_points (numpy.ndarray): Destination points for the transformation.

    Returns:
    numpy.ndarray: Image after perspective transformation.
    """
    # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply the perspective transformation
    transformed_image = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
    
    return transformed_image

# Example usage
image_path = 'input_image.jpg'
image = cv2.imread(image_path)

# Define source points (corners of the original image)
src_points = np.float32([[0, 0], [image.shape[1] - 1, 0], [image.shape[1] - 1, image.shape[0] - 1], [0, image.shape[0] - 1]])

# Define destination points (desired corners in the transformed image)
dst_points = np.float32([[100, 100], [image.shape[1] - 200, 50], [image.shape[1] - 100, image.shape[0] - 100], [200, image.shape[0] - 50]])

# Apply perspective transformation
transformed_image = perspective_transform(image, src_points, dst_points)

# Display the original and transformed images
cv2.imshow('Original Image', image)
cv2.imshow('Transformed Image', transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()