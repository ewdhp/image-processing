import cv2
import numpy as np

def calculate_disparity_map(left_image, right_image, num_disparities=16, block_size=15):
    stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
    disparity = stereo.compute(left_image, right_image)
    return disparity
def calculate_depth_map(disparity_map, focal_length, baseline):
    depth_map = np.zeros(disparity_map.shape, dtype=np.float32)
    for y in range(disparity_map.shape[0]):
        for x in range(disparity_map.shape[1]):
            if disparity_map[y, x] > 0:
                depth_map[y, x] = (focal_length * baseline) / disparity_map[y, x]
    return depth_map
def reconstruct_3D_points(disparity_map, focal_length, baseline):
    height, width = disparity_map.shape
    Q = np.float32([[1, 0, 0, -width / 2],
                    [0, -1, 0, height / 2],
                    [0, 0, 0, -focal_length],
                    [0, 0, 1 / baseline, 0]])
    points_3D = cv2.reprojectImageTo3D(disparity_map, Q)
    return points_3D
def process_stereo_pair(left_image_path, right_image_path, focal_length, baseline):
    left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

    disparity_map = calculate_disparity_map(left_image, right_image)
    depth_map = calculate_depth_map(disparity_map, focal_length, baseline)
    points_3D = reconstruct_3D_points(disparity_map, focal_length, baseline)

    return points_3D
def combine_3D_points(*points_3D_list):
    combined_points = np.vstack(points_3D_list)
    return combined_points

# Example usage
focal_length = 700  # Example focal length
baseline = 0.1  # Example baseline distance

# Process each stereo pair
rear_left_right_points = process_stereo_pair('rear_left.jpg', 'rear_right.jpg', focal_length, baseline)
front_left_right_points = process_stereo_pair('front_left.jpg', 'front_right.jpg', focal_length, baseline)

# Combine 3D points from all pairs
combined_points = combine_3D_points(rear_left_right_points, front_left_right_points)

# Optionally, visualize or save the combined 3D points
# For example, you can save the points to a PLY file
def export_to_ply(points, filename='output.ply'):
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex %d\n' % len(points))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')
        for point in points:
            f.write('%f %f %f\n' % (point[0], point[1], point[2]))

export_to_ply(combined_points)