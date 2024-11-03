import json
import sys
import cv2
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import skfuzzy as fuzz
from skimage.feature.texture import graycomatrix, graycoprops, local_binary_pattern


"""Image I/O"""

def read_image(image_path):
  """
  Read an image from the specified file path.

  Parameters:
  image_path (str): Path to the image file.

  Returns:
  numpy.ndarray: Image read from the file.
  """
  return cv2.imread(image_path)
def write_image(image, image_path):
  """
  Write the input image to the specified file path.

  Parameters:
  image (numpy.ndarray): Image to write.
  image_path (str): Path to write the image file.

  Returns:
  bool: True if the image is written successfully, False otherwise.
  """
  return cv2.imwrite(image_path, image)


"""Image vectorization"""

def vectorize_image(image):
  """
  Vectorize the input image.

  Parameters:
  image (numpy.ndarray): Input image.

  Returns:
  numpy.ndarray: Vectorized image.
  """
  return image.flatten()
def devectorize_image(vector, shape):
  """
  Devectorize the input vector.
  """
  return vector.reshape(shape)


"""Color Spaces"""

def convert_color(image, color_space='RGB'):
  """
  Convert the input image to a different color space.

  Parameters:
  image (numpy.ndarray): Input image.
  color_space (str): Color space to convert to.

  Returns:
  numpy.ndarray: Image in the new color space.
  """
  if color_space == 'RGB':
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  elif color_space == 'HSV':
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  elif color_space == 'HLS':
    return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
  elif color_space == 'LAB':
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
  elif color_space == 'YCrCb':
    return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
  elif color_space == 'XYZ':
    return cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
  elif color_space == 'Grayscale':
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  else:
    return image
def convert_to_rgb(image, color_space='BGR'):
  """
  Convert the input image to RGB color space.

  Parameters:
  image (numpy.ndarray): Input image.
  color_space (str): Color space of the input image.

  Returns:
  numpy.ndarray: Image in RGB color space.
  """
  if color_space == 'BGR':
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  elif color_space == 'HSV':
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
  elif color_space == 'LAB':
    return cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
  elif color_space == 'YCrCb':
    return cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB)
  elif color_space == 'XYZ':
    return cv2.cvtColor(image, cv2.COLOR_XYZ2RGB)
  elif color_space == 'Grayscale':
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
  else:
    return image
def convert_to_grayscale(image):
  """
  Convert the input image to grayscale.

  Parameters:
  image (numpy.ndarray): Input image.

  Returns:
  numpy.ndarray: Grayscale image.
  """
  return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
def convert_to_hsv(image):
  """
  Convert the input image to HSV color space.

  Parameters:
  image (numpy.ndarray): Input image.

  Returns:
  numpy.ndarray: Image in HSV color space.
  """
  return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
def convert_to_lab(image):
  """
  Convert the input image to CIE L*a*b* color space.

  Parameters:
  image (numpy.ndarray): Input image.

  Returns:
  numpy.ndarray: Image in CIE L*a*b* color space.
  """
  return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
def convert_to_hsv(image):
  """
  Convert the input image to HSV color space.

  Parameters:
  image (numpy.ndarray): Input image.

  Returns:
  numpy.ndarray: Image in HSV color space.
  """
  return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
def convert_to_hls(image):
  """
  Convert the input image to HLS color space.

  Parameters:
  image (numpy.ndarray): Input image.

  Returns:
  numpy.ndarray: Image in HLS color space.
  """
  return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
def convert_to_lab(image):
  """
  Convert the input image to CIE L*a*b* color space.

  Parameters:
  image (numpy.ndarray): Input image.

  Returns:
  numpy.ndarray: Image in CIE L*a*b* color space.
  """
  return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
def convert_to_ycrcb(image):
  """
  Convert the input image to YCrCb color space.
    """
  return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
def convert_to_xyz(image):
  """
  Convert the input image to XYZ color space.
  """ 
  return cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)


"""Thresholding"""

def global_thresholding(image, threshold=127):
  """
  Apply global thresholding to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  threshold (int): Threshold value.

  Returns:
  numpy.ndarray: Thresholded image.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
  return binary_image
def adaptive_thresholding(image, max_value=255, 
adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
threshold_type=cv2.THRESH_BINARY, block_size=11, C=2):
  """
  Apply adaptive thresholding to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  max_value (int): Maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types.
  adaptive_method (int): Adaptive thresholding algorithm to use.
  threshold_type (int): Type of thresholding to apply.
  block_size (int): Size of a pixel neighborhood that is used to calculate a threshold value for the pixel.
  C (int): Constant subtracted from the mean or weighted mean.

  Returns:
  numpy.ndarray: Thresholded image.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  binary_image = cv2.adaptiveThreshold(gray_image, max_value, adaptive_method, threshold_type, block_size, C)
  return binary_image
def otsu_thresholding(image):
  """
  Apply Otsu's thresholding to the input image.

  Parameters:
  image (numpy.ndarray): Input image.

  Returns:
  numpy.ndarray: Thresholded image.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  return binary_image



"""Morphological Operations"""

def apply_erosion(image, kernel_size=5):
  """
  Apply erosion to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  kernel_size (int): Kernel size for the erosion operation.

  Returns:
  numpy.ndarray: Image after erosion.
  """
  kernel = np.ones((kernel_size, kernel_size), np.uint8)
  eroded_image = cv2.erode(image, kernel, iterations=1)
  return eroded_image
def apply_dilation(image, kernel_size=5):
  """
  Apply dilation to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  kernel_size (int): Kernel size for the dilation operation.

  Returns:
  numpy.ndarray: Image after dilation.
  """
  kernel = np.ones((kernel_size, kernel_size), np.uint8)
  dilated_image = cv2.dilate(image, kernel, iterations=1)
  return dilated_image
def apply_opening(image, kernel_size=5):
  """
  Apply opening to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  kernel_size (int): Kernel size for the opening operation.

  Returns:
  numpy.ndarray: Image after opening.
  """
  kernel = np.ones((kernel_size, kernel_size), np.uint8)
  opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
  return opened_image
def apply_closing(image, kernel_size=5):
  """
  Apply closing to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  kernel_size (int): Kernel size for the closing operation.

  Returns:
  numpy.ndarray: Image after closing.
  """
  kernel = np.ones((kernel_size, kernel_size), np.uint8)
  closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
  return closed_image
def apply_morphological_gradient(image, kernel_size=5):
  """
  Apply morphological gradient to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  kernel_size (int): Kernel size for the morphological gradient operation.

  Returns:
  numpy.ndarray: Image after morphological gradient.
  """
  kernel = np.ones((kernel_size, kernel_size), np.uint8)
  gradient_image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
  return gradient_image
def apply_top_hat_transform(image, kernel_size=5):
  """
  Apply top hat transform to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  kernel_size (int): Kernel size for the top hat transform operation.

  Returns:
  numpy.ndarray: Image after top hat transform.
  """
  kernel = np.ones((kernel_size, kernel_size), np.uint8)
  top_hat_image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
  return top_hat_image
def apply_black_hat_transform(image, kernel_size=5):
  """
  Apply black hat transform to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  kernel_size (int): Kernel size for the black hat transform operation.

  Returns:
  numpy.ndarray: Image after black hat transform.
  """
  kernel = np.ones((kernel_size, kernel_size), np.uint8)
  black_hat_image = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
  return black_hat_image
def apply_hit_or_miss_transform(image):
  """
  Apply hit-or-miss transform to the input image.

  Parameters:
  image (numpy.ndarray): Input image.

  Returns:
  numpy.ndarray: Image after hit-or-miss transform.
  """
  kernel = np.array([[0, 1, 0], [1, -1, 1], [0, 1, 0]], np.uint8)
  hit_or_miss_image = cv2.morphologyEx(image, cv2.MORPH_HITMISS, kernel)
  return hit_or_miss_image



""" Image Enhancement"""

def histogram_equalization(image):
  """
  Apply histogram equalization to the input image.

  Parameters:
  image (numpy.ndarray): Input image.

  Returns:
  numpy.ndarray: Image after histogram equalization.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  equalized_image = cv2.equalizeHist(gray_image)
  return equalized_image
def clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
  """
  Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  clip_limit (float): Threshold for contrast limiting.
  tile_grid_size (tuple): Size of the grid for histogram equalization.

  Returns:
  numpy.ndarray: Image after CLAHE.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
  clahe_image = clahe.apply(gray_image)
  return clahe_image
def gamma_correction(image, gamma=1.0):
  """
  Apply gamma correction to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  gamma (float): Gamma correction value.

  Returns:
  numpy.ndarray: Image after gamma correction.
  """
  gamma_corrected = np.array(255 * (image / 255) ** gamma, dtype='uint8')
  return gamma_corrected
def sharpen_image(image):
  """
  Apply sharpening filter to the input image.

  Parameters:
  image (numpy.ndarray): Input image.

  Returns:
  numpy.ndarray: Sharpened image.
  """
  kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
  sharpened_image = cv2.filter2D(image, -1, kernel)
  return sharpened_image
def denoise_image(image):
  """
  Apply denoising filter to the input image.

  Parameters:
  image (numpy.ndarray): Input image.

  Returns:
  numpy.ndarray: Denoised image.
  """
  denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
  return denoised_image



"""Image Transformation"""

def translate_image(image, x=0, y=0):
  """
  Translate the input image by a given offset.

  Parameters:
  image (numpy.ndarray): Input image.
  x (int): X-coordinate offset.
  y (int): Y-coordinate offset.

  Returns:
  numpy.ndarray: Translated image.
  """
  rows, cols = image.shape[:2]
  translation_matrix = np.float32([[1, 0, x], [0, 1, y]])
  translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
  return translated_image
def resize_image(image, scale_percent=50):
  """
  Resize the input image by a given scale percentage.

  Parameters:
  image (numpy.ndarray): Input image.
  scale_percent (int): Scale percentage.

  Returns:
  numpy.ndarray: Resized image.
  """
  width = int(image.shape[1] * scale_percent / 100)
  height = int(image.shape[0] * scale_percent / 100)
  resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
  return resized_image
def rotate_image(image, angle=90):
  """
  Rotate the input image by a given angle.

  Parameters:
  image (numpy.ndarray): Input image.
  angle (int): Rotation angle.

  Returns:
  numpy.ndarray: Rotated image.
  """
  (h, w) = image.shape[:2]
  center = (w / 2, h / 2)
  M = cv2.getRotationMatrix2D(center, angle, 1.0)
  rotated_image = cv2.warpAffine(image, M, (w, h))
  return rotated_image
def flip_image(image, flip_code=1):
  """
  Flip the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  flip_code (int): Flip code.

  Returns:
  numpy.ndarray: Flipped image.
  """
  flipped_image = cv2.flip(image, flip_code)
  return flipped_image
def crop_image(image, x=0, y=0, width=100, height=100):
  """
  Crop the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  x (int): X-coordinate of the top-left corner.
  y (int): Y-coordinate of the top-left corner.
  width (int): Width of the cropped image.
  height (int): Height of the cropped image.

  Returns:
  numpy.ndarray: Cropped image.
  """
  cropped_image = image[y:y + height, x:x + width]
  return cropped_image



"""Image Filtering"""

def apply_filter(image, kernel):
  """
  Apply a filter to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  kernel (numpy.ndarray): Filter kernel.

  Returns:
  numpy.ndarray: Filtered image.
  """
  filtered_image = cv2.filter2D(image, -1, kernel)
  return filtered_image
def apply_blur(image, kernel_size=5):
  """
  Apply a blur filter to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  kernel_size (int): Kernel size for the blur filter.

  Returns:
  numpy.ndarray: Blurred image.
  """
  blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
  return blurred_image
def apply_sharpen(image):
  """
  Apply a sharpening filter to the input image.

  Parameters:
  image (numpy.ndarray): Input image.

  Returns:
  numpy.ndarray: Sharpened image.
  """
  kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
  sharpened_image = cv2.filter2D(image, -1, kernel)
  return sharpened_image
def lowpass_filter(image, cutoff_frequency=30, order=1):
  """
  Apply a lowpass filter to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  cutoff_frequency (int): Cutoff frequency for the lowpass filter.
  order (int): Order of the filter.

  Returns:
  numpy.ndarray: Filtered image.
  """
  dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
  dft_shift = np.fft.fftshift(dft)
  rows, cols = image.shape[:2]
  crow, ccol = rows // 2, cols // 2

  mask = np.zeros((rows, cols, 2), np.uint8)
  mask[crow - cutoff_frequency:crow + cutoff_frequency, ccol - cutoff_frequency:ccol + cutoff_frequency] = 1

  fshift = dft_shift * mask
  f_ishift = np.fft.ifftshift(fshift)
  img_back = cv2.idft(f_ishift)
  img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
  return img_back
def highpass_filter(image, cutoff_frequency=30, order=1):
  """
  Apply a highpass filter to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  cutoff_frequency (int): Cutoff frequency for the highpass filter.
  order (int): Order of the filter.

  Returns:
  numpy.ndarray: Filtered image.
  """
  dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
  dft_shift = np.fft.fftshift(dft)
  rows, cols = image.shape[:2]
  crow, ccol = rows // 2, cols // 2

  mask = np.ones((rows, cols, 2), np.uint8)
  mask[crow - cutoff_frequency:crow + cutoff_frequency, ccol - cutoff_frequency:ccol + cutoff_frequency] = 0

  fshift = dft_shift * mask
  f_ishift = np.fft.ifftshift(fshift)
  img_back = cv2.idft(f_ishift)
  img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
  return img_back
def bandpass_filter(image, low_cutoff=30, high_cutoff=60, order=1):
  """
  Apply a bandpass filter to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  low_cutoff (int): Low cutoff frequency for the bandpass filter.
  high_cutoff (int): High cutoff frequency for the bandpass filter.
  order (int): Order of the filter.

  Returns:
  numpy.ndarray: Filtered image.
  """
  dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
  dft_shift = np.fft.fftshift(dft)
  rows, cols = image.shape[:2]
  crow, ccol = rows // 2, cols // 2

  mask = np.zeros((rows, cols, 2), np.uint8)
  mask[crow - high_cutoff:crow + high_cutoff, ccol - high_cutoff:ccol + high_cutoff] = 1
  mask[crow - low_cutoff:crow + low_cutoff, ccol - low_cutoff:ccol + low_cutoff] = 0

  fshift = dft_shift * mask
  f_ishift = np.fft.ifftshift(fshift)
  img_back = cv2.idft(f_ishift)
  img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
  return img_back
def z_transform(image):
  """
  Apply Z-transform to the input image.

  Parameters:
  image (numpy.ndarray): Input image.

  Returns:
  numpy.ndarray: Z-transformed image.
  """
  mean = np.mean(image)
  std = np.std(image)
  z_transformed_image = (image - mean) / std
  return z_transformed_image
def fft(image):
  """
  Apply Fast Fourier Transform (FFT) to the input image.

  Parameters:
  image (numpy.ndarray): Input image.

  Returns:
  numpy.ndarray: FFT of the image.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  f = np.fft.fft2(gray_image)
  fshift = np.fft.fftshift(f)
  magnitude_spectrum = 20 * np.log(np.abs(fshift))
  return magnitude_spectrum



"""Image Segmentation"""

def apply_kmeans(image, num_clusters=3):
  """
  Apply K-Means clustering to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  num_clusters (int): Number of clusters.

  Returns:
  numpy.ndarray: Segmented image.
  """
  reshaped_image = image.reshape((-1, 3))
  kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(reshaped_image)
  segmented_image = kmeans.cluster_centers_[kmeans.labels_]
  segmented_image = segmented_image.reshape(image.shape)
  return
def apply_mean_shift(image, spatial_radius=10, color_radius=10, min_density=100):
  """
  Apply Mean Shift clustering to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  spatial_radius (int): Spatial radius.
  color_radius (int): Color radius.
  min_density (int): Minimum density.

  Returns:
  numpy.ndarray: Segmented image.
  """
  segmented_image = cv2.pyrMeanShiftFiltering(image, spatial_radius, color_radius, min_density)
  return segmented_image
def apply_watershed(image):
  """
  Apply watershed algorithm to the input image.

  Parameters:
  image (numpy.ndarray): Input image.

  Returns:
  numpy.ndarray: Segmented image.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  kernel = np.ones((3, 3), np.uint8)
  opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
  sure_bg = cv2.dilate(opening, kernel, iterations=3)
  dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
  _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
  sure_fg = np.uint8(sure_fg)
  unknown = cv2.subtract(sure_bg, sure_fg)
  _, markers = cv2.connectedComponents(sure_fg)
  markers += 1
  markers[unknown == 255] = 0
  segmented_image = cv2.watershed(image, markers)
  return segmented
def apply_contour_detection(image):
  """
  Apply contour detection to the input image.

  Parameters:
  image (numpy.ndarray): Input image.

  Returns:
  numpy.ndarray: Image with contours.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
  contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  contour_image = cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
  return contour_image
def apply_edge_detection(image, min_val=100, max_val=200):
  """
  Apply edge detection to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  min_val (int): Minimum threshold value.
  max_val (int): Maximum threshold value.

  Returns:
  numpy.ndarray: Image with edges.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray_image, min_val, max_val)
  return
def apply_hough_transform(image, rho=1, theta=np.pi / 180, threshold=100):
  """
  Apply Hough Transform to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  rho (int): Distance resolution of the accumulator in pixels.
  theta (float): Angle resolution of the accumulator in radians.
  threshold (int): Accumulator threshold parameter.

  Returns:
  numpy.ndarray: Image with Hough lines.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
  lines = cv2.HoughLines(edges, rho, theta, threshold)
  hough_image = np.copy(image)
  if lines is not None:
    for line in lines:
      rho, theta = line[0]
      a = np.cos(theta)
      b = np.sin(theta)
      x0 = a * rho
      y0 = b * rho
      x1 = int(x0 + 1000 * (-b))
      y1 = int(y0 + 1000 * (a))
      x2 = int(x0 - 1000 * (-b))
      y2 = int(y0 - 1000 * (a))
      cv2.line(hough_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
  return
def apply_hough_transform_p(image, rho=1, theta=np.pi / 180, threshold=100, min_line_length=100, max_line_gap=10):
  """
  Apply Probabilistic Hough Transform to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  rho (int): Distance resolution of the accumulator in pixels.
  theta (float): Angle resolution of the accumulator in radians.
  threshold (int): Accumulator threshold parameter.
  min_line_length (int): Minimum line length.
  max_line_gap (int): Maximum allowed gap between line segments.

  Returns:
  numpy.ndarray: Image with Hough lines.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
  lines = cv2.HoughLinesP(edges, rho, theta, threshold, min_line_length, max_line_gap)
  hough_image = np.copy(image)
  if lines is not None:
    for line in lines:
      x1, y1, x2, y2 = line[0]
      cv2.line(hough_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
  return hough_image
def apply_contour_approximation(image):
  """
  Apply contour approximation to the input image.

  Parameters:
  image (numpy.ndarray): Input image.

  Returns:
  numpy.ndarray: Image with approximated contours.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
  contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  epsilon = 0.1 * cv2.arcLength(contours[0], True)
  approx = cv2.approxPolyDP(contours[0], epsilon, True)
  approx_image = cv2.drawContours(image, [approx], 0, (0, 255, 0), 3)
  return approx_image
def apply_convex_hull(image):
  """
  Apply convex hull to the input image.

  Parameters:
  image (numpy.ndarray): Input image.

  Returns:
  numpy.ndarray: Image with convex hull.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
  contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  hull = cv2.convexHull(contours[0])
  hull_image = cv2.drawContours(image, [hull], 0, (0, 255, 0), 3)
  return hull_image
def apply_mser(image):
  """
  Apply Maximally Stable Extremal Regions (MSER) to the input image.

  Parameters:
  image (numpy.ndarray): Input image.

  Returns:
  numpy.ndarray: Image with MSER.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  mser = cv2.MSER_create()
  regions, _ = mser.detectRegions(gray_image)
  hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
  mser_image = np.copy(image)
  cv2.polylines(mser_image, hulls, 1, (0, 255, 0))
  return mser_image
def apply_felzenszwalb(image, scale=100, sigma=0.5, min_size=50):
  """
  Apply Felzenszwalb's method to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  scale (int): Free parameter.
  sigma (float): Width of Gaussian kernel.
  min_size (int): Minimum component size.

  Returns:
  numpy.ndarray: Image with segments.
  """
  segments = felzenszwalb(image, scale=scale, sigma=sigma, min_size=min_size)
  felzenszwalb_image = mark_boundaries(image, segments)
  return felzenszwalb_image
def apply_slic(image, num_segments=100, compactness=10.0, max_iter=10):
  """
  Apply Simple Linear Iterative Clustering (SLIC) to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  num_segments (int): Number of segments.
  compactness (float): Balances color proximity and space proximity.
  max_iter (int): Maximum number of iterations.

  Returns:
  numpy.ndarray: Image with segments.
  """
  segments = slic(image, n_segments=num_segments, compactness=compactness, max_iter=max_iter)
  slic_image = mark_boundaries(image, segments)
  return
def apply_quickshift(image, kernel_size=5, max_dist=10, ratio=1.0):
  """
  Apply Quickshift to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  kernel_size (int): Width of Gaussian kernel.
  max_dist (int): Cut-off point for data distances.
  ratio (float): Balances color-space proximity and image-space proximity.

  Returns:
  numpy.ndarray: Image with segments.
  """
  segments = quickshift(image, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio)
  quickshift_image = mark_boundaries(image, segments)
  return quickshift_image
def apply_watershed_segmentation(image):
  """
  Apply watershed segmentation to the input image.

  Parameters:
  image (numpy.ndarray): Input image.

  Returns:
  numpy.ndarray: Image with segments.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  negative_image = cv2.bitwise_not(binary_image)
  markers = cv2.connectedComponents(negative_image)[1]
  segments = watershed(image, markers)
  watershed_image = mark_boundaries(image, segments)
  return watershed_image



"""Image Restoration"""

def inpainting_deconvolution(image, kernel, iterations=10):
  """
  Apply inpainting deconvolution to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  kernel (numpy.ndarray): Deconvolution kernel.
  iterations (int): Number of iterations.

  Returns:
  numpy.ndarray: Image after inpainting deconvolution.
  """
  deconvolved_image = cv2.filter2D(image, -1, kernel)
  for _ in range(iterations):
    deconvolved_image = cv2.filter2D(deconvolved_image, -1, kernel)
  return deconvolved_image
def total_variation_denoising(image, weight=0.1, max_iter=100):
  """
  Apply total variation denoising to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  weight (float): Denoising weight.
  max_iter (int): Maximum number of iterations.

  Returns:
  numpy.ndarray: Denoised image.
  """
  denoised_image = cv2.denoise_TVL1([image], weight, max_iter)[0]
  return denoised_image
def anisotropic_diffusion(image, num_iter=10, kappa=50, gamma=0.1, option=1):
  """
  Apply anisotropic diffusion to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  num_iter (int): Number of iterations.
  kappa (float): Conduction coefficient.
  gamma (float): Integration constant.
  option (int): Diffusion equation option.

  Returns:
  numpy.ndarray: Image after anisotropic diffusion.
  """
  img = image.astype(np.float32)
  for _ in range(num_iter):
    nabla_north = np.roll(img, -1, axis=0) - img
    nabla_south = np.roll(img, 1, axis=0) - img
    nabla_east = np.roll(img, -1, axis=1) - img
    nabla_west = np.roll(img, 1, axis=1) - img

    if option == 1:
      c_north = np.exp(-(nabla_north / kappa) ** 2)
      c_south = np.exp(-(nabla_south / kappa) ** 2)
      c_east = np.exp(-(nabla_east / kappa) ** 2)
      c_west = np.exp(-(nabla_west / kappa) ** 2)
    elif option == 2:
      c_north = 1.0 / (1.0 + (nabla_north / kappa) ** 2)
      c_south = 1.0 / (1.0 + (nabla_south / kappa) ** 2)
      c_east = 1.0 / (1.0 + (nabla_east / kappa) ** 2)
      c_west = 1.0 / (1.0 + (nabla_west / kappa) ** 2)

    img += gamma * (
      c_north * nabla_north + c_south * nabla_south +
      c_east * nabla_east + c_west * nabla_west
    )
  return img.astype(np.uint8)
def richardson_lucy_deconvolution(image, psf, iterations=10):
  """
  Apply Richardson-Lucy deconvolution to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  psf (numpy.ndarray): Point Spread Function (PSF).
  iterations (int): Number of iterations.

  Returns:
  numpy.ndarray: Image after Richardson-Lucy deconvolution.
  """
  deconvolved_image = restoration.richardson_lucy(image, psf, iterations=iterations)
  return deconvolved
def blind_deconvolution(image, psf, iterations=10):
  """
  Apply blind deconvolution to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  psf (numpy.ndarray): Point Spread Function (PSF).
  iterations (int): Number of iterations.

  Returns:
  numpy.ndarray: Image after blind deconvolution.
  """
  deconvolved_image = restoration.unsupervised_wiener(image, psf, iterations=iterations)
  return deconvolved_image
def non_local_means_denoising(image, h=10, search_window=21, block_size=7):
  """
  Apply non-local means denoising to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  h (int): Filter strength.
  search_window (int): Size of the search window.
  block_size (int): Size of the block.

  Returns:
  numpy.ndarray: Denoised image.
  """
  denoised_image = cv2.fastNlMeansDenoising(image, h=h, templateWindowSize=block_size, searchWindowSize=search_window)
  return denoised_image
def bilateral_filtering(image, d=9, sigma_color=75, sigma_space=75):
  """
  Apply bilateral filtering to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  d (int): Diameter of each pixel neighborhood.
  sigma_color (int): Filter sigma in the color space.
  sigma_space (int): Filter sigma in the coordinate space.

  Returns:
  numpy.ndarray: Image after bilateral filtering.
  """
  denoised_image = cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
  return denoised_image
def median_filtering(image, kernel_size=5):
  """
  Apply median filtering to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  kernel_size (int): Size of the kernel.

  Returns:
  numpy.ndarray: Image after median filtering.
  """
  denoised_image = cv2.medianBlur(image, kernel_size)
  return denoised_image
def wiener_filtering(image, kernel, noise_var=0.01):
  """
  Apply Wiener filtering to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  kernel (numpy.ndarray): Point Spread Function (PSF).
  noise_var (float): Noise variance.

  Returns:
  numpy.ndarray: Image after Wiener filtering.
  """
  deconvolved_image = restoration.wiener(image, kernel, noise_var)
  return deconvolved_image



"""Gradients"""

def apply_sobel_operator(image, dx=1, dy=1, ksize=3):
  """
  Apply Sobel operator to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  dx (int): Order of the derivative in x-direction.
  dy (int): Order of the derivative in y-direction.
  ksize (int): Size of the extended Sobel kernel.

  Returns:
  numpy.ndarray: Image after applying Sobel operator.
  """
  sobelx = cv2.Sobel(image, cv2.CV_64F, dx, dy, ksize=ksize)
  return sobelx
def apply_scharr_operator(image, dx=1, dy=0, scale=1, delta=0):
  """
  Apply Scharr operator to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  dx (int): Order of the derivative in x-direction.
  dy (int): Order of the derivative in y-direction.
  scale (int): Optional scale factor for the computed derivative values.
  delta (int): Optional delta value that is added to the results.

  Returns:
  numpy.ndarray: Image after applying Scharr operator.
  """
  scharrx = cv2.Scharr(image, cv2.CV_64F, dx, dy, scale=scale, delta=delta)
  return scharrx
def apply_laplacian_operator(image, ksize=3):
  """
  Apply Laplacian operator to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  ksize (int): Optional kernel size.

  Returns:
  numpy.ndarray: Image after applying Laplacian operator.
  """
  laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=ksize)
  return laplacian
def apply_canny_edge_detection(image, threshold1=100, threshold2=200):
  """
  Apply Canny edge detection to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  threshold1 (int): First threshold for the hysteresis procedure.
  threshold2 (int): Second threshold for the hysteresis procedure.

  Returns:
  numpy.ndarray: Image after applying Canny edge detection.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray_image, threshold1, threshold2)
  return edges



"""Texture Features"""

def apply_gabor_filter(image, ksize=31, sigma=4.0, theta=1.0, lambd=10.0, gamma=0.5, psi=0):
  """
  Applies a Gabor filter to the input image.

  Parameters:
  image (numpy.ndarray): The input image on which the Gabor filter is to be applied.
  ksize (int, optional): Size of the filter (ksize x ksize). Default is 31.
  sigma (float, optional): Standard deviation of the Gaussian function used in the Gabor filter. Default is 4.0.
  theta (float, optional): Orientation of the normal to the parallel stripes of the Gabor function. Default is 1.0.
  lambd (float, optional): Wavelength of the sinusoidal factor. Default is 10.0.
  gamma (float, optional): Spatial aspect ratio. Default is 0.5.
  psi (float, optional): Phase offset. Default is 0.

  Returns:
  numpy.ndarray: The filtered image after applying the Gabor filter.
  
  The function uses `cv2.getGaborKernel` to create a Gabor kernel with the specified parameters 
  and then applies this kernel to the input image using `cv2.filter2D`.
  """
  gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
  filtered_image = cv2.filter2D(image, cv2.CV_8UC3, gabor_kernel)
  return filtered_image
def compute_cooccurrence_matrix(image, distances=[1], angles=[0], levels=256, 
symmetric=True, normed=True):
    """
    Compute the co-occurrence matrix of an image.

    Parameters:
    image (numpy.ndarray): Input image.
    distances (list of int): List of pixel pair distance offsets.
    angles (list of float): List of angles in radians for pixel pairs.
    levels (int): Number of gray levels in the image.
    symmetric (bool): If True, the co-occurrence matrix is symmetric.
    normed (bool): If True, normalize the co-occurrence matrix.

    Returns:
    numpy.ndarray: Co-occurrence matrix of the image.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    co_matrix = cv2.createCLAHE().apply(gray_image)
    return co_matrix
def compute_lbp(image, radius=1, n_points=8):
    """
    Compute the Local Binary Pattern (LBP) of an image.

    Parameters:
    image (numpy.ndarray): Input image.
    radius (int): Radius of the circle.
    n_points (int): Number of points to consider in the circle.

    Returns:
    numpy.ndarray: LBP image.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = cv2.LBP(gray_image, radius, n_points)
    return lbp
def compute_glcm_features(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True):
  """
  Compute the Gray Level Co-occurrence Matrix (GLCM) features of an image.

  Parameters:
  image (numpy.ndarray): Input image.
  distances (list of int): List of pixel pair distance offsets.
  angles (list of float): List of angles in radians for pixel pairs.
  levels (int): Number of gray levels in the image.
  symmetric (bool): If True, the co-occurrence matrix is symmetric.
  normed (bool): If True, normalize the co-occurrence matrix.

  Returns:
  dict: Dictionary of GLCM features.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  glcm = greycomatrix(gray_image, distances=distances, angles=angles, 
  levels=levels, symmetric=symmetric, normed=normed)
  features = {
    'contrast': greycoprops(glcm, 'contrast'),
    'dissimilarity': greycoprops(glcm, 'dissimilarity'),
    'homogeneity': greycoprops(glcm, 'homogeneity'),
    'energy': greycoprops(glcm, 'energy'),
    'correlation': greycoprops(glcm, 'correlation'),
    'ASM': greycoprops(glcm, 'ASM')
  }
  return features
def compute_lbp_features(image, radius=1, n_points=8):
  """
  Compute the Local Binary Pattern (LBP) features of an image.

  Parameters:
  image (numpy.ndarray): Input image.
  radius (int): Radius of the circle.
  n_points (int): Number of points to consider in the circle.

  Returns:
  numpy.ndarray: LBP image.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
  return lbp
def compute_haralick_features(image):
  """
  Compute the Haralick texture features of an image.

  Parameters:
  image (numpy.ndarray): Input image.

  Returns:
  numpy.ndarray: Haralick features.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  haralick_features = mahotas.features.haralick(gray_image).mean(axis=0)
  return haralick_features
def compute_gabor_features(image, frequencies=[0.1, 0.3, 0.5, 0.7, 0.9]):
  """
  Compute the Gabor filter features of an image.

  Parameters:
  image (numpy.ndarray): Input image.
  frequencies (list of float): List of frequencies for the Gabor filter.

  Returns:
  list: List of Gabor filter responses.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gabor_features = []
  for frequency in frequencies:
    real, imag = gabor(gray_image, frequency=frequency)
    gabor_features.append(real)
    gabor_features.append(imag)
  return gabor_features   
  


"""Shape Features"""

def compute_contours(image):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  return contours
def compute_hu_moments(image):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  moments = cv2.moments(gray_image)
  hu_moments = cv2.HuMoments(moments).flatten()
  return hu_moments
def compute_fourier_descriptors(image):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if contours:
    contour = max(contours, key=cv2.contourArea)
    contour_array = contour[:, 0, :]
    fourier_result = np.fft.fft(contour_array)
    return fourier_result
  return None
def compute_shape_context(image):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if contours:
    contour = max(contours, key=cv2.contourArea)
    shape_context = cv2.createShapeContextDistanceExtractor()
    return shape_context.computeDistance(contour, contour)
  return None
def compute_skeletonization(image):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
  skeleton = cv2.ximgproc.thinning(binary_image)
  return skeleton
def compute_convex_hull(image):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  hull = [cv2.convexHull(contour) for contour in contours]
  hull_image = np.zeros_like(image)
  cv2.drawContours(hull_image, hull, -1, (255, 255, 255), 2)
  return hull_image
  


"""Color Features"""

def compute_color_coherence_vector(image, threshold=0.1):
  """
  Compute the Color Coherence Vector (CCV) of an image.

  Parameters:
  image (numpy.ndarray): Input image.
  threshold (float): Threshold to determine coherent vs. incoherent pixels.

  Returns:
  dict: Dictionary with coherent and incoherent pixel counts for each color.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  _, binary_image = cv2.threshold(gray_image, int(threshold * 255), 255, cv2.THRESH_BINARY)
  num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
  
  ccv = {}
  for label in range(1, num_labels):
    color = tuple(image[labels == label][0])
    if color not in ccv:
      ccv[color] = {'coherent': 0, 'incoherent': 0}
    if stats[label, cv2.CC_STAT_AREA] >= threshold * image.size:
      ccv[color]['coherent'] += stats[label, cv2.CC_STAT_AREA]
    else:
      ccv[color]['incoherent'] += stats[label, cv2.CC_STAT_AREA]
  
  return ccv
def compute_dominant_color(image, k=3):
  """
  Compute the dominant color of an image using K-Means clustering.

  Parameters:
  image (numpy.ndarray): Input image.
  k (int): Number of clusters.

  Returns:
  numpy.ndarray: Dominant color.
  """
  data = image.reshape((-1, 3))
  data = np.float32(data)
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
  _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  dominant_color = centers[np.argmax(np.bincount(labels.flatten()))]
  return dominant_color
def compute_color_layout_descriptor(image, grid_size=(8, 8)):
  """
  Compute the Color Layout Descriptor (CLD) of an image.

  Parameters:
  image (numpy.ndarray): Input image.
  grid_size (tuple): Size of the grid to divide the image into.

  Returns:
  numpy.ndarray: Color layout descriptor.
  """
  h, w, _ = image.shape
  grid_h, grid_w = grid_size
  cld = []

  for i in range(grid_h):
    for j in range(grid_w):
      grid_image = image[i * h // grid_h:(i + 1) * h // grid_h, j * w // grid_w:(j + 1) * w // grid_w]
      mean_color = cv2.mean(grid_image)[:3]
      cld.extend(mean_color)
  
  return np.array(cld)
def compute_color_histograms(image, bins=(8, 8, 8)):
  hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
  cv2.normalize(hist, hist)
  return hist.flatten()
def compute_color_moments(image):
  moments = cv2.meanStdDev(image)
  color_moments = np.concatenate([moments[0], moments[1]]).flatten()
  return color_moments
def compute_color_correlogram(image, distance=1):
  correlogram = np.zeros((256, 256))
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      for di in range(-distance, distance + 1):
        for dj in range(-distance, distance + 1):
          if 0 <= i + di < image.shape[0] and 0 <= j + dj < image.shape[1]:
            correlogram[gray_image[i, j], gray_image[i + di, j + dj]] += 1
  return correlogram
def compute_color_cooccurrence_matrix(image, distances=[1], angles=[0], 
levels=256, symmetric=True, normed=True):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  co_matrix = cv2.createCLAHE().apply(gray_image)
  return co_matrix
def compute_dominant_color(image, k=3):
  data = image.reshape((-1, 3))
  data = np.float32(data)
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
  _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  dominant_color = centers[np.argmax(np.bincount(labels.flatten()))]
  return dominant_color



"""Edge features"""

def compute_edge_maps(image):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray_image, 100, 200)
  return edges
def compute_edge_histograms(image, bins=16):
  edges = compute_edge_maps(image)
  hist = cv2.calcHist([edges], [0], None, [bins], [0, 256])
  cv2.normalize(hist, hist)
  return hist.flatten()
def canny_edge_detection(image, threshold1=100, threshold2=200):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray_image, threshold1, threshold2)
  return edges
def sobel_edge_detection(image, ksize=3):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=ksize)
  sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=ksize)
  sobel = cv2.magnitude(sobelx, sobely)
  return sobel
def log_edge_detection(image, ksize=5):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray_image, (ksize, ksize), 0)
  log = cv2.Laplacian(blur, cv2.CV_64F)
  return log
def apply_sobel_x(image):
  """
  Apply Sobel edge detection in the x-direction to the input image.

  Parameters:
  image (numpy.ndarray): Input image.

  Returns:
  numpy.ndarray: Image after Sobel edge detection in the x-direction.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
  sobelx = cv2.convertScaleAbs(sobelx)
  return sobelx
def apply_sobel_y(image):
  """
  Apply Sobel edge detection in the y-direction to the input image.

  Parameters:
  image (numpy.ndarray): Input image.

  Returns:
  numpy.ndarray: Image after Sobel edge detection in the y-direction.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
  sobely = cv2.convertScaleAbs(sobely)
  return sobely
def apply_sobel(image):
  """
  Apply Sobel edge detection to the input image.

  Parameters:
  image (numpy.ndarray): Input image.

  Returns:
  numpy.ndarray: Image after Sobel edge detection.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
  sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
  sobel = cv2.magnitude(sobelx, sobely)
  return sobel



"""Contour features"""

def compute_contour_area(image):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
  contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contour = max(contours, key=cv2.contourArea)
  area = cv2.contourArea(contour)
  return area
def compute_contour_perimeter(image):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
  contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contour = max(contours, key=cv2.contourArea)
  perimeter = cv2.arcLength(contour, True)
  return perimeter
def compute_contour_moments(image):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
  contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contour = max(contours, key=cv2.contourArea)
  moments = cv2.moments(contour)
  return moments
def compute_contour_orientation(image):
  moments = compute_contour_moments(image)
  angle = 0.5 * np.arctan((2 * moments['mu11']) / (moments['mu20'] - moments['mu02']))
  return angle
def compute_contour_aspect_ratio(image):
  moments = compute_contour_moments(image)
  aspect_ratio = (moments['mu20'] + moments['mu02'] + np.sqrt(4 * moments['mu11'] ** 2 + (moments['mu20'] - moments['mu02']) ** 2)) / (moments['mu20'] + moments['mu02'] - np.sqrt(4 * moments['mu11'] ** 2 + (moments['mu20'] - moments['mu02']) ** 2))
  return aspect_ratio
def compute_contour_roundness(image):
  area = compute_contour_area(image)
  perimeter = compute_contour_perimeter(image)
  roundness = (4 * np.pi * area) / (perimeter ** 2)
  return roundness
def compute_contour_solidity(image):
  area = compute_contour_area(image)
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
  contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contour = max(contours, key=cv2.contourArea)
  hull = cv2.convexHull(contour)
  hull_area = cv2.contourArea(hull)
  solidity = area / hull_area
  return solidity
def compute_contour_extent(image):
  area = compute_contour_area(image)
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
  contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contour = max(contours, key=cv2.contourArea)
  x, y, w, h = cv2.boundingRect(contour)
  bounding_area = w * h
  extent = area / bounding_area
  return extent
def compute_contour_eccentricity(image):
  moments = compute_contour_moments(image)
  major_axis_length = 2 * np.sqrt(moments['mu20'] + moments['mu02'] + np.sqrt(4 * moments['mu11'] ** 2 + (moments['mu20'] - moments['mu02']) ** 2))
  minor_axis_length = 2 * np.sqrt(moments['mu20'] + moments['mu02'] - np.sqrt(4 * moments['mu11'] ** 2 + (moments['mu20'] - moments['mu02']) ** 2))
  eccentricity = np.sqrt(1 - (minor_axis_length / major_axis_length) ** 2)
  return eccentricity
def compute_contour_convexity(image):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
  contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contour = max(contours, key=cv2.contourArea)
  hull = cv2.convexHull(contour)
  area = cv2.contourArea(contour)
  hull_area = cv2.contourArea(hull)
  convexity = area / hull_area
  return convexity
def compute_contour_hu_moments(image):
  moments = compute_contour_moments(image)
  hu_moments = cv2.HuMoments(moments).flatten()
  return hu_moments



"""Interest Point Detection"""

def harris_corner_detection(image, block_size=2, ksize=3, k=0.04):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gray_image = np.float32(gray_image)
  dst = cv2.cornerHarris(gray_image, block_size, ksize, k)
  dst = cv2.dilate(dst, None)
  image[dst > 0.01 * dst.max()] = [0, 0, 255]
  return image
def blob_detection(image, min_threshold=10, max_threshold=200, min_area=1500):

  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  params = cv2.SimpleBlobDetector_Params()
  params.minThreshold = min_threshold
  params.maxThreshold = max_threshold
  params.filterByArea = True
  params.minArea = min_area
  detector = cv2.SimpleBlobDetector_create(params)
  keypoints = detector.detect(gray_image)
  image_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  return image_with_keypoints
def ridge_detection(image):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  ridge_filter = cv2.ximgproc.RidgeDetectionFilter_create()
  ridges = ridge_filter.getRidgeFilteredImage(gray_image)
  return ridges
def sift_detection(image):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  sift = cv2.SIFT_create()
  keypoints, descriptors = sift.detectAndCompute(gray_image, None)
  image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)
  return image_with_keypoints

      

"""Local Features"""

def surf_detection(image):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  surf = cv2.xfeatures2d.SURF_create()
  keypoints, descriptors = surf.detectAndCompute(gray_image, None)
  image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)
  return image_with_keypoints
def orb_detection(image):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  orb = cv2.ORB_create()
  keypoints, descriptors = orb.detectAndCompute(gray_image, None)
  image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)
  return image_with_keypoints
def brief_detection(image):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  star = cv2.xfeatures2d.StarDetector_create()
  brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
  keypoints = star.detect(gray_image, None)
  keypoints, descriptors = brief.compute(gray_image, keypoints)
  image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)
  return image_with_keypoints
def liop_detection(image):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  liop = cv2.xfeatures2d.LATCH_create()
  keypoints = cv2.goodFeaturesToTrack(gray_image, maxCorners=500, qualityLevel=0.01, minDistance=10)
  keypoints = [cv2.KeyPoint(x[0][0], x[0][1], 1) for x in keypoints]
  keypoints, descriptors = liop.compute(gray_image, keypoints)
  image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)
  return image_with_keypoints
def hog_detection(image):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  hog = cv2.HOGDescriptor()
  h = hog.compute(gray_image)
  return h



"""Clustering"""

def kmeans_clustering(image, k=3):
  """
  Apply K-Means clustering to segment the image.

  Parameters:
  image (numpy.ndarray): Input image.
  k (int): Number of clusters.

  Returns:
  numpy.ndarray: Clustered image.
  """
  data = image.reshape((-1, 3))
  data = np.float32(data)
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
  _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  centers = np.uint8(centers)
  clustered_image = centers[labels.flatten()]
  clustered_image = clustered_image.reshape(image.shape)
  return clustered_image
def hierarchical_clustering(image, n_clusters=3):
  """
  Apply Hierarchical clustering to segment the image.

  Parameters:
  image (numpy.ndarray): Input image.
  n_clusters (int): Number of clusters.

  Returns:
  numpy.ndarray: Clustered image.
  """
  data = image.reshape((-1, 3))
  clustering = AgglomerativeClustering(n_clusters=n_clusters)
  labels = clustering.fit_predict(data)
  clustered_image = labels.reshape(image.shape[:2])
  return clustered_image
def fuzzy_cmeans_clustering(image, n_clusters=3):
  """
  Apply Fuzzy C-Means clustering to segment the image.

  Parameters:
  image (numpy.ndarray): Input image.
  n_clusters (int): Number of clusters.

  Returns:
  numpy.ndarray: Clustered image.
  """
  data = image.reshape((-1, 3)).T
  cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(data, n_clusters, 2, error=0.005, maxiter=1000)
  labels = np.argmax(u, axis=0)
  clustered_image = labels.reshape(image.shape[:2])
  return clustered_image



"""Region Growing"""

def region_growing(image, seed_point, threshold=10):
  """
  Apply region growing algorithm to segment the image.

  Parameters:
  image (numpy.ndarray): Input image.
  seed_point (tuple): The starting point (x, y) for region growing.
  threshold (int): The threshold value to determine the region.

  Returns:
  numpy.ndarray: Segmented image.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  height, width = gray_image.shape
  segmented_image = np.zeros_like(gray_image)
  seed_value = gray_image[seed_point[1], seed_point[0]]
  stack = [seed_point]

  while stack:
    x, y = stack.pop()
    if segmented_image[y, x] == 0:
      segmented_image[y, x] = 255
      for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
          nx, ny = x + dx, y + dy
          if 0 <= nx < width and 0 <= ny < height:
            if abs(int(gray_image[ny, nx]) - int(seed_value)) < threshold:
              stack.append((nx, ny))

  return segmented_image



"""Triangulation"""

def delaunay_triangulation(image):
  """
  Apply Delaunay triangulation to the input image.

  Parameters:
  image (numpy.ndarray): Input image.

  Returns:
  numpy.ndarray: Image with Delaunay triangulation applied.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  height, width = gray_image.shape
  subdiv = cv2.Subdiv2D((0, 0, width, height))

  points = []
  for y in range(0, height, 10):
    for x in range(0, width, 10):
      points.append((x, y))

  for p in points:
    subdiv.insert(p)

  triangles = subdiv.getTriangleList()
  triangles = np.array(triangles, dtype=np.int32)

  triangulated_image = image.copy()
  for t in triangles:
    pt1 = (t[0], t[1])
    pt2 = (t[2], t[3])
    pt3 = (t[4], t[5])
    cv2.line(triangulated_image, pt1, pt2, (255, 0, 0), 1)
    cv2.line(triangulated_image, pt2, pt3, (255, 0, 0), 1)
    cv2.line(triangulated_image, pt3, pt1, (255, 0, 0), 1)

  return triangulated_image
def voronoi_diagram(image):
  """
  Apply Voronoi diagram to the input image.

  Parameters:
  image (numpy.ndarray): Input image.

  Returns:
  numpy.ndarray: Image with Voronoi diagram applied.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  height, width = gray_image.shape
  subdiv = cv2.Subdiv2D((0, 0, width, height))

  points = []
  for y in range(0, height, 10):
    for x in range(0, width, 10):
      points.append((x, y))

  for p in points:
    subdiv.insert(p)

  (facets, centers) = subdiv.getVoronoiFacetList([])

  voronoi_image = image.copy()
  for i in range(len(facets)):
    ifacet_arr = []
    for f in facets[i]:
      ifacet_arr.append(f)
    ifacet = np.array(ifacet_arr, np.int)
    color = (0, 255, 0)
    cv2.fillConvexPoly(voronoi_image, ifacet, color)
    cv2.polylines(voronoi_image, [ifacet], True, (0, 0, 0), 1)
    cv2.circle(voronoi_image, (centers[i][0], centers[i][1]), 3, (0, 0, 255), -1)

  return voronoi_image



"""Image Blending"""

def blend_images(image1, image2, alpha=0.5, beta=0.5, gamma=0):
  """
  Blend two images together.

  Parameters:
  image1 (numpy.ndarray): First input image.
  image2 (numpy.ndarray): Second input image.
  alpha (float): Weight for the first image.
  beta (float): Weight for the second image.
  gamma (float): Scalar added to each sum.

  Returns:
  numpy.ndarray: Blended image.
  """
  blended_image = cv2.addWeighted(image1, alpha, image2, beta, gamma)
  return blended_image
def alpha_blending(image1, image2, alpha=0.5):
  """
  Apply alpha blending to blend two images.

  Parameters:
  image1 (numpy.ndarray): First input image.
  image2 (numpy.ndarray): Second input image.
  alpha (float): Weight for the first image.

  Returns:
  numpy.ndarray: Blended image.
  """
  blended_image = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)
  return blended_image
def gradient_blending(image1, image2, mask):
  """
  Apply gradient blending to blend two images using a mask.

  Parameters:
  image1 (numpy.ndarray): First input image.
  image2 (numpy.ndarray): Second input image.
  mask (numpy.ndarray): Mask to blend the images.

  Returns:
  numpy.ndarray: Blended image.
  """
  blended_image = cv2.seamlessClone(image1, image2, mask, (image2.shape[1]//2, image2.shape[0]//2), cv2.NORMAL_CLONE)
  return blended_image
def pyramid_blending(image1, image2, levels=6):
  """
  Apply pyramid blending to blend two images.

  Parameters:
  image1 (numpy.ndarray): First input image.
  image2 (numpy.ndarray): Second input image.
  levels (int): Number of pyramid levels.

  Returns:
  numpy.ndarray: Blended image.
  """
  # Generate Gaussian pyramid for image1
  G1 = image1.copy()
  gp1 = [G1]
  for i in range(levels):
    G1 = cv2.pyrDown(G1)
    gp1.append(G1)

  # Generate Gaussian pyramid for image2
  G2 = image2.copy()
  gp2 = [G2]
  for i in range(levels):
    G2 = cv2.pyrDown(G2)
    gp2.append(G2)

  # Generate Laplacian pyramid for image1
  lp1 = [gp1[levels]]
  for i in range(levels, 0, -1):
    GE = cv2.pyrUp(gp1[i])
    L = cv2.subtract(gp1[i-1], GE)
    lp1.append(L)

  # Generate Laplacian pyramid for image2
  lp2 = [gp2[levels]]
  for i in range(levels, 0, -1):
    GE = cv2.pyrUp(gp2[i])
    L = cv2.subtract(gp2[i-1], GE)
    lp2.append(L)

  # Add left and right halves of images in each level
  LS = []
  for l1, l2 in zip(lp1, lp2):
    rows, cols, dpt = l1.shape
    ls = np.hstack((l1[:, 0:cols//2], l2[:, cols//2:]))
    LS.append(ls)

  # Reconstruct the image
  blended_image = LS[0]
  for i in range(1, levels+1):
    blended_image = cv2.pyrUp(blended_image)
    blended_image = cv2.add(blended_image, LS[i])

  return blended_image
def image_stitching(images):
  """
  Apply image stitching to combine multiple images into a panorama.

  Parameters:
  images (list of numpy.ndarray): List of input images.

  Returns:
  numpy.ndarray: Stitched image.
  """
  stitcher = cv2.Stitcher_create()
  status, stitched_image = stitcher.stitch(images)
  if status != cv2.Stitcher_OK:
    print("Error during stitching")
    return None
  return stitched_image
def seamless_cloning(src, dst, mask, center):
  """
  Apply seamless cloning to blend the source image into the destination image.

  Parameters:
  src (numpy.ndarray): Source image.
  dst (numpy.ndarray): Destination image.
  mask (numpy.ndarray): Mask to specify the region to blend.
  center (tuple): Center point (x, y) for seamless cloning.

  Returns:
  numpy.ndarray: Image with seamless cloning applied.
  """
  cloned_image = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
  return cloned_image
def add_noise(image, noise_type='gaussian', mean=0, std=25):
  """
  Add noise to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  noise_type (str): Type of noise to add.
  mean (int): Mean value for Gaussian noise.
  std (int): Standard deviation for Gaussian noise.

  Returns:
  numpy.ndarray: Image with added noise.
  """
  if noise_type == 'gaussian':
    noise = np.random.normal(mean, std, image.shape)
  elif noise_type == 'salt_and_pepper':
    noise = np.random.randint(0, 2, image.shape) * 255
  noisy_image = cv2.addWeighted(image, 0.5, noise, 0.5, 0)
  return noisy_image
def blend_images(image1, image2, alpha=0.5, beta=0.5, gamma=0):
  """
  Blend two images together.

  Parameters:
  image1 (numpy.ndarray): First input image.
  image2 (numpy.ndarray): Second input image.
  alpha (float): Weight for the first image.
  beta (float): Weight for the second image.
  gamma (float): Scalar added to each sum.

  Returns:
  numpy.ndarray: Blended image.
  """
  blended_image = cv2.addWeighted(image1, alpha, image2, beta, gamma)
  return blended_image
def apply_mask(image, mask):
  """
  Apply a mask to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  mask (numpy.ndarray): Mask to apply.

  Returns:
  numpy.ndarray: Image with mask applied.
  """
  masked_image = cv2.bitwise_and(image, image, mask=mask)
  return masked_image
def apply_threshold(image, threshold=128, max_value=255, threshold_type=cv2.THRESH_BINARY):
  """
  Apply a threshold to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  threshold (int): Threshold value.
  max_value (int): Maximum value.
  threshold_type (int): Threshold type.

  Returns:
  numpy.ndarray: Image with threshold applied.
  """
  _, thresholded_image = cv2.threshold(image, threshold, max_value, threshold_type)
  return thresholded_image
def apply_morphology(image, kernel_size=5, operation='erode'):
  """
  Apply morphological operations to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  kernel_size (int): Kernel size for the structuring element.
  operation (str): Morphological operation to apply.

  Returns:
  numpy.ndarray: Image after morphological operation.
  """
  kernel = np.ones((kernel_size, kernel_size), np.uint8)
  if operation == 'erode':
    morphed_image = cv2.erode(image, kernel, iterations=1)
  elif operation == 'dilate':
    morphed_image = cv2.dilate(image, kernel, iterations=1)
  elif operation == 'opening':
    morphed_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
  elif operation == 'closing':
    morphed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
  return morphed_image
def apply_contour(image):
  """
  Apply contour detection to the input image.

  Parameters:
  image (numpy.ndarray): Input image.

  Returns:
  numpy.ndarray: Image with contours applied.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contour_image = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)
  return contour_image

  """Structure from Motion (SfM)"""



"""2d to 3d conversion"""


"""Structure from Stereo (SfS)"""
def estimate_depth_map(left_image, right_image, num_disparities=16, block_size=15):
  """
  Estimate depth map from stereo images.

  Parameters:
  left_image (numpy.ndarray): Left input image.
  right_image (numpy.ndarray): Right input image.
  num_disparities (int): Number of disparities.
  block_size (int): Block size.

  Returns:
  numpy.ndarray: Depth map.
  """
  stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
  disparity = stereo.compute(left_image, right_image)
  depth_map = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
  return depth_map
def reconstruct_3D_from_stereo(left_image, right_image, focal_length, baseline):
  """
  Reconstruct 3D points from stereo images.

  Parameters:
  left_image (numpy.ndarray): Left input image.
  right_image (numpy.ndarray): Right input image.
  focal_length (float): Focal length of the camera.
  baseline (float): Distance between the two cameras.

  Returns:
  numpy.ndarray: 3D points.
  """
  depth_map = estimate_depth_map(left_image, right_image)
  height, width = depth_map.shape
  Q = np.float32([[1, 0, 0, -width / 2],
          [0, -1, 0, height / 2],
          [0, 0, 0, -focal_length],
          [0, 0, 1 / baseline, 0]])
  points_3D = cv2.reprojectImageTo3D(depth_map, Q)
  return points_3D

"""Shape from Shading (SfS)"""
def estimate_surface_normals(image, light_direction):
  """
  Estimate surface normals from the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  light_direction (numpy.ndarray): Light direction vector.

  Returns:
  numpy.ndarray: Surface normals.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
  gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
  normals = np.dstack((-gradient_x, -gradient_y, np.ones_like(gray_image)))
  normals /= np.linalg.norm(normals, axis=2, keepdims=True)
  return normals
def estimate_depth_map_from_normals(normals, light_direction):
  """
  Estimate depth map from surface normals.

  Parameters:
  normals (numpy.ndarray): Surface normals.
  light_direction (numpy.ndarray): Light direction vector.

  Returns:
  numpy.ndarray: Depth map.
  """
  height, width, _ = normals.shape
  depth_map = np.zeros((height, width), dtype=np.float32)
  for y in range(1, height):
    for x in range(1, width):
      depth_map[y, x] = depth_map[y - 1, x] + normals[y, x, 1] / normals[y, x, 2]
      depth_map[y, x] += depth_map[y, x - 1] + normals[y, x, 0] / normals[y, x, 2]
  return depth_map


"""Shape from Silhouette (SfS)"""
def extract_silhouette(image, threshold=127):
  """
  Extract silhouette from the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  threshold (int): Threshold value.

  Returns:
  numpy.ndarray: Silhouette image.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  _, silhouette = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
  return silhouette
def volume_intersection(silhouettes, projection_matrices):
  """
  Perform volume intersection to reconstruct 3D shape from silhouettes.

  Parameters:
  silhouettes (list of numpy.ndarray): List of silhouette images.
  projection_matrices (list of numpy.ndarray): List of projection matrices.

  Returns:
  numpy.ndarray: 3D volume.
  """
  # Placeholder for volume intersection implementation
  # This typically involves voxel carving and is complex to implement from scratch
  return None
def reconstruct_3D_from_silhouettes(silhouettes, projection_matrices):
  """
  Reconstruct 3D shape from silhouettes.

  Parameters:
  silhouettes (list of numpy.ndarray): List of silhouette images.
  projection_matrices (list of numpy.ndarray): List of projection matrices.

  Returns:
  numpy.ndarray: 3D shape.
  """
  volume = volume_intersection(silhouettes, projection_matrices)
  return volume

def apply_disparity_map(left_image, right_image, num_disparities=16, block_size=15):
  """
  Apply disparity map calculation to the input stereo images.

  Parameters:
  left_image (numpy.ndarray): Left input image.
  right_image (numpy.ndarray): Right input image.
  num_disparities (int): Number of disparities.
  block_size (int): Block size.

  Returns:
  numpy.ndarray: Disparity map.
  """
  stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
  disparity = stereo.compute(left_image, right_image)
  return disparity
def apply_depth_map(disparity_map, focal_length=1, baseline=1):
  """
  Apply depth map calculation to the input disparity map.

  Parameters:
  disparity_map (numpy.ndarray): Disparity map.
  focal_length (float): Focal length of the camera.
  baseline (float): Distance between the two cameras.

  Returns:
  numpy.ndarray: Depth map.
  """
  depth_map = np.zeros_like(disparity_map, dtype=np.float32)
  depth_map[disparity_map > 0] = focal_length * baseline / disparity_map[disparity_map > 0]
  return depth_map
def apply_epipolar_geometry(left_image, right_image, fundamental_matrix):
  """
  Apply epipolar geometry to the input stereo images.

  Parameters:
  left_image (numpy.ndarray): Left input image.
  right_image (numpy.ndarray): Right input image.
  fundamental_matrix (numpy.ndarray): Fundamental matrix.

  Returns:
  numpy.ndarray: Image with epipolar lines.
  """
  lines1 = cv2.computeCorrespondEpilines(np.array([[0, 0]]), 1, fundamental_matrix)
  lines1 = lines1.reshape(-1, 3)
  epilines1 = cv2.computeCorrespondEpilines(np.array([[0, 0]]), 2, fundamental_matrix)
  epilines1 = epilines1.reshape(-1, 3)
  epipolar_image = cv2.cvtColor(left_image, cv2.COLOR_GRAY2BGR)
  for r, pt1, pt2 in zip(lines1, epilines1):
    color = tuple(np.random.randint(0, 255, 3).tolist())
    x0, y0 = map(int, [0, -r[2] / r[1]])
    x1, y1 = map(int, [left_image.shape[1], -(r[2] + r[0] * left_image.shape[1]) / r[1]])
    epipolar_image = cv2.line(epipolar_image, (x0, y0), (x1, y1), color, 1)
  return
def apply_fundamental_matrix(left_points, right_points):
  """
  Apply fundamental matrix calculation to the input points.

  Parameters:
  left_points (numpy.ndarray): Left points.
  right_points (numpy.ndarray): Right points.

  Returns:
  numpy.ndarray: Fundamental matrix.
  """
  fundamental_matrix, _ = cv2.findFundamentalMat(left_points, right_points, cv2.FM_LMEDS)
  return fundamental_matrix
def apply_homography_matrix(src_points, dst_points):
  """
  Apply homography matrix calculation to the input points.

  Parameters:
  src_points (numpy.ndarray): Source points.
  dst_points (numpy.ndarray): Destination points.

  Returns:
  numpy.ndarray: Homography matrix.
  """
  homography_matrix, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
  return
def apply_camera_calibration(object_points, image_points, image_size):
  """
  Apply camera calibration to the input object and image points.

  Parameters:
  object_points (numpy.ndarray): Object points.
  image_points (numpy.ndarray): Image points.
  image_size (tuple): Image size.

  Returns:
  tuple: Camera matrix, distortion coefficients, rotation vectors, translation vectors.
  """
  camera_matrix = np.zeros((3, 3))
  dist_coeffs = np.zeros((1, 5))
  _, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(object_points, image_points, image_size, camera_matrix, dist_coeffs)
  return camera_matrix, dist
def apply_perspective_transformation(image, src_points, dst_points):
  """
  Apply perspective transformation to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  src_points (numpy.ndarray): Source points.
  dst_points (numpy.ndarray): Destination points.

  Returns:
  numpy.ndarray: Image after perspective transformation.
  """
  height, width = image.shape[:2]
  perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
  perspective_image = cv2.warpPerspective(image, perspective_matrix, (width, height))
  return perspective_image
def apply_affine_transformation(image, src_points, dst_points):
  """
  Apply affine transformation to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  src_points (numpy.ndarray): Source points.
  dst_points (numpy.ndarray): Destination points.

  Returns:
  numpy.ndarray: Image after affine transformation.
  """
  height, width = image.shape[:2]
  affine_matrix = cv2.getAffineTransform(src_points, dst_points)
  affine_image = cv2.warpAffine(image, affine_matrix, (width, height))
  return affine_image
def apply_rotation(image, angle=90):
  """
  Apply rotation to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  angle (float): Angle of rotation.

  Returns:
  numpy.ndarray: Image after rotation.
  """
  height, width = image.shape[:2]
  rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
  rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
  return rotated_image
def apply_translation(image, dx=100, dy=100):
  """
  Apply translation to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  dx (int): Translation in the x-direction.
  dy (int): Translation in the y-direction.

  Returns:
  numpy.ndarray: Image after translation.
  """
  height, width = image.shape[:2]
  translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
  translated_image = cv2.warpAffine(image, translation_matrix, (width, height))
  return translated_image
def apply_scaling(image, fx=2, fy=2):
  """
  Apply scaling to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  fx (float): Scale factor along the x-axis.
  fy (float): Scale factor along the y-axis.

  Returns:
  numpy.ndarray: Image after scaling.
  """
  scaled_image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
  return scaled_image
def apply_pyr_down(image):
  """
  Apply pyrDown to the input image.

  Parameters:
  image (numpy.ndarray): Input image.

  Returns:
  numpy.ndarray: Image after pyrDown.
  """
  pyr_down_image = cv2.pyrDown(image)
  return pyr_down_image
def apply_pyr_up(image):
  """
  Apply pyrUp to the input image.

  Parameters:
  image (numpy.ndarray): Input image.

  Returns:
  numpy.ndarray: Image after pyrUp.
  """
  pyr_up_image = cv2.pyrUp(image)
  return pyr_up_image

def extract_features(image):
  """
  Extract features from the input image using SIFT.

  Parameters:
  image (numpy.ndarray): Input image.

  Returns:
  keypoints, descriptors: Keypoints and descriptors.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  sift = cv2.SIFT_create()
  keypoints, descriptors = sift.detectAndCompute(gray_image, None)
  return keypoints, descriptors
def track_features(image1, image2, keypoints1, descriptors1):
  """
  Track features between two images using FLANN-based matcher.

  Parameters:
  image1 (numpy.ndarray): First input image.
  image2 (numpy.ndarray): Second input image.
  keypoints1: Keypoints from the first image.
  descriptors1: Descriptors from the first image.

  Returns:
  keypoints2, matches: Keypoints from the second image and matches.
  """
  sift = cv2.SIFT_create()
  keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
  index_params = dict(algorithm=1, trees=5)
  search_params = dict(checks=50)
  flann = cv2.FlannBasedMatcher(index_params, search_params)
  matches = flann.knnMatch(descriptors1, descriptors2, k=2)
  good_matches = []
  for m, n in matches:
    if m.distance < 0.7 * n.distance:
      good_matches.append(m)
  return keypoints2, good_matches
def triangulate_points(proj_matrix1, proj_matrix2, points1, points2):
  """
  Triangulate points between two images.

  Parameters:
  proj_matrix1 (numpy.ndarray): Projection matrix of the first image.
  proj_matrix2 (numpy.ndarray): Projection matrix of the second image.
  points1 (numpy.ndarray): Points from the first image.
  points2 (numpy.ndarray): Points from the second image.

  Returns:
  numpy.ndarray: Triangulated 3D points.
  """
  points4D = cv2.triangulatePoints(proj_matrix1, proj_matrix2, points1.T, points2.T)
  points3D = points4D[:3] / points4D[3]
  return points3D.T
def bundle_adjustment(points3D, keypoints1, keypoints2, proj_matrix1, proj_matrix2):
  """
  Perform bundle adjustment to refine 3D points and camera parameters.

  Parameters:
  points3D (numpy.ndarray): Initial 3D points.
  keypoints1: Keypoints from the first image.
  keypoints2: Keypoints from the second image.
  proj_matrix1 (numpy.ndarray): Projection matrix of the first image.
  proj_matrix2 (numpy.ndarray): Projection matrix of the second image.

  Returns:
  refined_points3D, refined_proj_matrix1, refined_proj_matrix2: Refined 3D points and projection matrices.
  """
  # Placeholder for bundle adjustment implementation
  # This typically involves optimization techniques and is complex to implement from scratch
  return points3D, proj_matrix1, proj_matrix2
def estimate_camera_pose(image1, image2):
  """
  Estimate the camera pose between two images.

  Parameters:
  image1 (numpy.ndarray): First input image.
  image2 (numpy.ndarray): Second input image.

  Returns:
  numpy.ndarray: Rotation and translation vectors.
  """
  keypoints1, descriptors1 = extract_features(image1)
  keypoints2, descriptors2 = extract_features(image2)
  keypoints1, matches = track_features(image1, image2, keypoints1, descriptors1)
  points1 = np.array([keypoints1[m.queryIdx].pt for m in matches])
  points2 = np.array([keypoints2[m.trainIdx].pt for m in matches])
  camera_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
  _, rvec, tvec, _ = cv2.solvePnPRansac(points1, points2, camera_matrix, None)
  return rvec, tvec
def estimate_camera_matrix(image, focal_length, principal_point):
  """
  Estimate the camera matrix from the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  focal_length (float): Focal length of the camera.
  principal_point (tuple): Principal point (x, y) of the camera.

  Returns:
  numpy.ndarray: Camera matrix.
  """
  height, width = image.shape[:2]
  camera_matrix = np.array([[focal_length, 0, principal_point[0]], [0, focal_length, principal_point[1]], [0, 0, 1]])
  return camera_matrix
def estimate_projection_matrix(camera_matrix, rvec, tvec):
  """
  Estimate the projection matrix from the camera matrix, rotation, and translation vectors.

  Parameters:
  camera_matrix (numpy.ndarray): Camera matrix.
  rvec (numpy.ndarray): Rotation vector.
  tvec (numpy.ndarray): Translation vector.

  Returns:
  numpy.ndarray: Projection matrix.
  """
  rotation_matrix, _ = cv2.Rodrigues(rvec)
  projection_matrix = np.hstack((rotation_matrix, tvec))
  projection_matrix = np.dot(camera_matrix, projection_matrix)
  return projection_matrix
def reconstruct_3D_points(image1, image2, focal_length, principal_point):
  """
  Reconstruct 3D points between two images.

  Parameters:
  image1 (numpy.ndarray): First input image.
  image2 (numpy.ndarray): Second input image.
  focal_length (float): Focal length of the camera.
  principal_point (tuple): Principal point (x, y) of the camera.

  Returns:
  numpy.ndarray: 3D points.
  """
  rvec, tvec = estimate_camera_pose(image1, image2)
  camera_matrix = estimate_camera_matrix(image1, focal_length, principal_point)
  projection_matrix1 = estimate_projection_matrix(camera_matrix, np.zeros(3), np.zeros(3))
  projection_matrix2 = estimate_projection_matrix(camera_matrix, rvec, tvec)
  keypoints1, descriptors1 = extract_features(image1)
  keypoints2, descriptors2 = extract_features(image2)
  keypoints1, matches = track_features(image1, image2, keypoints1, descriptors1)
  points1 = np.array([keypoints1[m.queryIdx].pt for m in matches])
  points2 = np.array([keypoints2[m.trainIdx].pt for m in matches])
  points3D = triangulate_points(projection_matrix1, projection_matrix2, points1, points2)
  return points3D
def refine_camera_pose(points3D, keypoints1, keypoints2, camera_matrix, rvec, tvec):
  """
  Refine the camera pose using bundle adjustment.

  Parameters:
  points3D (numpy.ndarray): 3D points.
  keypoints1: Keypoints from the first image.
  keypoints2: Keypoints from the second image.
  camera_matrix (numpy.ndarray): Camera matrix.
  rvec (numpy.ndarray): Rotation vector.
  tvec (numpy.ndarray): Translation vector.

  Returns:
  numpy.ndarray: Refined rotation and translation vectors.
  """
  projection_matrix1 = estimate_projection_matrix(camera_matrix, np.zeros(3), np.zeros(3))
  projection_matrix2 = estimate_projection_matrix(camera_matrix, rvec, tvec)
  points1 = np.array([keypoints1[m.queryIdx].pt for m in matches])
  points2 = np.array([keypoints2[m.trainIdx].pt for m in matches])
  points3D = triangulate_points(projection_matrix1, projection_matrix2, points1, points2)
  refined_points3D, refined_projection_matrix1, refined_projection_matrix2 = bundle_adjustment(points3D, keypoints1, keypoints2, projection_matrix1, projection_matrix2)
  rvec, tvec = cv2.solvePnP(refined_points3D, points2, camera_matrix, None)
  return rvec, tvec


"""Stereo Vision"""

def apply_stereo_block_matching(left_image, right_image, num_disparities=16, block_size=15):
  """
  Apply block matching to calculate the disparity map.

  Parameters:
  left_image (numpy.ndarray): Left input image.
  right_image (numpy.ndarray): Right input image.
  num_disparities (int): Number of disparities.
  block_size (int): Block size.

  Returns:
  numpy.ndarray: Disparity map.
  """
  stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
  disparity = stereo.compute(left_image, right_image)
  return disparity
def apply_stereo_sgbm(left_image, right_image, num_disparities=16, block_size=15):
  """
  Apply semi-global block matching to calculate the disparity map.

  Parameters:
  left_image (numpy.ndarray): Left input image.
  right_image (numpy.ndarray): Right input image.
  num_disparities (int): Number of disparities.
  block_size (int): Block size.

  Returns:
  numpy.ndarray: Disparity map.
  """
  stereo = cv2.StereoSGBM_create(numDisparities=num_disparities, blockSize=block_size)
  disparity = stereo.compute(left_image, right_image)
  return disparity
def apply_stereo_hh(left_image, right_image):
  """
  Apply H. Hirschmuller algorithm to calculate the disparity map.

  Parameters:
  left_image (numpy.ndarray): Left input image.
  right_image (numpy.ndarray): Right input image.

  Returns:
  numpy.ndarray: Disparity map.
  """
  stereo = cv2.StereoSGBM_create(minDisparity=16, numDisparities=32, blockSize=15)
  disparity = stereo.compute(left_image, right_image)
  return disparity
def apply_stereo_var(left_image, right_image):
  """
  Apply Variational Refinement algorithm to calculate the disparity map.

  Parameters:
  left_image (numpy.ndarray): Left input image.
  right_image (numpy.ndarray): Right input image.

  Returns:
  numpy.ndarray: Disparity map.
  """
  stereo = cv2.StereoVar_create()
  disparity = stereo.compute(left_image, right_image)
  return disparity
def apply_stereo_bm(left_image, right_image):
  """
  Apply Block Matching algorithm to calculate the disparity map.

  Parameters:
  left_image (numpy.ndarray): Left input image.
  right_image (numpy.ndarray): Right input image.

  Returns:
  numpy.ndarray: Disparity map.
  """
  stereo = cv2.StereoBM_create()
  disparity = stereo.compute(left_image, right_image)
  return disparity
def apply_stereo_elas(left_image, right_image):
  """
  Apply Efficient Large-Scale Stereo algorithm to calculate the disparity map.

  Parameters:
  left_image (numpy.ndarray): Left input image.
  right_image (numpy.ndarray): Right input image.

  Returns:
  numpy.ndarray: Disparity map.
  """
  stereo = cv2.ELAS_create()
  disparity = stereo.compute(left_image, right_image)
  return disparity
def apply_stereo_bm(left_image, right_image):
  """
  Apply Block Matching algorithm to calculate the disparity map.

  Parameters:
  left_image (numpy.ndarray): Left input image.
  right_image (numpy.ndarray): Right input image.

  Returns:
  numpy.ndarray: Disparity map.
  """
  stereo = cv2.StereoBM_create()
  disparity = stereo.compute(left_image, right_image)
  return disparity
def apply_stereo_elas(left_image, right_image):
  """
  Apply Efficient Large-Scale Stereo algorithm to calculate the disparity map.

  Parameters:
  left_image (numpy.ndarray): Left input image.
  right_image (numpy.ndarray): Right input image.

  Returns:
  numpy.ndarray: Disparity map.
  """
  stereo = cv2.ELAS_create()
  disparity = stereo.compute(left_image, right_image)
  return disparity
def apply_stereo_var(left_image, right_image):
  """
  Apply Variational Refinement algorithm to calculate the disparity map.

  Parameters:
  left_image (numpy.ndarray): Left input image.
  right_image (numpy.ndarray): Right input image.

  Returns:
  numpy.ndarray: Disparity map.
  """
  stereo = cv2.StereoVar_create()
  disparity = stereo.compute(left_image, right_image)
  return disparity
def apply_stereo_hh(left_image, right_image):
  """
  Apply H. Hirschmuller algorithm to calculate the disparity map.

  Parameters:
  left_image (numpy.ndarray): Left input image.
  right_image (numpy.ndarray): Right input image.

  Returns:
  numpy.ndarray: Disparity map.
  """
  stereo = cv2.StereoSGBM_create(minDisparity=16, numDisparities=32, blockSize=15)
  disparity = stereo.compute(left_image, right_image)
  return
def apply_stereo_sgbm(left_image, right_image, num_disparities=16, block_size=15):
  """
  Apply semi-global block matching to calculate the disparity map.

  Parameters:
  left_image (numpy.ndarray): Left input image.
  right_image (numpy.ndarray): Right input image.
  num_disparities (int): Number of disparities.
  block_size (int): Block size.

  Returns:
  numpy.ndarray: Disparity map.
  """
  stereo = cv2.StereoSGBM_create(numDisparities=num_disparities, blockSize=block_size)
  disparity = stereo.compute(left_image, right_image)
  return disparity
def apply_stereo_block_matching(left_image, right_image, num_disparities=16, block_size=15):
  """
  Apply block matching to calculate the disparity map.

  Parameters:
  left_image (numpy.ndarray): Left input image.
  right_image (numpy.ndarray): Right input image.
  num_disparities (int): Number of disparities.
  block_size (int): Block size.

  Returns:
  numpy.ndarray: Disparity map.
  """
  stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
  disparity = stereo.compute(left_image, right_image)
  return disparity



def apply_hough_lines(image, rho=1, theta=np.pi/180, threshold=100):
  """
  Apply Hough line detection to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  rho (int): Distance resolution of the accumulator in pixels.
  theta (float): Angle resolution of the accumulator in radians.
  threshold (int): Accumulator threshold parameter.

  Returns:
  numpy.ndarray: Image with Hough lines applied.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
  lines = cv2.HoughLines(edges, rho, theta, threshold)
  hough_image = image.copy()
  if lines is not None:
    for line in lines:
      rho, theta = line[0]
      a = np.cos(theta)
      b = np.sin(theta)
      x0 = a * rho
      y0 = b * rho
      x1 = int(x0 + 1000 * (-b))
      y1 = int(y0 + 1000 * (a))
      x2 = int(x0 - 1000 * (-b))
      y2 = int(y0 - 1000 * (a))
      cv2.line(hough_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
  return hough_image
def apply_hough_circles(image, method=cv2.HOUGH_GRADIENT, dp=1, min_dist=20, param1=50, param2=30, min_radius=0, max_radius=0):
  """
  Apply Hough circle detection to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  method (int): Detection method.
  dp (float): Inverse ratio of the accumulator resolution to the image resolution.
  min_dist (int): Minimum distance between the centers of the detected circles.
  param1 (int): First method-specific parameter.
  param2 (int): Second method-specific parameter.
  min_radius (int): Minimum circle radius.
  max_radius (int): Maximum circle radius.

  Returns:
  numpy.ndarray: Image with Hough circles applied.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  circles = cv2.HoughCircles(gray_image, method, dp, min_dist, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
  hough_image = image.copy()
  if circles is not None:
    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
      center = (circle[0], circle[1])
      radius = circle[2]
      cv2.circle(hough_image, center, radius, (0, 255, 0), 2)
  return hough_image
def apply_template_matching(image, template, method=cv2.TM_CCOEFF_NORMED):
  """
  Apply template matching to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  template (numpy.ndarray): Template image.
  method (int): Template matching method.

  Returns:
  numpy.ndarray: Image with template matching applied.
  """
  result = cv2.matchTemplate(image, template, method)
  min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
  top_left = max_loc
  h, w = template.shape[:2]
  bottom_right = (top_left[0] + w, top_left[1] + h)
  cv2.rectangle(image, top_left, bottom_right, 255, 2)
  return image
def apply_face_detection(image, cascade_path='haarcascade_frontalface_default.xml'):
  """
  Apply face detection to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  cascade_path (str): Path to the Haar cascade XML file.

  Returns:
  numpy.ndarray: Image with face detection applied.
  """
  face_cascade = cv2.CascadeClassifier(cascade_path)
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
  face_image = image.copy()
  for (x, y, w, h) in faces:
    cv2.rectangle(face_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
  return face_image
def apply_eye_detection(image, cascade_path='haarcascade_eye.xml'):
  """
  Apply eye detection to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  cascade_path (str): Path to the Haar cascade XML file.

  Returns:
  numpy.ndarray: Image with eye detection applied.
  """
  eye_cascade = cv2.CascadeClassifier(cascade_path)
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  eyes = eye_cascade.detectMultiScale(gray_image)
  eye_image = image.copy()
  for (x, y, w, h) in eyes:
    cv2.rectangle(eye_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
  return eye_image
def apply_smile_detection(image, cascade_path='haarcascade_smile.xml'):
  """
  Apply smile detection to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  cascade_path (str): Path to the Haar cascade XML file.

  Returns:
  numpy.ndarray: Image with smile detection applied.
  """
  smile_cascade = cv2.CascadeClassifier(cascade_path)
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  smiles = smile_cascade.detectMultiScale(gray_image, scaleFactor=1.8, minNeighbors=20, minSize=(30, 30))
  smile_image = image.copy()
  for (x, y, w, h) in smiles:
    cv2.rectangle(smile_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
  return smile_image
def apply_object_detection(image, config_path='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt', weights_path='frozen_inference_graph.pb'):
  """
  Apply object detection to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  config_path (str): Path to the configuration file.
  weights_path (str): Path to the weights file.

  Returns:
  numpy.ndarray: Image with object detection applied.
  """
  net = cv2.dnn_DetectionModel(weights_path, config_path)
  classes, _ = net.readNet('coco.names')
  class_ids, confidences, boxes = net.detect(image, confThreshold=0.5)
  object_image = image.copy()
  if len(class_ids) > 0:
    for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), boxes):
      cv2.rectangle(object_image, box, color=(0, 255, 0), thickness=2)
      cv2.putText(object_image, classes[class_id - 1], (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
  return object_image
def apply_style_transfer(image, style_path='starry_night.jpg'):
  """
  Apply style transfer to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  style_path (str): Path to the style image.

  Returns:
  numpy.ndarray: Image with style transfer applied.
  """
  style_image = cv2.imread(style_path)
  style_image = cv2.resize(style_image, (image.shape[1], image.shape[0]))
  style_transfer = cv2.stylization(image, style_image, sigma_s=60, sigma_r=0.07)
  return style_transfer
def apply_super_resolution(image, model_path='ESPCN_x4.pb'):
  """
  Apply super resolution to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  model_path (str): Path to the super resolution model.

  Returns:
  numpy.ndarray: Image with super resolution applied.
  """
  sr = cv2.dnn_superres.DnnSuperResImpl_create()
  sr.readModel(model_path)
  sr.setModel('espcn', 4)
  super_resolution = sr.upsample(image)
  return super_resolution
def apply_deep_dream(image, model_path='deepdream.pb'):
  """
  Apply DeepDream to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  model_path (str): Path to the DeepDream model.

  Returns:
  numpy.ndarray: Image with DeepDream applied.
  """
  deep_dream = cv2.dnn.readNetFromTensorflow(model_path)
  blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(224, 224), mean=(0, 0, 0), swapRB=False, crop=False)
  deep_dream.setInput(blob)
  output = deep_dream.forward()
  return output
def apply_neural_style_transfer(image, model_path='wave.pb'):
  """
  Apply neural style transfer to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  model_path (str): Path to the neural style transfer model.

  Returns:
  numpy.ndarray: Image with neural style transfer applied.
  """
  neural_style_transfer = cv2.dnn.readNetFromTensorflow(model_path)
  blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(224, 224), 
  mean=(0, 0, 0), swapRB=False, crop=False)
  neural_style_transfer.setInput(blob)
  output = neural_style_transfer.forward()
  return output
def apply_colorization(image, model_path='colorization_deploy_v2.prototxt', 
weights_path='colorization_release_v2.caffemodel'):
  """
  Apply colorization to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  model_path (str): Path to the colorization model.
  weights_path (str): Path to the colorization weights.

  Returns:
  numpy.ndarray: Image with colorization applied.
  """
  prototxt = cv2.dnn.readNetFromCaffe(model_path, weights_path)
  prototxt.setInput(cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(224, 224), mean=(103.939, 116.779, 123.68), swapRB=False, crop=False))
  output = prototxt.forward()
  return output
def apply_inpainting(image, mask, method='telea'):
  """
  Apply inpainting to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  mask (numpy.ndarray): Mask for inpainting.
  method (str): Inpainting method.

  Returns:
  numpy.ndarray: Image with inpainting applied.
  """
  if method == 'telea':
    inpainted_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
  elif method == 'ns':
    inpainted_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
  return inpainted_image
def apply_image_segmentation(image, method='watershed'):
  """
  Apply image segmentation to the input image.

  Parameters:
  image (numpy.ndarray): Input image.
  method (str): Segmentation method.

  Returns:
  numpy.ndarray: Image with segmentation applied.
  """
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  if method == 'watershed':
    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]
  elif method == 'grabcut':
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (50, 50, image.shape[1] - 50, image.shape[0] - 50)
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    image = image * mask2[:, :, np.newaxis]
  return image



"""Utility Functions"""
def load_json_file(file_path):
    print(f"Loading JSON file from: {file_path}")
    with open(file_path) as f:
        data = json.load(f)
    print("JSON file loaded successfully")
    return data
"""Pipeline Functions"""
def process_data(data, image):
    for phase in data['phases']:
        print(f"------ Processing {phase['phase']} ------")
        for func_name in phase['func']:
            func = globals().get(func_name)
            if func:
                image = func(image)
                if image is None or image.size == 0:
                    print(f"Error: {func_name} returned an invalid image.")
                    return None
            else:
                print(f"Function {func_name} not found")
    return image
"""Display Functions"""
def display_image(image, title='Image'):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()
"""Main Function"""
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 pipeline.py <path_to_json_file>")
        return
    
    file_path = sys.argv[1]
    data = load_json_file(file_path)
    
    # Load your image here
    image = cv2.imread('circle.jpg')
    
    if image is None:
        print("Error: Image not found or unable to load.")
        return
    
    processed_image = process_data(data, image)
    
    if processed_image is None:
        print("Error: Processing failed.")
        return
    
    # Save or display the processed image
    cv2.imwrite('processed_image.jpg', processed_image)
    display_image(processed_image, 'Processed Image')
if __name__ == "__main__":
    main()