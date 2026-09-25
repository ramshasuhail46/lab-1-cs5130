import numpy as np
import cv2 as cv
from config import (
    TARGET_SIZE,
    K_COLORS,
    KMEANS_MAX_ITER,
    KMEANS_EPSILON,
    KMEANS_ATTEMPTS,
)

def quantization(img):
    """
    Applies K-Means color quantization to reduce the number of unique colors 
    in the image down to K_COLORS, simplifying the color palette for mosaic matching.
    """
    
    # Reshape the 3D image matrix (Height, Width, Channels) into a 2D array of pixels: (Total Pixels, 3)
    quantized_image = img.reshape((-1,3)) 

    # Convert pixel data to float32, as OpenCV's kmeans function strictly requires floating-point input
    quantized_image = np.float32(quantized_image) 

    # Define clustering termination criteria: combination of maximum iterations and desired epsilon (accuracy)
    criteria = (
        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
        KMEANS_MAX_ITER,
        KMEANS_EPSILON,
    )

    # Execute K-Means clustering to find K_COLORS dominant color centers
    ret, label, center = cv.kmeans(
        quantized_image,
        K_COLORS,
        None,
        criteria,
        KMEANS_ATTEMPTS,
        cv.KMEANS_PP_CENTERS, # Use K-Means++ center initialization for better convergence
    )   

    # Convert cluster centers back from float back to standard 8-bit unsigned integers (uint8: 0-255)
    center = np.uint8(center)

    # Map every pixel to its corresponding cluster center color using the flattened labels array
    res = center[label.flatten()]

    # Reshape the flat pixel array back into the original 3D image dimensions
    res2 = res.reshape((img.shape))

    return res2

def resize(img):
    """
    Resizes the input image to a uniform square resolution defined by TARGET_SIZE.
    """
    target_resolution = (TARGET_SIZE, TARGET_SIZE)

    # Use INTER_AREA interpolation which is optimal for image downsampling/shrinking
    resized_image = cv.resize(img, target_resolution, interpolation=cv.INTER_AREA)

    return resized_image

def crop(img):
    """
    Crops a rectangular image into a square (1:1 aspect ratio) by 
    symmetrically trimming pixels from the center.
    """
    # Extract height, width, and number of color channels from the image shape
    height, width, channels = np.shape(img)
    # Find the smaller dimension to determine the maximum square size possible
    min_dim = np.min([height, width])

    # Calculate starting coordinates to center the crop window
    start_y = (height - min_dim) // 2 # 200
    start_x = (width - min_dim) // 2 # 0

    # Slice the image array to extract the centered square region
    cropped_image = img[start_y:start_y + min_dim, start_x:start_x + min_dim]
    return cropped_image


def preprocess(img):
    """
    Executes the full image preprocessing pipeline:
    1. Center-crops the image to a square aspect ratio.
    2. Resizes the image to the standardized TARGET_SIZE.
    3. Performs K-Means color quantization to establish dominant color profiles.
    """

    # Step 1: Crop to a 1:1 square
    preprocessed_image = crop(img)

    # Step 2: Resize to target resolution
    preprocessed_image = resize(preprocessed_image)

    # Step 3: Reduce color complexity via K-Means quantization
    preprocessed_image = quantization(preprocessed_image)

    return preprocessed_image

