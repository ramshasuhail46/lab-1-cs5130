import cv2 as cv
import numpy as np

def calculate_similarity(original, mosaic):
    """
    Compares the original image and the generated mosaic using 
    Mean Squared Error (MSE) and Structural Similarity Index (SSIM).
    Accepts either image file paths (strings) or NumPy image arrays.
    """
    # 1. Load images from file paths if strings were passed instead of arrays
    if isinstance(original, (str, np.bytes_)):
        original = cv.imread(original)
    if isinstance(mosaic, (str, np.bytes_)):
        mosaic = cv.imread(mosaic)
        
    # Ensure both images loaded successfully
    if original is None or mosaic is None:
        raise ValueError("Could not process images. Ensure valid images or paths are provided.")
    
    # 2. Make sure both images have the exact same dimensions before comparing
    if original.shape != mosaic.shape:
        print("Resizing mosaic to match original dimensions for accurate comparison...")
        mosaic = cv.resize(mosaic, (original.shape[1], original.shape[0]))
    
    # 3. Calculate Mean Squared Error (MSE)
    # Measures average pixel difference (lower is better; 0 means identical)
    mse = np.mean((original.astype("float") - mosaic.astype("float")) ** 2)

    # 4. Calculate Structural Similarity Index (SSIM)
    # Convert images to grayscale and cast to float for precision
    img1 = cv.cvtColor(original, cv.COLOR_BGR2GRAY).astype(np.float64)
    img2 = cv.cvtColor(mosaic, cv.COLOR_BGR2GRAY).astype(np.float64)
    
    # Constants for stability based on standard SSIM formula (C1 and C2)
    L = 255.0
    k1 = 0.01
    k2 = 0.03
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    
    # Compute means using global averages (or you can use uniform sliding window filters for local SSIM)
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    # Compute variances and covariance
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.cov(img1.flatten(), img2.flatten())[0, 1]
    
    # SSIM formula calculation
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_score = numerator / denominator
    
    # Print clean results to the console
    print("\n--- Image Similarity Metrics ---")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Structural Similarity (SSIM): {ssim_score:.4f}")
    
    return mse, ssim_score