import numpy as np
import cv2 as cv
import os
from config import GRID_SIZE, TILE_SIZE
from config import (
    TILE_FOLDER,
    IMAGE_EXTENSIONS,
    GRID_SIZE,
    TILE_SIZE,
)

def image_grid(img):
    """
    Splits the image into a grid of blocks and calculates the average BGR color for each cell.
    """

    grid_size = GRID_SIZE
    tile_size = TILE_SIZE

    # Reshape the image into a 5D block tensor: (rows, tile_height, cols, tile_width, channels)
    blocks = img.reshape(grid_size, tile_size, grid_size, tile_size, 3)

    # Compute the average color for each grid cell across the tile height and width axes
    cell_average = blocks.mean(axis=(1,3))

    return cell_average

def image_colour_map_display(preprocessed_img, cell_average):
    """
    Quantizes and maps grid cell colors to the dominant K-Means palette colors.
    """
    
    # 1. Extract unique cluster colors from the preprocessed image
    palette = np.unique(preprocessed_img.reshape(-1, 3), axis=0)
    print(f"Extracted palette shape: {palette.shape}")

    # 2. Compute Euclidean distance between every cell and all 8 palette colors vectorially
    # cell_average: (32, 32, 3) -> expand to (32, 32, 1, 3)
    # palette: (8, 3) -> expand to (1, 1, 8, 3)
    differences = cell_average[:, :, np.newaxis, :] - palette[np.newaxis, np.newaxis, :, :]
    distances = np.sum(differences ** 2, axis=-1)  # Shape: (32, 32, 8)

    # 3. Find the index of the closest palette color for every cell
    color_categories = np.argmin(distances, axis=-1)  # Shape: (32, 32)

    # 4. Map the indices back to their actual BGR palette colors
    color_mapped_img = palette[color_categories]

    return color_mapped_img

def tile_mapping(processed_image, cell_average, preprocessed_img):
    """
    Loads custom tiles, computes their average colors using their center regions, 
    and replaces each grid cell with the closest matching tile image.
    """

    tile_images = []

    # Load all valid tile images from the tile folder
    if os.path.exists(TILE_FOLDER):
        for filename in os.listdir(TILE_FOLDER):
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                tile_path = os.path.join(TILE_FOLDER, filename)
                tile_img = cv.imread(tile_path)
                if tile_img is not None:
                    # Resize tile to match the required grid cell size
                    tile_img = cv.resize(tile_img, (TILE_SIZE, TILE_SIZE), interpolation=cv.INTER_AREA)
                    tile_images.append(tile_img)

    tile_images = np.array(tile_images)  # Shape: (num_tiles, height, width, 3)


    # 1. Calculate average color using ONLY the center 50% of each tile (to avoid border noise)
    h, w = TILE_SIZE, TILE_SIZE
    center_crops = tile_images[:, h//4 : 3*h//4, w//4 : 3*w//4, :]
    tile_averages = center_crops.mean(axis=(1, 2)).astype(np.uint8) # Shape: (num_tiles, 3)

    # 2. Convert grid cell averages to LAB color space for better human perceptual color matching
    cell_avg_uint8 = np.clip(cell_average, 0, 255).astype(np.uint8)
    cell_avg_lab = cv.cvtColor(cell_avg_uint8, cv.COLOR_BGR2LAB).astype(np.float32)

    # 3. Convert tile averages to LAB color space as well
    tile_avg_lab = cv.cvtColor(tile_averages.reshape(1, -1, 3), cv.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)

    # 4. Compare every grid cell to every tile in LAB space using Euclidean distance
    differences_tiles = cell_avg_lab[:, :, np.newaxis, :] - tile_avg_lab[np.newaxis, np.newaxis, :, :]
    distances_tiles = np.sum(differences_tiles ** 2, axis=-1)
    best_tile_indices = np.argmin(distances_tiles, axis=-1)

    # 5. Create a blank canvas and paste the best-matching tiles onto their grid locations
    canvas = np.zeros_like(preprocessed_img)

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            tile_idx = best_tile_indices[i, j]
            chosen_tile = tile_images[tile_idx]
            
            y_start = i * TILE_SIZE
            x_start = j * TILE_SIZE
            
            canvas[y_start:y_start + TILE_SIZE, x_start:x_start + TILE_SIZE] = chosen_tile

    return canvas

def engine(preprocessed_img):
    """
    Main orchestration function: splits image into a grid, maps color categories, 
    and builds the final tiled mosaic canvas.
    """

    # Step 1: Calculate average color per grid cell
    cell_average = image_grid(preprocessed_img)

    # Step 2: Map grid cells to dominant color categories
    processed_image = image_colour_map_display(preprocessed_img, cell_average, )

    # Step 3: Replace grid blocks with matching physical image tiles
    processed_image = tile_mapping(processed_image, cell_average, preprocessed_img)

    return processed_image