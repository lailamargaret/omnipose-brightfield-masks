import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import pandas as pd
from tifffile import imread, imsave
import shutil
import matplotlib.pyplot as plt
import re
from concurrent.futures import ProcessPoolExecutor
import cv2
from math import sqrt
import seaborn as sns
from matplotlib.colors import SymLogNorm
from math import ceil
from natsort import natsorted
from scipy import ndimage
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import ceil, sqrt
from matplotlib.colors import LogNorm
import os
from skimage import io
from matplotlib import colors
import concurrent.futures
from matplotlib.colors import ListedColormap

def extract_metrics(mask_filepath, area, total_well_area, filter_min_size=None):
    if filter_min_size:
        #Filter masks by size (get rid of weird artifacts)
        unfiltered_dir = os.path.join(os.path.dirname(mask_filepath), "unfiltered_masks")
        os.makedirs(unfiltered_dir, exist_ok=True)

        unfiltered_mask_path = os.path.join(unfiltered_dir, os.path.basename(mask_filepath))

        if not os.path.exists(unfiltered_mask_path):
            shutil.move(mask_filepath, unfiltered_mask_path)
            if ".png" in unfiltered_mask_path:
                unfiltered_masks = cv2.imread(str(unfiltered_mask_path), cv2.IMREAD_UNCHANGED)
            else:
                unfiltered_masks = imread(unfiltered_mask_path)

            labels = np.unique(unfiltered_masks)
            labels = labels[labels != 0]
            areas = ndimage.sum(np.ones_like(unfiltered_masks), unfiltered_masks, index=labels[labels != 0])
            large_mask_indices = areas >= filter_min_size
            large_labels = labels[large_mask_indices]
            masks = np.zeros_like(unfiltered_masks)
            for label in large_labels:
                masks[unfiltered_masks == label] = label

            imsave(mask_filepath, masks)
    else:
        if ".png" in mask_filepath:
            masks = cv2.imread(str(mask_filepath), cv2.IMREAD_UNCHANGED)
        else:
            masks = imread(mask_filepath)

    well, row, column, pos = extract_well_info(mask_filepath)

    # Count masks
    labels = np.unique(masks)
    mask_count = len(labels[labels != 0])

    results = {
        'file_path' : mask_filepath,
        'file': os.path.basename(mask_filepath),
        'p1': os.path.basename(os.path.dirname(mask_filepath)),
        'p2': os.path.basename(os.path.dirname(os.path.dirname(mask_filepath))),
        'p3': os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(mask_filepath)))),
        'predicted_count': mask_count,
        'cell_density': mask_count / area,
        'total_cells_in_well': mask_count / area * total_well_area,
        'predicted_confluency': 1 - (np.sum(masks == 0) / masks.size),
        'Well' : well,
        'Row' : row,
        "Column" : column,
        'Position' : pos
    }
    return results

def extract_well_info(file_path):
    match = re.match(r"([A-Za-z]+)(\d+)_pos(\d+)", os.path.basename(file_path))
    if match:
        well = match.group(1) + match.group(2)
        row = match.group(1)
        column = match.group(2)
        pos = match.group(3)
        return well, row, column, pos
    else:
        return None, None, None, None

def natural_sort_wells(well):
    match = re.match(r"([A-Za-z]+)([0-9]+)", well)
    if match:
        return (match.group(1), int(match.group(2)))
    return (well,)

def row_to_index(row_letter):
    return ord(row_letter.upper()) - ord('A')

def extract_column_number(column_label):
    return int(''.join(filter(str.isdigit, column_label)))

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from math import ceil, sqrt
from natsort import natsorted
from skimage import io
from datetime import datetime

def extract_time_from_folder(folder_name):
    """
    Example function you provided to parse date/time from folder name
    which is assumed to end with _YYYYMMDD_HHMMSS.
    """
    folder_name = folder_name.rstrip('\\/')  # remove trailing slash
    parts = folder_name.split('_')
    date_str = parts[-2]
    time_str = parts[-1]
    return datetime.strptime(f'{date_str} {time_str}', '%Y%m%d %H%M%S')

def find_previous_timepoint_csv(current_csv_file):
    """
    Given the path to the current CSV, attempt to:
      1. Identify the parent folder that has the date/time in its name.
      2. Extract the 'prefix' before the first underscore.
      3. Find all sibling subfolders in the main experiment directory
         that have the same prefix and parse their times.
      4. Pick the folder with the largest date/time that is strictly
         earlier than the current folder’s date/time.
      5. Return the path to that folder's corresponding CSV (if found).
    
    Returns:
      (previous_csv_file, previous_folder_time, current_folder_time)
      or (None, None, None) if not found.
    """
    # 1) Identify the parent folder that includes the date/time in its name.
    #    e.g. ...\PTEN_Growth_Assay_..._20241227_185901\model_outputs\...
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_csv_file)))          # ...\model_outputs
    current_folder_name = os.path.basename(parent_dir)

    # 2) Extract the prefix before the first underscore (e.g. 'PTEN')
    split_folder = current_folder_name.split('_')
    if not split_folder:
        return None, None, None
    experiment_prefix = split_folder[0]  # e.g. PTEN

    # 3) The 'main experiment directory' is two levels above parent_parent_dir
    main_experiment_dir = os.path.dirname(parent_dir)
    current_time = extract_time_from_folder(current_folder_name)

    # Gather all subfolders that start with the same prefix:
    # e.g. PTEN_Growth_Assay_...
    candidate_folders = []
    for entry in os.scandir(main_experiment_dir):
        if entry.is_dir() and entry.name.startswith(experiment_prefix):
            try:
                folder_time = extract_time_from_folder(entry.name)
                candidate_folders.append((entry.path, folder_time))
            except Exception:
                # If parsing fails, skip
                pass

    # 4) Among candidate folders, pick the one with the largest date/time
    #    that is still strictly earlier than 'current_time'
    candidate_folders = sorted(candidate_folders, key=lambda x: x[1])  # sort by time
    previous_folder = None
    previous_time = None
    for folder_path, folder_time in candidate_folders:
        if folder_time < current_time:
            previous_folder = folder_path
            previous_time = folder_time
        else:
            # Once we hit a time >= current_time, stop
            break

    if previous_folder is None:
        return None, None, None  # no earlier timepoint found

    # 5) Attempt to locate the CSV within the 'model_outputs' subfolder
    #    (adjust wildcard as needed to find your actual results CSV)
    model_outputs_path = os.path.join(previous_folder, 'model_outputs')
    if not os.path.exists(model_outputs_path):
        return None, None, None

    # Example: searching recursively for "*_results.csv" in model_outputs
    # You may need to refine how you pick the correct CSV if multiple exist
    results_csvs = glob.glob(os.path.join(model_outputs_path, '**', '*_results.csv'), recursive=True)
    if not results_csvs:
        return None, None, None

    # In case there's more than one, pick any or pick the first natsorted
    results_csvs = natsorted(results_csvs)
    previous_csv_file = results_csvs[0]

    return previous_csv_file, previous_time, current_time

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import ListedColormap
from math import ceil, sqrt
from natsort import natsorted
from skimage import io  # for reading images if needed

###############################################################################
# Custom normalization that reserves a slot for zero.
###############################################################################
class LogNormZeroReserved(colors.Normalize):
    def __init__(self, vmin, vmax, clip=False):
        """
        For cell_density:
          - 0 maps to 0 (drawn as black)
          - Positive values are mapped (in log10 space) from vmin to vmax
            into the range [1/N, 1] (with N=257).
        """
        new_vmin = vmin if vmin > 0 else 1e-6
        new_vmax = vmax if vmax > new_vmin else new_vmin * 10
        self.N = 257  # total number of discrete color slots
        super().__init__(new_vmin, new_vmax, clip)

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip
        value = np.asarray(value, dtype=np.float64)
        normed = np.empty_like(value)
        
        # For exactly zero, assign normalized 0.
        mask0 = (value == 0)
        normed[mask0] = 0.0
        
        # For positive values, apply log10 scaling.
        maskpos = (value > 0)
        if np.any(maskpos):
            try:
                log_vmin = np.log10(self.vmin)
                log_vmax = np.log10(self.vmax)

                denom = log_vmax - log_vmin
                if denom == 0:
                    denom = 1.0
                normed[maskpos] = (1/self.N) + (1 - 1/self.N) * (
                    (np.log10(value[maskpos]) - log_vmin) / denom
                )
            except Exception as e:
                print("[DEBUG] Exception in __call__ log scaling:", e)
                raise e
        return normed

    def inverse(self, normed):
        normed = np.asarray(normed, dtype=np.float64)
        inv = np.empty_like(normed)
        # For normalized values at or below 1/N, return 0.
        mask0 = (normed <= (1/self.N))
        inv[mask0] = 0.0
        
        # For positive normalized values, invert the mapping.
        maskpos = (normed > (1/self.N))
        try:
            log_vmin = np.log10(self.vmin)
            log_vmax = np.log10(self.vmax)
            denom = log_vmax - log_vmin
            if denom == 0:
                denom = 1.0
            inv[maskpos] = 10 ** (
                ((normed[maskpos] - 1/self.N) / (1 - 1/self.N)) * denom + log_vmin
            )
        except Exception as e:
            print("[DEBUG] Exception in inverse:", e)
            raise e
        return inv

###############################################################################
# Create a custom viridis colormap that reserves index 0 for black.
###############################################################################
def create_custom_viridis():
    # Get 256 colors from viridis using the new API.
    base = plt.colormaps['viridis'](np.linspace(0, 1, 256))
    # Prepend black so that index 0 is black.
    newcolors = np.vstack((np.array([0, 0, 0, 1]), base))
    return ListedColormap(newcolors)

###############################################################################
# Main heatmap creation function.
###############################################################################
def create_heatmaps(csv_file, value_col='cell_density', simple=True):
    # --- Read and preprocess CSV data ---
    df = pd.read_csv(csv_file)
    df['Row'] = df['Row'].str.upper().str[0].map(lambda x: ord(x) - ord('A')).astype(int)
    df['Column'] = df['Well'].str.extract('(\d+)').astype(int)
    if 'Position' not in df.columns:
        df['Position'] = 1
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce').astype(int)
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df = df.dropna(subset=['Row', 'Column', 'Position', value_col])
    
    # --- Set up subplot grid based on well layout ---
    unique_rows = natsorted(df['Row'].unique())
    unique_cols = natsorted(df['Column'].unique())
    row_indices = {row: idx for idx, row in enumerate(unique_rows)}
    col_indices = {col: idx for idx, col in enumerate(unique_cols)}
    nrows, ncols = len(unique_rows), len(unique_cols)
    subplot_side_length = 5
    figsize = (ncols * subplot_side_length, nrows * subplot_side_length)
    fig_total, axs_total = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    
    # --- Determine normalization limits from positive (nonzero) values ---
    nonzero_values = df[df[value_col] > 0][value_col]
    vmin = nonzero_values.min() if not nonzero_values.empty else 1e-6
    vmax = nonzero_values.max() if not nonzero_values.empty else vmin * 10

    norm = LogNormZeroReserved(vmin=vmin, vmax=vmax)
    custom_cmap = create_custom_viridis()
    
    # --- Group data by well and plot each heatmap ---
    grouped = df.groupby('Well')
    wells = natsorted(grouped.groups.keys())
    
    for well in wells:
        well_data = grouped.get_group(well)
        if well_data.empty:
            continue

        # Sort the data by Position so that assignment is in natural order.
        well_data = well_data.sort_values('Position')
        
        row_key = int(well_data['Row'].iloc[0])
        col_key = int(well_data['Column'].iloc[0])
        ax = axs_total[row_indices[row_key], col_indices[col_key]]
        
        # Arrange cell positions into a square grid.
        max_pos = well_data['Position'].max()
        grid_size = int(ceil(sqrt(max_pos)))
        heatmap_data = np.full((grid_size, grid_size), np.nan)
        
        # Compute indices so that 0-index corresponds to top row:
        positions = well_data['Position'] - 1
        x_indices = (positions % grid_size).astype(int)
        # Reverse the row order to put the first position at the top:
        y_indices = (grid_size - 1 - (positions // grid_size)).astype(int)
        heatmap_data[y_indices, x_indices] = well_data[value_col].values
        
        # Compute simple statistics for annotation.
        avg_value = np.nanmean(heatmap_data)
        med_value = np.nanmedian(heatmap_data)
        std_value = np.nanstd(heatmap_data)
        cov_value = std_value / avg_value if avg_value != 0 else np.nan
        
        # IMPORTANT: set origin='lower' so that the array indices match the coordinates
        ax.imshow(
            heatmap_data,  
            cmap=custom_cmap,
            norm=norm,
            extent=[0, grid_size, 0, grid_size],
            interpolation='nearest',
            origin='lower'
        )
                
        # Place text annotations for each cell
        for (y, x), val in np.ndenumerate(heatmap_data):
            if np.isfinite(val):
                ax.text(x + 0.5, y + 0.5, f"{val:.1f}", ha='center', va='center', fontsize=8,
                        color='white', backgroundcolor='black')
                
        # Set title and axis properties (done once per well)
        ax.set_title(f"Well: {well}\nAvg: {avg_value:.2f}, Med: {med_value:.2f}, Cov: {cov_value:.2f}")
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.axis('off')
    
    plt.tight_layout()
    
    # --- Create the colorbar ---
    boundaries = np.concatenate(([-0.1], np.linspace(vmin, vmax, norm.N - 1)))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap)
    sm.set_array([])
    cbar = fig_total.colorbar(sm, ax=axs_total.ravel().tolist(), boundaries=boundaries, extend='min', shrink=0.5)
    
    output_file = os.path.join(
        os.path.dirname(csv_file),
        f"{os.path.splitext(os.path.basename(csv_file))[0]}_heatmap_{value_col}.png"
    )
    plt.savefig(output_file, dpi=200)
    plt.close()
    print(f"Saved heatmap to {output_file}")

def process_mask_files(masks_dir, cm_per_pixel, total_well_area, force_save=False, save_outlines=False, filter_min_size=None):
    csv_filepath = os.path.join(masks_dir, f"{os.path.basename(masks_dir)}_results.csv")
    if (not os.path.exists(csv_filepath)) or force_save:
        mask_files = [os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if '_cp_masks' in f]
        if mask_files:
            if ".png" in mask_files[0]:
                masks = cv2.imread(str(mask_files[0]), cv2.IMREAD_UNCHANGED)
            else:
                masks = imread(mask_files[0])
            area = masks.size * cm_per_pixel * cm_per_pixel

            # results = [extract_metrics(file, area, total_well_area) for file in mask_files]

            with ProcessPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(extract_metrics, mask_files, [area]*len(mask_files), [total_well_area]*len(mask_files), [filter_min_size]*len(mask_files)))

            if save_outlines:
                outline_cells_directory(os.path.dirname(os.path.dirname(masks_dir)), masks_dir, None)

            model_results = pd.DataFrame(results)
            if model_results['file'].any():
                model_results.to_csv(csv_filepath, index=False)
                print(f"Saved {csv_filepath}")
        try:
            create_heatmaps(csv_filepath, "cell_density")
        except Exception as e:
            print(f"failed to process heatmap cause {e}")

    return csv_filepath

def outline_cells_directory(img_directory, live_directory, dead_directory=None, channel=1):
    if not live_directory:
        print("no live cells directory")
        return
    tif_files = [f for f in os.listdir(img_directory) if f.lower().endswith('.tif')]
    pairs = [(f, f.lower().replace('.tif', '_cp_masks.png')) for f in tif_files if '_cp_masks' not in f]
    for img_file, mask_file in pairs:
        img_path = os.path.join(img_directory, img_file)
        live_mask_path = os.path.join(live_directory, mask_file)
        
        if dead_directory is not None:
            dead_mask_path = os.path.join(dead_directory, mask_file)
        else:
            dead_mask_path = None

        color_and_outline_cells_per_channel(img_path, live_mask_path, dead_mask_path, channel=channel)

def color_and_outline_cells_per_channel(img_path, green_masks_path, red_masks_path=None, alpha=1, beta=0.2, gamma=0, thickness=4, quality=60, channel=1):
    try:
        # Determine the output directory based on the green_masks_path
        output_dir = os.path.join(os.path.dirname(green_masks_path), "colored_and_outlined_cells")
        os.makedirs(output_dir, exist_ok=True)
        
        # Define the channel indices you want to process
        channel_indices = [channel]  # Modify this list if you want to process more channels
        
        for channel_idx in channel_indices:
            # Construct the output image path
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_img_filename = f"{base_name}_channel_{channel_idx}.jpg"
            output_img_path = os.path.join(output_dir, output_img_filename)
            
            # Check if the output file already exists
            if os.path.exists(output_img_path):
                continue  # Skip to the next channel if the output exists
            
            # Proceed with processing since the output does not exist
            print(img_path)
            img = imread(img_path).astype(np.float32)
            
            # Handle image dimensions
            if img.ndim == 2:
                img = np.stack((img, img, img), axis=0)
            if img.ndim == 3 and img.shape[0] == 5:
                img = img[[0, 1, 3], :, :]
            elif img.ndim == 3 and img.shape[0] != 3:
                raise ValueError("Image has 3 dimensions but does not have 3 channels")
            
            # Read mask images
            green_masks = cv2.imread(green_masks_path, cv2.IMREAD_UNCHANGED)
            if red_masks_path is not None:
                red_masks = cv2.imread(red_masks_path, cv2.IMREAD_UNCHANGED)
            else:
                red_masks = np.zeros_like(green_masks)
            
            # Extract and normalize the specific channel
            channel_img = img[channel_idx, :, :]
            low = np.percentile(channel_img, 1)
            high = np.percentile(channel_img, 99)
            scale = 255
            low_clip = 0
            channel_img_normalized = ((channel_img - low) / (high - low) * scale).clip(low_clip, scale)
            
            # Convert to BGR format
            channel_img_colored = cv2.cvtColor(channel_img_normalized.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            
            # Create color overlays
            green_overlay = np.zeros_like(channel_img_colored)
            red_overlay = np.zeros_like(channel_img_colored)
            yellow_overlay = np.zeros_like(channel_img_colored)
            
            # Apply green mask
            green_overlay[green_masks > 0] = [0, 255, 0]
            
            # Apply red mask if provided
            if red_masks_path is not None:
                red_overlay[red_masks > 0] = [0, 0, 255]
            
            # Blend the overlays with the original image
            shaded_img = cv2.addWeighted(channel_img_colored, alpha, green_overlay, beta, 0)
            shaded_img = cv2.addWeighted(shaded_img, alpha, red_overlay, beta, 0)
            shaded_img = cv2.addWeighted(shaded_img, alpha, yellow_overlay, beta, gamma)
            
            # Define the structuring element for dilation
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness))
            
            # Dilate green masks to get outlines
            dilated_green = cv2.dilate(green_masks, kernel)
            outlines_green = dilated_green - green_masks
            
            # Apply green outlines
            shaded_img[outlines_green > 0] = [0, 255, 0]
            
            # Apply red outlines if red_masks_path is provided
            if red_masks_path is not None:
                dilated_red = cv2.dilate(red_masks, kernel)
                outlines_red = dilated_red - red_masks
                shaded_img[outlines_red > 0] = [0, 0, 255]
            
            # Save the processed image
            cv2.imwrite(str(output_img_path), shaded_img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

def outline_cells_per_channel(img_path, green_masks_path, red_masks_path, thickness=3, quality=60, model_name=""):
    try:
        img = imread(img_path).astype(np.float32)
            # Check the number of dimensions in the image
        if img.ndim == 2:
            # If the image is 2D (grayscale), add three channels
            img = np.stack((img, img, img), axis=0)
        if img.ndim == 3 and img.shape[0] == 5:
            img = img[[0, 1, 3], :, :]
        elif img.ndim == 3 and img.shape[0] != 3:
            # If the image is 3D but does not have 3 channels, we should handle it as needed
            # This condition is specific to images with non-standard shapes, modify if necessary
            raise ValueError("Image has 3 dimensions but does not have 3 channels")
        
        red_masks = imread(red_masks_path)
        green_masks = imread(green_masks_path)
        
        outlines_dir = os.path.join(os.path.dirname(green_masks_path), "outlines")
        os.makedirs(outlines_dir, exist_ok=True)

        for channel_idx in [1]:
        # for channel_idx in range(img.shape[0]):
            channel_img = img[channel_idx, :, :]
            low = np.percentile(channel_img, 1)
            high = np.percentile(channel_img, 99)
            scale = 255
            low_clip = 0
            channel_img_normalized = ((channel_img - low) / (high - low) * scale).clip(low_clip, scale)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness))

            dilated_red = cv2.dilate(red_masks, kernel)
            outlines_red = dilated_red - red_masks

            dilated_green = cv2.dilate(green_masks, kernel)
            outlines_green = dilated_green - green_masks

            channel_img_outlined = cv2.cvtColor(channel_img_normalized.astype(np.uint8), cv2.COLOR_GRAY2BGR)

            # Apply green outlines
            channel_img_outlined[outlines_green > 0] = [0, 255, 0]

            # Apply red outlines
            channel_img_outlined[outlines_red > 0] = [0, 0, 255]

            # Apply yellow outlines
            channel_img_outlined[np.logical_and(outlines_red > 0, outlines_green > 0)] = [0, 255, 255]

            outlined_img_path = os.path.join(outlines_dir, f"{os.path.basename(img_path).replace('.tif', '')}_channel_{channel_idx}.jpg")
            cv2.imwrite(str(outlined_img_path), channel_img_outlined, [cv2.IMWRITE_JPEG_QUALITY, quality])
    except Exception as e:
        print(e)


# Constants and configurations
PLATE_TYPE = "12W"
MAGNIFICATION = "20x"
CYTATION = True

PLATE_AREAS = {"6W": 9.6, "12W": 3.8, "24W": 2, "48W": 1.1, "96W": 0.32}
CM_PER_MICRON = 1 / 10000
if CYTATION:
    MICRONS_PER_PIXEL = 1389 / 1992 if MAGNIFICATION == "10x" else 694 / 1992
    IMAGE_AREA_CM = 1992 * 1992 * MICRONS_PER_PIXEL**2 * CM_PER_MICRON**2
else:
    if MAGNIFICATION=="10x":
        MICRONS_PER_PIXEL = 0.61922571983322461  # Taken from image metadata EVOS 10x
        IMAGE_AREA_CM = 0.0120619953 # Taken from image metadata EVOS 10x
    else:
        MICRONS_PER_PIXEL = 1.5188172690164046  # Taken from image metadata EVOS 4x
        IMAGE_AREA_CM = 0.0725658405  # Taken from image metadata EVOS 4x
CM_PER_PIXEL = CM_PER_MICRON*MICRONS_PER_PIXEL

# MICRONS_PER_PIXEL = 0.65 # for MSSR 10x
# IMAGE_AREA_CM = 1992*1992 * MICRONS_PER_PIXEL**2 * CM_PER_MICRON**2

PRETRAINED_MODEL_INFOS = [
    # [r"C:\Users\Tim\.cellpose\models\cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_2_nchan_3_dim_2_cropped_2023_11_13_21_35_54.454698_epoch_3999", 0, 0, 3, 448, [0.98665061,0.17478241]],
    # [r"D:\202310-12W-stuff\merged_train\models\cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_2_nchan_3_dim_2_merged_train_2024_03_07_08_22_14.829043_epoch_400", 0, 0, 3, 448], # np.array[1.00190246,-0.09723356]
    # [r"D:\202310-12W-stuff\merged_train\models\cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_2_nchan_3_dim_2_merged_train_2024_03_08_16_33_38.722130_epoch_994", 0, 0, 3, 448],9
    # [r"D:\202310-12W-stuff\NEW-MERGED-TRAIN-FROM-DAPI\cropped\448\top_results_back780\models\cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_2_nchan_3_dim_2_top_results_2024_03_18_04_00_37.307042_epoch_780", 0, 0, 3, 448],
    # [r"D:\202310-12W-stuff\NEW-MERGED-TRAIN-FROM-DAPI\models/cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_2_nchan_3_dim_2_NEW-MERGED-TRAIN-FROM-DAPI_2024_03_29_02_03_10.875324_epoch_390", 0, 0, 3, 448],
    # [r"D:\202310-12W-stuff\NEW-MERGED-TRAIN-FROM-DAPI\models/cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_2_nchan_3_dim_2_NEW-MERGED-TRAIN-FROM-DAPI_2024_03_29_02_03_10.875324_epoch_740", 0, 0, 3, 448],
    # [r"D:\202310-12W-stuff\NEW-MERGED-TRAIN-FROM-DAPI\models/cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_2_nchan_3_dim_2_NEW-MERGED-TRAIN-FROM-DAPI_2024_03_29_02_03_10.875324_epoch_810", 0, 0, 3, 448],
    [r"D:\202310-12W-stuff\NEW-MERGED-TRAIN-FROM-DAPI\models/cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_2_nchan_3_dim_2_NEW-MERGED-TRAIN-FROM-DAPI_2024_03_29_02_03_10.875324_epoch_960", 0, 0, 3, 448],
    ]

import os
from multiprocessing import Pool, cpu_count
from functools import partial

def process_subfolder(subfolder_name, base_dir, model_output_suffix):
    subfolder_path = os.path.join(base_dir, subfolder_name)
    # Check if the item is a directory
    if os.path.isdir(subfolder_path):
        param1 = subfolder_path
        param2 = os.path.join(subfolder_path, model_output_suffix)
        param3 = None
        # Call the function with the specified parameters
        outline_cells_directory(param1, param2, param3)

def extract_intensity_metrics(cellprob_filepath, area, total_well_area):
    """
    Reads the cell probability image and computes total intensity metrics.
    The total intensity is forced to be at least 0.
    """
    try:
        # Read the cell probability image (assumed to be numeric)
        cellprob_img = imread(cellprob_filepath)
    except Exception as e:
        print(f"Error reading {cellprob_filepath}: {e}")
        return None

    total_intensity = max(np.sum(cellprob_img), 0)
    max_intensity = max(np.max(cellprob_img), 0)

    # Extract well info (assuming the filename contains the well info in the same format)
    well, row, column, pos = extract_well_info(cellprob_filepath)

    results = {
        'file_path': cellprob_filepath,
        'file': os.path.basename(cellprob_filepath),
        'p1': os.path.basename(os.path.dirname(cellprob_filepath)),
        'p2': os.path.basename(os.path.dirname(os.path.dirname(cellprob_filepath))),
        'p3': os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(cellprob_filepath)))),
        'total_intensity': total_intensity,
        'max_intensity': max_intensity,
        'Well': well,
        'Row': row,
        'Column': column,
        'Position': pos
    }
    return results

def process_cellprob_files(masks_dir, cm_per_pixel, total_well_area, force_save=False):
    """
    Processes cell probability files by:
      - Replacing _cp_masks.tif with _cellProb.tif to obtain the intensity images.
      - Computing intensity metrics (total intensity, intensity density, etc.).
      - Scaling the dataset so that the minimum total_intensity is 0.
      - Merging (appending) these new intensity columns with the existing CSV (if available).
      - Creating a heatmap (using intensity_density) and a cumulative intensity plot.
    """
    # CSV to store intensity metrics.
    intensity_csv_filepath = os.path.join(masks_dir, f"{os.path.basename(masks_dir)}_results_intensity.csv")
    
    # Find the _cp_masks.tif files and derive the corresponding _cellProb.tif file paths.
    mask_files = [os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if '_cp_masks.tif' in f]
    if not mask_files:
        print("No mask files found.")
        return None

    cellprob_files = [f.replace('_cp_masks.tif', '_cellProb.tif') for f in mask_files]

    # Determine the image area (in cm²) from the first cellProb image.
    sample_img = imread(cellprob_files[0])
    area = sample_img.size * (cm_per_pixel ** 2)

    # Process all cellProb files in parallel.
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(
            extract_intensity_metrics,
            cellprob_files,
            [area] * len(cellprob_files),
            [total_well_area] * len(cellprob_files)
        ))
    results = [r for r in results if r is not None]
    intensity_df = pd.DataFrame(results)

    # Scale the entire intensity dataset so that the minimum total_intensity is 0.
    min_intensity = intensity_df['total_intensity'].min()
    intensity_df['total_intensity'] = intensity_df['total_intensity'] - min_intensity
    intensity_df['intensity_density'] = intensity_df['total_intensity'] / area
    intensity_df['total_intensity_in_well'] = intensity_df['intensity_density'] * total_well_area

    # Save the intensity metrics to CSV.
    intensity_df.to_csv(intensity_csv_filepath, index=False)
    print(f"Saved intensity results CSV to {intensity_csv_filepath}")

    # Check if the "old" CSV exists (from cell density metrics).
    old_csv_filepath = os.path.join(masks_dir, f"{os.path.basename(masks_dir)}_results.csv")
    if os.path.exists(old_csv_filepath):
        try:
            old_df = pd.read_csv(old_csv_filepath)
            # Create a common key 'base' by stripping the specific suffixes from the file names.
            old_df['base'] = old_df['file'].str.replace('_cp_masks.tif', '', regex=False)
            intensity_df['base'] = intensity_df['file'].str.replace('_cellProb.tif', '', regex=False)
            # Merge on the 'base' column.
            merged_df = pd.merge(
                old_df,
                intensity_df[['base', 'total_intensity', 'intensity_density', 'total_intensity_in_well']],
                on='base',
                how='left'
            )
            # Optionally drop the helper "base" column.
            merged_df = merged_df.drop(columns=['base'])
            # Save the merged CSV (you can choose to overwrite or create a new file).
            merged_csv_filepath = os.path.join(masks_dir, f"{os.path.basename(masks_dir)}_results_merged.csv")
            merged_df.to_csv(merged_csv_filepath, index=False)
            print(f"Saved merged CSV with appended intensity columns to {merged_csv_filepath}")
        except Exception as e:
            print(f"Error merging intensity data with old CSV: {e}")
    else:
        print("Old CSV not found; intensity CSV remains separate.")

    # Create a heatmap based on intensity_density using the intensity CSV.
    try:
        create_heatmaps(intensity_csv_filepath, value_col="intensity_density")
    except Exception as e:
        print(f"Failed to create intensity heatmap: {e}")

    # Create an additional plot: cumulative total intensity vs. number of images.
    try:
        plot_cumulative_intensity(intensity_csv_filepath)
    except Exception as e:
        print(f"Failed to create cumulative intensity plot: {e}")
        
    return intensity_csv_filepath


def plot_cumulative_intensity(csv_file):
    """
    Reads the CSV file with intensity metrics, sorts the images by total intensity (highest first),
    computes the cumulative sum of total intensity, and plots cumulative total intensity versus
    the number of images.
    """
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file {csv_file}: {e}")
        return

    # Sort by total_intensity in descending order
    df_sorted = df.sort_values('total_intensity', ascending=False).reset_index(drop=True)
    df_sorted['cumulative_total_intensity'] = df_sorted['total_intensity'].cumsum()

    plt.figure(figsize=(8, 6))
    plt.plot(
        np.arange(1, len(df_sorted) + 1),
        df_sorted['cumulative_total_intensity'],
        marker='o',
        linestyle='-'
    )
    plt.xlabel("Number of Images (sorted by highest total intensity)")
    plt.ylabel("Cumulative Total Intensity")
    plt.title("Cumulative Total Intensity vs. Number of Images")
    plt.grid(True)

    output_plot = os.path.join(os.path.dirname(csv_file), "cumulative_total_intensity_plot.png")
    plt.savefig(output_plot, dpi=200)
    plt.close()
    print(f"Saved cumulative intensity plot to {output_plot}")


if __name__ == "__main__":
    # process_mask_files(r"E:\MERGED_sspsygene_growthassay_Genes_1-10-041025-IPSC-NPC\IPSC-01_04-ARID1B_ASXL3_CACNA1G_CHD8-GA-Day_1_20250409_200124\model_outputs\2024_09_01_00_42_19.503900_epoch_1999_0.4_0", CM_PER_PIXEL, total_well_area=0.96, force_save=False, save_outlines=False)
    # create_heatmaps(r"E:\MERGED_sspsygene_growthassay_colonies\blank_day1_20250210_192713\model_outputs\2025_02_12_01_33_04.460354_epoch_800_0.4_0\2025_02_12_01_33_04.460354_epoch_800_0.4_0_results.csv", "cell_density")
    # create_heatmaps(r"E:\MERGED_sspsygene_growthassay_colonies\blank_day1_20250210_192713\model_outputs\2024_09_01_00_42_19.503900_epoch_1999_0.4_0\2024_09_01_00_42_19.503900_epoch_1999_0.4_0_results.csv", "cell_density")
    # process_cellprob_files(r"E:\MERGED_sspsygene_growthassay_colonies\blank_day1_20250210_192713\model_outputs\2025_02_17_22_42_34.566635_epoch_399_0.4_0", cm_per_pixel=CM_PER_PIXEL,
    #                        total_well_area=0.96,  # use your well area value
    #                        force_save=True)
    # outline_cells_directory(r"E:\MERGED_sspsygene_growthassay_colonies\0_day9_20250218_165816\train", r"E:\MERGED_sspsygene_growthassay_colonies\0_day9_20250218_165816\train\model_outputs\2025_02_18_09_30_51.537567_epoch_799_0.4_0")
    # outline_cells_directory(r"E:\MERGED_sspsygene_growthassay_colonies\0_day9_20250218_165816\train", r"E:\MERGED_sspsygene_growthassay_colonies\0_day9_20250218_165816\train\model_outputs\2025_02_26_01_15_09.957112_epoch_899_0.4_0")
    # outline_cells_directory(r"E:\MERGED_sspsygene_growthassay_colonies\0_day5_20250214_121744\train1and2", r"E:\MERGED_sspsygene_growthassay_colonies\0_day5_20250214_121744\train1and2\model_outputs\2025_02_18_09_30_51.537567_epoch_799_0.4_0")
    # outline_cells_directory(r"E:\MERGED_sspsygene_growthassay_colonies\0_day5_20250214_121744\train1and2", r"E:\MERGED_sspsygene_growthassay_colonies\0_day5_20250214_121744\train1and2\model_outputs\2025_02_26_01_15_09.957112_epoch_899_0.4_0")

    # outline_cells_directory(r"E:\MERGED_20250312_MAZKO_Growthrate_AR_IB\mazKO_day1_20250312_154329", r"E:\MERGED_20250312_MAZKO_Growthrate_AR_IB\mazKO_day1_20250312_154329\model_outputs\2024_10_01_10_59_49.919832_epoch_2499_0.4_0")

    # base_dir = r"E:\MERGED_20250323_LS_JS_NPC_Village"

    # for entry in os.listdir(base_dir):
    #     if "J1" in entry:
    #         full_path = os.path.join(base_dir, entry)
    #         # model_output1 = os.path.join(full_path, "model_outputs", "2025_03_03_23_34_16.794792_epoch_899_0.4_2")
    #         model_output1 = os.path.join(full_path, "model_outputs", "2024_03_29_02_03_10.875324_epoch_960_0.4_0")
    #         model_output2 = os.path.join(full_path, "model_outputs", "2025_03_07_02_10_15.252341_epoch_3999_0.4_1")
    #         outline_cells_directory(full_path, model_output1, model_output2, channel=2)

    # model = r"2024_03_29_02_03_10.875324_epoch_960_0.4_0"
    # model = r"2024_09_01_00_42_19.503900_epoch_1999_0.4_0"
    model = r"2024_10_01_10_59_49.919832_epoch_2499_0.4_0"

    # full_path = r"E:\MERGED_Transduction eff for SNaP2.0 GFP espressing lines YK 4_4_25_10x\Transduction eff for SNaP2.0 GFP espressing lines YK 4_4_25_20250404_122623"
    # full_path = r"E:\MERGED_sspsygene_growthassay_Genes_1-10-041025-IPSC-NPC\IPSC-01_04-ARID1B_ASXL3_CACNA1G_CHD8-GA-Day_3_20250411_120206"
    full_path = r"E:\MERGED_1-10_NPC\NPC-01_04-ARID1B_ASXL3_CACNA1G_CHD8-GA-Day_3_20250411_182656"

    outline_cells_directory(full_path, os.path.join(full_path, "model_outputs", model), channel=2)
