import os
import rasterio
import numpy as np
import scipy.ndimage
import geopandas as gpd
from skimage.transform import resize

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap


#############################
### Historic Mining Area ####
#############################

def plot_multiple_masks_on_images(image_files, mask_files):
    """Plot multiple images and their corresponding masks side by side.
    
    Args:
        image_files (list): A list of file paths to the images.
        mask_files (list): A list of file paths to the masks.
    """

    # Ensure that the image_files and mask_files lists have the same length
    assert len(image_files) == len(mask_files), "The number of image files and mask files must be the same."

    # Calculate the number of rows and columns for the grid
    num_images = len(image_files)
    num_cols = min(num_images, 2)  # Maximum 2 images per row
    num_rows = (num_images + num_cols - 1) // num_cols  # Round up to the nearest integer

    # Create a figure
    fig = plt.figure(figsize=(10 * num_cols, 10 * num_rows))

    # Iterate over the image and mask files
    for i, (image_file, mask_file) in enumerate(zip(image_files, mask_files)):
        # Create a subplot
        ax = fig.add_subplot(num_rows, num_cols, i + 1)

        # Plot the mask on the image
        plot_mask_on_image(image_file, mask_file, ax)

    plt.tight_layout()
    plt.show()


def plot_mask_on_image(image_path, mask_path, ax):
    """Plot an image and its corresponding mask side by side.
    
    Args:
        image_path (str): The file path to the image.
        mask_path (str): The file path to the mask.
        ax (matplotlib.axes.Axes): The matplotlib axis to plot on.
    """
    
    # Plot the image
    image = plt.imread(image_path)
    plt.imshow(image)

    # Open and read the mask
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    mask = np.where(mask > 0, 1, 0)  # Convert the mask to binary

    # Create a square-shaped structure element
    structure = np.ones((3, 3))

    # Erode the mask and subtract it from the original mask to get the outline
    eroded_mask = scipy.ndimage.binary_erosion(mask, structure)
    outline = mask - eroded_mask

    # Create a new RGBA image
    rgba = np.zeros((outline.shape[0], outline.shape[1], 4))  # 4 for RGBA

    # Set the red channel to the outline
    rgba[..., 0] = outline

    # Set the alpha channel to the outline
    rgba[..., 3] = outline

    # Plot the RGBA image
    ax.imshow(rgba)

    # Extract the year from the image file name and set it as the title
    year = os.path.basename(image_path).split('_')[0]  # Change this line if the year is located differently in the file name
    ax.set_title(f'Year: {year}')


def plot_area_per_year(gpkg_files):
    """Plot the total mining area per year.

    Args:
        gpkg_files (list): A list of file paths to the .gpkg files.
    """

    # Initialize a list to store the total area for each year
    areas = []

    # Iterate over the .gpkg files
    for gpkg_file in gpkg_files:
        # Read the polygons from the .gpkg file
        polygons = gpd.read_file(gpkg_file)

        # Calculate the area of each polygon and sum them up
        total_area = polygons.area.sum()

        # Append the total area to the list
        areas.append(total_area)

    # Extract the years from the file names
    years = [os.path.basename(gpkg_file).split('_')[0] for gpkg_file in gpkg_files]  # Change this line if the year is located differently in the file name

    # Plot the total area per year
    plt.figure(figsize=(10, 6))
    plt.plot(years, areas, marker='o')
    plt.xlabel('Year')
    plt.ylabel('Total Area')
    plt.title('Total Area per Year')
    plt.grid(True)
    plt.show()


##############################
### Plot Chips and Masks #####
##############################

def plot_chips_and_masks(root, seed=0):
    """
    Plot randomly selected chips (numpy arrays) and their corresponding masks (numpy arrays).

    Parameters:
    - root (str): The root directory path
    - seed (int): The seed value for random number generation.

    Returns:
    None
    """

    chips_dir = root + "/data/processed/chips/train/chips"
    masks_dir = root + "/data/processed/chips/train/labels"

    files = os.listdir(chips_dir)
    fig, axs = plt.subplots(5, 2, figsize=(10, 25))

    # generate 5 random indices in the range of the number of files
    np.random.seed(seed)
    indices = list(np.random.choice(len(files), 5, replace=False))
    print(indices)

    for i, file_index in enumerate(indices):
        filename = os.path.join(chips_dir, files[file_index])
        img = np.load(filename)
        im2display = img.transpose((1, 2, 0))
        im2display = (im2display - im2display.min()) / (im2display.max() - im2display.min())
        im2display = np.clip(im2display, 0, 1)
        
        mask_filename = masks_dir + "/" + files[file_index].replace("_img", "_mask")
        mask = np.load(mask_filename).squeeze()
        
        resized_img = resize(im2display, (im2display.shape[0] // 2, im2display.shape[1] // 2))
        resized_mask = resize(mask, (mask.shape[0] // 2, mask.shape[1] // 2))

        # get date from the filename
        date = files[file_index].split("_")[3][:8]
        date = f"{date[:4]}-{date[4:6]}-{date[6:]}"

        axs[i, 0].imshow(resized_img)
        axs[i, 0].set_title(f"Image from {date}")
        
        axs[i, 1].imshow(resized_mask)
        axs[i, 1].set_title("Mask")
    
    plt.tight_layout()
    plt.show()


def plot_S2_geotiff_rgb(file_path):

    # Assuming 'image_path' is the path to your GeoTIFF file
    with rasterio.open(file_path) as src:
        # Read the red, green, and blue bands from the GeoTIFF
        r = src.read(1)
        g = src.read(2)
        b = src.read(3)

        # Stack the R, G, and B bands to create an RGB image
        rgb = np.dstack((r, g, b))

    # Normalize the RGB image
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

    # Display the RGB image
    plt.imshow(rgb)
    plt.show()

##############################
### Plotting the Prediction ##
##############################
def plot_pred_vs_true_mask(
    image, 
    true_mask, 
    pred_mask,
    ax=None,
    alpha=0.4,
    blend=True,
    add_legend=True,
    **kwargs
):
    """Show the predicted mask and the true mask on the input image.
    
    Args:
        image (numpy.ndarray): The input image.
        true_mask (numpy.ndarray): The true mask.
        pred_mask (numpy.ndarray): The predicted mask.
        ax (matplotlib.axes.Axes, optional): The matplotlib axis to plot on. If None, a new figure will be created. Defaults to None.
        alpha (float, optional): The alpha value for the mask. Defaults to 0.4.
        blend (bool, optional): Whether to blend the mask with the input image. Defaults to True.
        add_legend (bool, optional): Whether to add a legend to the plot. Defaults to True.
        kwargs (dict, optional): Additional arguments for matplotlib.pyplot.savefig().

    Returns:
        None
    """

    # Custom single-color colormap
    blue_cmap = ListedColormap(['#FF000000', 'blue'])
    yellow_cmap = ListedColormap(['#FF000000', 'yellow'])
    green_cmap = ListedColormap(['#FF000000', 'green'])
    red_cmap = ListedColormap(['#FF000000', 'red'])

    # Calculate different areas
    true_positive = np.logical_and(true_mask == 1, pred_mask == 1)
    true_negative = np.logical_and(true_mask == 0, pred_mask == 0)
    false_positive = np.logical_and(true_mask == 0, pred_mask == 1)
    false_negative = np.logical_and(true_mask == 1, pred_mask == 0)

    # Calculate metrics
    intersection = np.logical_and(true_mask, pred_mask)
    union = np.logical_or(true_mask, pred_mask)
    iou = np.sum(intersection) / np.sum(union)
    precision = np.sum(true_positive) / (np.sum(true_positive) + np.sum(false_positive))
    recall = np.sum(true_positive) / (np.sum(true_positive) + np.sum(false_negative))
    f1_score = 2 * precision * recall / (precision + recall)

    # Create a new figure if ax is None
    if ax is None:
        fig, ax = plt.subplots()

    # Plot mask
    if blend:
        ax.imshow(image, cmap='gray')
        ax.imshow(true_positive, cmap=yellow_cmap, alpha=alpha)
        ax.imshow(true_negative, cmap=blue_cmap, alpha=alpha)
        ax.imshow(false_positive, cmap=red_cmap, alpha=alpha)
        ax.imshow(false_negative, cmap=green_cmap, alpha=alpha)
    else:
        ax.imshow(true_positive, cmap=yellow_cmap, alpha=alpha)
        ax.imshow(true_negative, cmap=blue_cmap, alpha=alpha)
        ax.imshow(false_positive, cmap=red_cmap, alpha=alpha)
        ax.imshow(false_negative, cmap=green_cmap, alpha=alpha)
    ax.axis('off')

    # Add legend
    legend_elements = [
        patches.Patch(facecolor='yellow', alpha=alpha, label='True Positive'),
        patches.Patch(facecolor='blue', alpha=alpha, label='True Negative'),
        patches.Patch(facecolor='red', alpha=alpha, label='False Positive'),
        patches.Patch(facecolor='green', alpha=alpha, label='False Negative')
    ]
    if add_legend:
        ax.legend(handles=legend_elements)

    # Add metrics to title
    title = f"IoU: {iou:.2f}, F1 Score: {f1_score:.2f}"
    ax.set_title(title)

    # Save the figure if kwargs are provided
    if kwargs:
        plt.savefig(**kwargs)
    else:
        plt.show()


def plot_predictions(images, labels, preds):
    fig, axes = plt.subplots(1, 4, figsize=(15, 6))

    # Plot the image
    axes[0].imshow(images)
    axes[0].axis("off")
    axes[0].set_title("Image", fontsize=12)

    # Plot the actual segmentation
    axes[1].imshow(labels, vmin=0, vmax=1)
    axes[1].axis("off")
    axes[1].set_title("Actual", fontsize=12)

    # Plot the predicted segmentation
    axes[2].imshow(preds, vmin=0, vmax=1)
    axes[2].axis("off")
    axes[2].set_title("Pred", fontsize=12)

    # Plot the plot_pred_vs_true_mask
    plot_pred_vs_true_mask(images, labels, preds.squeeze(), ax=axes[3], add_legend=False)
    axes[3].set_title("Pred vs True", fontsize=12)

    plt.tight_layout()
    plt.show()