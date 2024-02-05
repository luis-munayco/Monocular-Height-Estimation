import os

def rename_mask_files(images_dir, masks_dir, mask_suffix='_mask'):
    # List all files in the images directory
    image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]

    for image_file in image_files:
        # Extract the base filename without extension
        base_filename, file_extension = os.path.splitext(image_file)

        # Replace the last part of the base filename (AGL) with BAM
        base_filename = base_filename.rsplit('_', 1)[0]
        base_filename_m = base_filename + '_BAM'
        # Generate the new mask file name based on the modified image base filename and mask_suffix
        mask_file = base_filename_m + mask_suffix + file_extension

        # Generate the full file paths
        image_path = os.path.join(images_dir, image_file)
        original_mask_path = os.path.join(masks_dir, base_filename + '_AGL' + file_extension)
        mask_path = os.path.join(masks_dir, mask_file)

        # Rename the mask files
        os.rename(original_mask_path, mask_path)

if __name__ == "__main__":
    # Specify your images and masks directories
    images_directory = 'data/imgs/'
    masks_directory = 'data/masks/'

    # Rename only the mask files by adding the '_mask' suffix
    rename_mask_files(images_directory, masks_directory)
