import os
import shutil
import logging

def setup_logger(output_folder):
    log_file = os.path.join(output_folder, "copy_confidences.log")
    logging.basicConfig(
        filename=log_file,
        filemode='w',  # Overwrite on each run
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def copy_confidence_files(parent_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    setup_logger(output_folder)

    logging.info(f"Starting to copy confidence files from {parent_folder} to {output_folder}")

    for subfolder_name in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder_name)
        if os.path.isdir(subfolder_path):
            expected_filename = f"{subfolder_name}_confidences.pkl"
            expected_filepath = os.path.join(subfolder_path, expected_filename)

            if os.path.isfile(expected_filepath):
                dest_path = os.path.join(output_folder, expected_filename)
                shutil.copy2(expected_filepath, dest_path)
                logging.info(f"Copied: {expected_filepath} → {dest_path}")
            else:
                logging.warning(f"Missing: {expected_filename} not found in {subfolder_path}")

    logging.info("Finished copying confidence files.")

# Example usage
if __name__ == "__main__":
    parent_folder = "/pasteur/appa/scratch/nportal/results_inference"
    output_folder = "/pasteur/appa/scratch/nportal/af3_distograms"
    copy_confidence_files(parent_folder, output_folder)
