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

    missing_fasta_path = "missing_fastas"
    with open(missing_fasta_path, 'w') as missing_file:
        logging.info(f"Starting to copy confidence files from {parent_folder} to {output_folder}")

        for subfolder_name in os.listdir(parent_folder):
            name_id = subfolder_name.split('_')[-1]
            subfolder_path = os.path.join(parent_folder, subfolder_name, 'predictions', name_id)

            if os.path.isdir(subfolder_path):
                expected_filename = f"distogram_{name_id}_model_0.pkl"
                expected_filepath = os.path.join(subfolder_path, expected_filename)

                if os.path.isfile(expected_filepath):
                    dest_path = os.path.join(output_folder, expected_filename)
                    shutil.copy2(expected_filepath, dest_path)
                    logging.info(f"Copied: {expected_filepath} → {dest_path}")
                else:
                    logging.warning(f"Missing: {expected_filename} not found in {subfolder_path}")
            else:
                fasta_name = f"{name_id}.fasta"
                missing_file.write(f"{fasta_name}\n")
                logging.warning(f"Missing: {subfolder_path}")

        logging.info("Finished copying confidence files.")

# Example usage
if __name__ == "__main__":
    parent_folder = "/pasteur/appa/scratch/nportal/results_boltz2"
    output_folder = "/pasteur/appa/scratch/nportal/boltz2_distograms"
    copy_confidence_files(parent_folder, output_folder)
