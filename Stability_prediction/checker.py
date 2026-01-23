import os

# Change these as needed
parent_folder = "/pasteur/appa/scratch/nportal/results_boltz2_sp"
#parent_folder = "fake_parent_boltz"
output_file = "check_results.txt"

with open(output_file, "w") as out:
    for folder in os.listdir(parent_folder):
        number = folder.split('_')[-1]
        folder_path = os.path.join(parent_folder, folder)

        if os.path.isdir(folder_path):
            # Go into the subfolder(s) inside this folder
            for subfolder in os.listdir(folder_path):
                if 'msa' in subfolder:
                    subfolder_path = os.path.join(folder_path, subfolder)
                    file_to_check = f'protein_{number}_0.csv'

                    if os.path.isdir(subfolder_path):
                        file_path = os.path.join(subfolder_path, file_to_check)

                        if os.path.exists(file_path):
                            out.write(f"✅ Found in: {subfolder_path}\n")
                        else:
                            out.write(f"❌ Missing in: {subfolder_path}\n")

print(f"Results written to: {output_file}")
