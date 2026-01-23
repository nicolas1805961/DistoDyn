from pymol import cmd
import os

folder_path = '/path/to/your/pdb/folder'

for filename in os.listdir(folder_path):
    if filename.endswith('.pdb'):
        file_path = os.path.join(folder_path, filename)
        object_name = os.path.splitext(filename)[0]
        
        cmd.load(file_path, object_name)
        print(f"Loaded {filename} as {object_name}")
