from pathlib import Path
import shutil

parent_dir = Path("/pasteur/appa/scratch/nportal/boltz/stability_prediction/missing_results")
output_dir = parent_dir.parent / "all_cif_files"
output_dir.mkdir(exist_ok=True)

for subdir in parent_dir.glob("boltz_results_protein_*"):
    cif = next(subdir.glob("predictions/*/*_model_0.cif"), None)
    if cif is None:
        continue

    protein_name = cif.stem.replace("_model_0", "")
    shutil.copy2(cif, output_dir / f"{protein_name}.cif")

