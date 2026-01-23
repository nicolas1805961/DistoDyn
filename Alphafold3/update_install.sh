#!/bin/bash

# Step 1: Remove the previous custom virtual environment
echo "Removing existing alphafold3_venv_custom..."
rm -rf ~/alphafold3/alphafold3_venv_custom/

# Step 2: Enter the container interactively to copy the virtual environment
echo "Entering the container to copy the virtual environment..."
apptainer exec --nv -B /pasteur -B /local -B /opt/gensoft/data/alphafold/3.0.1 \
    -B ~/alphafold3/alphafold_custom/:/app/alphafold/ \
    /opt/gensoft/exe/alphafold/3.0.1/libexec/alphafold-3.0.1.simg bash -c "
    echo 'Copying virtual environment...';
    cp -r /alphafold3_venv ~/alphafold3/alphafold3_venv_custom;
    echo 'Virtual environment copied. Exiting...';
    exit
"

# Step 5: Start the container again with the copied environment
echo "Starting the container with the custom virtual environment..."
NV_REQUIRED=${NV_REQUIRED} apptainer exec --nv -B /pasteur -B /local -B /opt/gensoft/data/alphafold/3.0.1 \
    -B ~/alphafold3/alphafold_custom/:/app/alphafold/ \
    -B ~/alphafold3/alphafold3_venv_custom/:/alphafold3_venv \
    /opt/gensoft/exe/alphafold/3.0.1/libexec/alphafold-3.0.1.simg bash -c "
    echo 'Activating the environment...';
    . /alphafold3_venv/bin/activate;
    cd /app/alphafold;
    echo 'Installing dependencies...';
    pip3 install -r dev-requirements.txt;
    pip3 install --no-deps .;
    echo 'Building data...';
    build_data;
    echo 'Installation and setup complete.';
    exit
"