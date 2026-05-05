This repository contains the code used to run all experiments described in the paper "Distodyn: Using Distograms to Learn Dynamic Protein Representations at Scale"

# Data pre-processing and model training for the binding affinity prediction task

## Stability_prediction/disto_model/Affinity_task

Once in the Stability_prediction\disto_model\Affinity_task folder, run the following command to train a model:

```python3 train_misato_affinity.py config${SLURM_ARRAY_TASK_ID}.yaml```

where SLURM_ARRAY_TASK_ID is the config file number

To generate pytorch geometric graphs (.pt files) containing the distance and distogram relations, run the command:

```python3 -u create_pt_distance_distogram_new.py --threshold 0.0001 -pdb "${PDB_ID}"```

where PDB_ID is the name of the pdb file in the misato dataset and threshold is the value used to binarize the probability matrix computed from distograms.

The following command allows to run inference on the binding affinity prediction task:

```python test.py --model_path <Path to the folder containing model's weights>```


# Data pre-processing for the binding site prediction task

## ./misato-dataset

To generate pytorch geometric graphs (.pt files) containing the distance and distogram relations, run the command:

```python3 -u create_pt_distance_distogram.py --threshold 0.0001 -pdb "${PDB_ID}"```

where PDB_ID is the name of the pdb file in the misato dataset and threshold is the value used to binarize the probability matrix computed from distograms.



# Model training for the binding site prediction task

## Stability_prediction/disto_model

Once in the Stability_prediction/disto_model folder, run the following command to train a model:

```python3 train_misato.py config${SLURM_ARRAY_TASK_ID}.yaml```

where SLURM_ARRAY_TASK_ID is the config file number.

The following command allows to run inference on the binding site prediction task:

```python test.py --model_path <Path to the folder containing model's weights>```


# Model training for the protein stability prediction task

## ./ThermoMPNN_modified

Once in the ./ThermoMPNN_modified, run the following command to train a model:

```python3 train_thermompnn.py config${SLURM_ARRAY_TASK_ID}.yaml```

where SLURM_ARRAY_TASK_ID is the config file number

The following command allows to run inference on the protein stability prediction task:

```python thermompnn_benchmarking.py --model_path <Path to the folder containing model's weights>```

