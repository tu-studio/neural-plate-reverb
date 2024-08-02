import itertools
import subprocess
import os 
import yaml

# Load parameters from params.yaml
def load_params():
    with open('params.yaml', 'r') as file:
        return yaml.safe_load(file)

# Submit experiment for hyperparameter combination
def submit_batch_job(index, params):
    # Convert params dictionary to a string of Hydra overrides
    param_string = ' '.join([f"-S {k}={v}" for k, v in params.items()])
    
    # Set dynamic parameters for the batch job as environment variables
    env = {
        **os.environ,
        "EXP_PARAMS": param_string,
        "INDEX": str(index)
    }
    # Run sbatch command with the environment variables
    subprocess.run(['/usr/bin/bash', '-c', 'sbatch batchjob.sh'], env=env)

if __name__ == "__main__":
    # Load default parameters
    default_params = load_params()

    # Define parameter ranges for grid search
    param_grid = {
        'train.latent_dim': [128, 64, 32],
        'train.kernel_size': [13, 16, 20],
        'train.n_epochs': [500, 1000, 1500],
    }

    # Generate all combinations of parameters
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Submit a job for each parameter combination
    for index, exp_params in enumerate(experiments):
        # Update default parameters with the current experiment parameters
        current_params = default_params.copy()
        for k, v in exp_params.items():
            current_params['train'][k.split('.')[-1]] = v
        
        # Flatten the nested dictionary
        flattened_params = {f"{outer_key}.{inner_key}": value 
                            for outer_key, inner_dict in current_params.items() 
                            for inner_key, value in inner_dict.items()}
        
        submit_batch_job(index, flattened_params)

    print(f"Submitted {len(experiments)} experiments for grid search.")