import itertools
import subprocess
import os 

# Submit experiment for hyperparameter combination
def submit_batch_job(index, latent_dim, kernel_size):
    # Set dynamic parameters for the batch job as environment variables
    # But dont forget to add the os.environ to the new environment variables otherwise the PATH is not found
    env = {
        **os.environ,
        "EXP_PARAMS": f"-S train.latent_dim={latent_dim} -S train.kernel_size={kernel_size}",
        "INDEX": str(index)
    }
    # Run sbatch command with the environment variables as bash! subprocess! command (otherwise module not found)
    subprocess.run(['/usr/bin/bash', '-c', 'sbatch batchjob.sh'], env=env)

if __name__ == "__main__":
    latent_dim_list = [128, 64, 32]
    kernel_size_list = [13, 16, 20]
    for index,(latent_dim, kernel_size) in enumerate(itertools.product(latent_dim_list, kernel_size_list)):
        submit_batch_job(index,latent_dim,kernel_size)
