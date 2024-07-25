import torch

from models.encoder import EncoderTCN
from models.decoder import DecoderTCN
from utilities.training import train
from utilities.evaluate import evaluate
from torch.utils.data import  random_split, DataLoader
from utilities.dataset import AudioDataset
from utilities.metrics import spectral_distance
from utils.config import load_params
import os
import socket
import time
import random
import numpy as np
import shutil
from tensorboard.plugins.hparams import api as hp
from utils.save_logs import CustomSummaryWriter
import os
from utils.config import load_params
from utils.config import flatten_dict
from utils.save_logs import copy_tensorboard_log

def prepare_device(request):
    if request == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS device")
        else:
            device = torch.device("cpu")
            print("MPS requested but not available. Using CPU device")
    elif request == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA device")
        else:
            device = torch.device("cpu")
            print("CUDA requested but not available. Using CPU device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device

# Main
def main():
    # Get the path to the tensorboard directory
    tustu_logs_path = os.environ.get('TUSTU_LOGS_PATH')
    if tustu_logs_path is None:
        raise EnvironmentError("The environment variable ‘TUSTU_LOGS_PATH’ is not set.")
    default_dir = os.environ.get('DEFAULT_DIR')
    if default_dir is None: 
        raise EnvironmentError("The environment variable 'DEFAULT_DIR' is not set.")
    experiment_name = os.environ.get('DVC_EXP_NAME', 'default_experiment')
    if experiment_name is None:
        raise EnvironmentError(r"The environment variable 'DVC_EXP_NAME' is not set.")
    
    tensorboard_path = os.path.join(default_dir, tustu_logs_path, 'tensorboard', experiment_name)

    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    # Create a SummaryWriter object to write the tensorboard logs
    writer = CustomSummaryWriter(log_dir=tensorboard_path)

    # Load the parameters from the config file
    params = load_params()

    # Write params to tensorboard
    metrics = {}
    params_dict = params.to_dict()
    params_dict = flatten_dict(params_dict)
    writer.add_hparams(hparam_dict=params_dict, metric_dict=metrics, run_name=tensorboard_path)

    n_inputs = params.train.n_inputs # number of input channels, to use if not using the multiband decomposition
    n_bands = params.train.n_bands # number of bands for the multband decomposition
    latent_dim = params.train.latent_dim # Dimension of the latent space
    n_epochs = params.train.n_epochs # number of epochs
    batch_size= params.train.batch_size # batch size
    kernel_size = params.train.kernel_size # kernel size
    n_blocks = params.train.n_blocks # Number of Encoder and Decoder blocks
    dilation_growth = params.train.dilation_growth # dilation growth
    n_channels = params.train.n_channels # number of initial channels
    lr = params.train.lr # learning rate
    use_kl = params.train.use_kl # use kl divergence, That is to say, the latent space is a normal distribution
    device_request = params.train.device
    random_seed = params.general.random_seed # seed
    input_file = params.train.input_file

    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    device = prepare_device(device_request)

    # Build the model
    encoder = EncoderTCN(
        n_inputs=n_bands,
        kernel_size=kernel_size, 
        n_blocks=n_blocks, 
        dilation_growth=dilation_growth, 
        n_channels=n_channels,
        latent_dim=latent_dim,
        use_kl=use_kl)
    
    decoder = DecoderTCN(
        n_outputs=n_bands,
        kernel_size=kernel_size,
        n_blocks=n_blocks, 
        dilation_growth=dilation_growth, 
        n_channels=n_channels,
        latent_dim=latent_dim,
        use_kl=use_kl)
    
    # setup loss function, optimizer, and scheduler
    criterion = spectral_distance

    # Setup optimizer
    params = list(encoder.parameters())
    params += list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr, (0.5, 0.9))

    # Create the dataset
    full_dataset = AudioDataset(input_file, apply_augmentations=False)

    # Define the sizes of your splits
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    # Get the sample rate
    sample_rate = full_dataset.get_sample_rate()

    # Create the splits
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )

    # Create the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Train the model
    train(encoder, decoder, train_loader, val_loader, criterion, optimizer, tensorboard_writer=writer, num_epochs=n_epochs, device= device, n_bands=n_bands, use_kl=use_kl, sample_rate=sample_rate)

    # Evaluate the model
    evaluate(encoder, decoder, test_loader, criterion, writer, device, n_bands, use_kl, sample_rate)

    if not os.path.exists('exp-logs/'):
        os.makedirs('exp-logs/')

    writer.close()
    copy_tensorboard_log(tensorboard_path, experiment_name)

    print("Done!")

if __name__ == "__main__":
    main()