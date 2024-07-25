import pickle
import os
import torch
from torch.utils.data import Dataset
from pedalboard.io import AudioFile
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, file_path, apply_augmentations=False):
        self.apply_augmentations = apply_augmentations
        
        # Load the data from the file, either pickle or pt
        if file_path.endswith('.pt'):
            data = self.load_from_pt(file_path)
        elif file_path.endswith('.pkl'):
            data = self.load_from_pickle(file_path)

        # Compute sample rate and extract batches
        self.sample_rate = data['sample_rate']
        self.batches = [(dry, wet) for dry, wet in data['batches'] if dry.shape[0] != 1 and wet.shape[0] != 1]

        print(f"Removed {len(data['batches']) - len(self.batches)} samples from batches")
        print(f"Sample rate: {self.sample_rate} Hz")

    def __len__(self):
        return len(self.batches)
    
    def dequantize(self, audio, bits=16):
        """Add dequantization noise to the audio."""
        audio = audio.astype(np.float64)
        noise = np.random.rand(*audio.shape) / (2**(bits - 1))
        return audio + noise

    def random_crop(self, audio, target_length):
        """Randomly crop the audio to a specified length."""
        audio = audio.astype(np.float64)
        if audio.shape[-1] > target_length:
            start = np.random.randint(0, audio.shape[-1] - target_length)
            return audio[..., start:start+target_length]
        return audio
    
    def allpass_filter(self, audio):
        """Apply an allpass filter with random coefficients."""
        audio = audio.astype(np.float64)
        coeff = np.float64(np.random.rand() * 2 - 1)  # Random coefficient between -1 and 1
        
        output = np.zeros_like(audio, dtype=np.float64)
        x1, x2, y1, y2 = np.float64(0), np.float64(0), np.float64(0), np.float64(0)
        
        for i in range(audio.shape[-1]):
            x0 = audio[..., i]
            y0 = coeff * (x0 - y2) + x2
            
            output[..., i] = y0
            
            x2, x1 = x1, x0
            y2, y1 = y1, y0
        
        return output

    def __getitem__(self, idx):
        dry_audio, wet_audio = self.batches[idx]

        # Apply augmentations
        if self.apply_augmentations:
            dry_audio = self.dequantize(dry_audio)
            wet_audio = self.dequantize(wet_audio)
            
            # Ensure both audios have the same length after random crop
            min_length = min(dry_audio.shape[-1], wet_audio.shape[-1])
            dry_audio = self.random_crop(dry_audio, min_length)
            wet_audio = self.random_crop(wet_audio, min_length)
            
            dry_audio = self.allpass_filter(dry_audio)
            wet_audio = self.allpass_filter(wet_audio)

        return torch.tensor(dry_audio), torch.tensor(wet_audio)

    def get_sample_rate(self):
        return self.sample_rate
    
    @staticmethod
    def save_to_pickle(dry_audio_files, wet_audio_files, filename):
        def process_audio(file_path):
            with AudioFile(file_path) as f:
                audio = f.read(int(f.samplerate * f.duration))
                return audio, f.samplerate
            
        # Process the first file to get the sample rate
        _, sample_rate = process_audio(dry_audio_files[0])

        # Full audio files (dry and wet)
        batches = [(process_audio(dry)[0], process_audio(wet)[0]) for dry, wet in zip(dry_audio_files, wet_audio_files)]

        data = {
            'sample_rate': sample_rate,
            'batches': batches
        }

        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def save_to_pt(dry_audio_files, wet_audio_files, filename):
        def process_audio(file_path):
            with AudioFile(file_path) as f:
                audio = f.read(int(f.samplerate * f.duration))
                return audio, f.samplerate

        # Process the first file to get the sample rate
        _, sample_rate = process_audio(dry_audio_files[0])

        # Full audio files (dry and wet)
        batches = [(process_audio(dry)[0], process_audio(wet)[0]) for dry, wet in zip(dry_audio_files, wet_audio_files)]

        data = {
            'sample_rate': sample_rate,
            'batches': batches
        }

        torch.save(data, filename)

    @staticmethod
    def load_from_pickle(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data
    
    @staticmethod
    def load_from_pt(filename):
        data = torch.load(filename)
        return data