import os
import torch
import models
from scipy import signal, special
import soundfile as sf
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Run this script to use a CNN for the detection of cetacean vocalizations on a folder of audio files.")
parser.add_argument('audio_folder', type=str, help='Path of the folder with audio files to process')
parser.add_argument('specie', type=str, help='Target specie to detect', choices=['megaptera', 'delphinid', 'orcinus', 'physeter', 'balaenoptera', 'globicephala'])
parser.add_argument('-lensample', type=float, help='Length of the signal for each sample (in seconds)', default=5),
parser.add_argument('-batch_size', type=int, help='Amount of samples to process at a time (usefull for parallel computation using a GPU)', default=32),
parser.add_argument('-channel', type=int, help='Channel of the audio file to use in the model inference (starting from 0)', default=0)
parser.add_argument('-maxPool', help='Wether to keep only the maximal prediction of each sample or the full sequence', action='store_true'),
parser.add_argument('-no-maxPool', dest='maxPool', action='store_false')
parser.add_argument('-output_filename', type=str, help='Name of the output file for saving predictions', default='')
parser.set_defaults(maxPool=True)
args = parser.parse_args()

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if len(batch) > 0 else None

norm = lambda arr: (arr - np.mean(arr) ) / np.std(arr)

# Pytorch dataset class to load audio samples
class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(Dataset, self)
        self.samples = []
        for fn in tqdm(os.listdir(args.audio_folder), desc='Dataset initialization', leave=False):
            try:
                info = sf.info(os.path.join(args.audio_folder, fn))
                duration, fs = info.duration, info.samplerate
                self.samples.extend([{'fn':fn, 'offset':offset, 'fs':fs} for offset in np.arange(0, duration+.01 - args.lensample, args.lensample)])
            except:
                continue
            assert info.channels > args.channel, f"The desired channel is unavailable for the audio file {fn}"
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            sig, fs = sf.read(os.path.join(args.audio_folder, sample['fn']), start=int(sample['offset']*sample['fs']), stop=int((sample['offset']+args.lensample)*sample['fs']), always_2d=True)
        except:
            print('Failed loading '+sample['fn'])
            return None
        sig = sig[:, args.channel]
        if fs != models.get[args.specie]['fs']:
            sig = signal.resample(sig, int(args.lensample * models.get[args.specie]['fs']))
        sig = norm(sig)
        return torch.tensor(sig).float(), sample


# prepare model
model = models.get[args.specie]['archi']
model.load_state_dict(torch.load(f"{os.path.dirname(__file__)}/weights/{models.get[args.specie]['weights']}"))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# prepare data loader and output storage for predictions
loader = torch.utils.data.DataLoader(Dataset(),
                                     batch_size=args.batch_size, collate_fn=collate_fn, num_workers=8, prefetch_factor=4)
if len(loader) == 0:
    print(f'Unable to open any audio file in the given folder {args.audio_folder}')
    exit()

out = pd.DataFrame(columns=['filename', 'offset', 'prediction'])
fns, offsets, preds = [], [], []

# forward the model on each batch
with torch.no_grad():
    for x, meta in tqdm(loader, desc='Model inference'):
        x = x.to(device)
        pred = special.expit(model(x).cpu().detach().numpy())
        if args.maxPool:
            pred = pred.max(axis=-1).reshape(len(x))
        else:
            pred = pred.reshape(len(x), -1)
        preds.extend(pred)
        fns.extend(meta['fn'])
        offsets.extend(meta['offset'].numpy())

out.filename, out.offset, out.prediction = fns, offsets, preds
pred_fn = list(filter(lambda e: e!='', args.audio_folder.split('/')))[-1] + ('.csv' if args.maxPool else '.pkl') if args.output_filename == '' else args.output_filename
print(f'Saving results into {pred_fn}')
if pred_fn.endswith('csv'):
    out.to_csv(pred_fn, index=False)
else:
    out.to_pickle(pred_fn)
