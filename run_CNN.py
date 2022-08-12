import os
import torch
import models
from scipy import signal
import soundfile as sf
from torch.utils import data
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Run this script to use a CNN for inference on a folder of audio files.")
parser.add_argument('audio_folder', type=str, help='Path of the folder with audio files to process')
parser.add_argument('specie', type=str, help='Target specie to detect', choices=['megaptera', 'delphinid', 'orcinus', 'physeter', 'balaenoptera'])
parser.add_argument('pred_fn', type=str, help='Filename for the output table containing model predictions')
parser.add_argument('-lensample', type=float, help='Length of the signal excerpts to process (sec)', default=5),
parser.add_argument('-batch_size', type=int, help='Amount of samples to process at a time', default=32),
parser.add_argument('-maxPool', help='Wether to keep only the maximal prediction of a sample or the full sequence', action='store_true'),
parser.add_argument('-no-maxPool', dest='maxPool', action='store_false')
parser.set_defaults(maxPool=True)
args = parser.parse_args()

meta_model = {
    'delphinid': {
        'stdc': 'sparrow_dolphin_train8_pcen_conv2d_noaugm_bs32_lr.005_.stdc',
        'fs': 96000
    },
    'megaptera': {
        'stdc': 'sparrow_whales_train8C_2610_frontend2_conv1d_noaugm_bs32_lr.05_.stdc',
        'fs': 11025
    },
    'orcinus': '',
    'physeter': {
        'stdc': 'stft_depthwise_ovs_128_k7_r1.stdc',
        'fs': 50000
    },
    'balaenoptera': {
        'stdc': 'dw_m128_brown_200Hzhps32_prod_w4_128_k5_r_sch97.stdc',
        'fs': 200
    }
}[args.specie]


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return data.dataloader.default_collate(batch) if len(batch) > 0 else None

norm = lambda arr: (arr - np.mean(arr) ) / np.std(arr)

class Dataset(data.Dataset):
    def __init__(self, folder, fs, lensample):
        super(Dataset, self)
        print('initializing dataset...')
        self.samples = []
        for fn in os.listdir(folder):
            try:
                duration = sf.info(folder+fn).duration
            except:
                print(f'Skipping {fn} (unable to read as audio)')
                continue
            self.samples.extend([{'fn':fn, 'offset':offset} for offset in np.arange(0, duration+.01-lensample, lensample)])
        self.fs, self.folder, self.lensample = fs, folder, lensample

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        fs = sf.info(self.folder+sample['fn']).samplerate
        try:
            sig, fs = sf.read(self.folder+sample['fn'], start=int(sample['offset']*fs), stop=int((sample['offset']+self.lensample)*fs), always_2d=True)
        except:
            print('Failed loading '+sample['fn'])
            return None
        sig = sig[:,0]
        if fs != self.fs:
            sig = signal.resample(sig, self.lensample*self.fs)
        sig = norm(sig)
        return torch.tensor(sig).float(), sample


# prepare model
model = models.get[args.specie]
model.load_state_dict(torch.load(f"weights/{meta_model['stdc']}"))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# prepare data loader and output storage for predictions
loader = data.DataLoader(Dataset(args.audio_folder, meta_model['fs'], args.lensample), batch_size=args.batch_size, collate_fn=collate_fn, num_workers=8, prefetch_factor=4)
out = pd.DataFrame(columns=['filename', 'offset', 'prediction'])
fns, offsets, preds = [], [], []
if len(loader) == 0:
    print('Unable to open any audio file in the given folder')
    exit()

with torch.no_grad():
    for x, meta in tqdm(loader):
        x = x.to(device)
        pred = model(x).cpu().detach().numpy()
        if args.maxPool:
            pred = pred.max(axis=-1).reshape(len(x))
        else:
            pred = pred.reshape(len(x), -1)
        preds.extend(pred)
        fns.extend(meta['fn'])
        offsets.extend(meta['offset'].numpy())

out.filename, out.offset, out.prediction = fns, offsets, preds
out.to_pickle(args.pred_fn)
