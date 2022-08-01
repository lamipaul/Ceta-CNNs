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
parser.add_argument('-batchsize', type=int, help='Amount of samples to process at a time', default=32),
parser.add_argument('-maxPool', type=bool, help='Wether to keep only the maximal prediction of a sample or the full sequence', default=True),

args = parser.parse_args()

meta_model = {
    'delphinid': {
        'stdc':'sparrow_dolphin_train8_pcen_conv2d_noaugm_bs32_lr.005_.stdc',
        'fs': 96000
    },
    'megaptera': {
        'stdc':'sparrow_whales_train8C_2610_frontend2_conv1d_noaugm_bs32_lr.05_.stdc',
        'fs':11025
    },
    'orcinus': '',
    'physeter': '',
    'balaenoptera': ''
}[args.specie]


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return data.dataloader.default_collate(batch) if len(batch) > 0 else None

norm = lambda arr: (arr - np.mean(arr) ) / np.std(arr)


def run(folder, stdcfile, model, fs, lensample, batch_size, maxPool):
    model.load_state_dict(torch.load(stdcfile))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    out = pd.DataFrame(columns=['fn', 'offset', 'pred'])
    fns, offsets, preds = [], [], []
    loader = data.DataLoader(Dataset(folder, fs, lensample), batch_size=batch_size, collate_fn=collate_fn, num_workers=8, prefetch_factor=4)
    with torch.no_grad():
        for x, meta in tqdm(loader):
            x = x.to(device)
            pred = model(x).cpu().detach().numpy()
            if maxPool:
                pred = np.maximum(pred)
            else:
                pred.reshape(len(x), -1)
            fns.extend(meta['fn'])
            offsets.extend(meta['offset'].numpy())
            preds.extend(pred)
    out.fn, out.offset, out.pred = fns, offsets, preds
    return out


class Dataset(data.Dataset):
    def __init__(self, folder, fs, lensample):
        super(Dataset, self)
        print('initializing dataset...')
        self.samples = []
        for fn in os.listdir(folder):
            try:
                duration = sf.info(folder.fn).duration
            except:
                print(f'Skipping {fn} (unable to read)')
                continue
            for offset in np.arange(0, duration+.01-lensample, lensample):
                self.samples.append({'fn':fn, 'offset':offset})
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


preds = run(args.audio_folder,
            meta_model['stdc'],
            models.get[args.specie],
            meta_model['fs'],
            batch_size=args.batch_size,
            lensample=args.lensample,
            maxPool=args.maxPool
        )

preds.to_pickle(args.pred_fn)
