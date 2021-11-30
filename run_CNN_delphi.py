from model import delphi_model
from scipy import signal
import soundfile as sf
from torch import load, no_grad, tensor, device, cuda
from torch.utils import data
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('files', type=str, nargs='+')
parser.add_argument('-outfn', type=str, default='delphi_preds.pkl')
args = parser.parse_args()

stdc = 'sparrow_dolphin_train8_pcen_conv2d_noaugm_bs32_lr.005_.stdc'

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return data.dataloader.default_collate(batch) if len(batch) > 0 else None

def run(files, stdcfile, model, folder, fe=96000, lensample=5, batch_size=32):
    model.load_state_dict(load(stdcfile))
    model.eval()
    cuda0 = device('cuda' if cuda.is_available() else 'cpu')
    model.to(cuda0)

    out = pd.DataFrame(columns=['fn', 'offset', 'pred'])
    fns, offsets, preds = [], [], []
    with no_grad():
        for x, meta in tqdm(data.DataLoader(Dataset(files, folder, fe=fe, lensample=lensample), batch_size=batch_size, collate_fn=collate_fn, num_workers=8,prefetch_factor=4)):
            x = x.to(cuda0, non_blocking=True)
            pred = model(x)
            temp = pd.DataFrame().from_dict(meta)
            fns.extend(meta['fn'])
            offsets.extend(meta['offset'].numpy())
            preds.extend(pred.reshape(len(x), -1).cpu().detach().numpy())
#            print(meta, temp, pred.reshape(len(x), -1).shape)
#            temp['pred'] = pred.reshape(len(x), -1).cpu().detach()
#            preds = preds.append(temp, ignore_index=True)
    out.fn, out.offset, out.pred = fns, offsets, preds
    #preds.pred = preds.pred.apply(np.array)
    return out



class Dataset(data.Dataset):
    def __init__(self, fns, folder, fe=96000, lenfile=120, lensample=50): # lenfile and lensample in seconds
        super(Dataset, self)
        print('init dataset')
        self.samples = np.concatenate([[{'fn':fn, 'offset':offset} for offset in np.arange(0, sf.info(folder+fn).duration-lensample+1, lensample)] for fn in fns if sf.info(folder+fn).duration>10])
        self.lensample = lensample
        self.fe, self.folder = fe, folder

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        fs = sf.info(self.folder+sample['fn']).samplerate
        try:
            sig, fs = sf.read(self.folder+sample['fn'], start=max(0,int(sample['offset']*fs)), stop=int((sample['offset']+self.lensample)*fs))
        except:
            print('failed loading '+sample['fn'])
            return None
        if sig.ndim > 1:
            sig = sig[:,0]
        if len(sig) != fs*self.lensample:
            print('to short file '+sample['fn']+' \n'+str(sig.shape))
            return None
        if fs != self.fe:
            sig = signal.resample(sig, self.lensample*self.fe)

        sig = norm(sig)
        return tensor(sig).float(), sample

def norm(arr):
    return (arr - np.mean(arr) ) / np.std(arr)

preds = run(args.files, stdc, delphi_model, './', batch_size=3, lensample=50)
preds.to_pickle(args.outfn)
