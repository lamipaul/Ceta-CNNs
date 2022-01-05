from torch import nn
import torch
import numpy as np
from torch import tensor, nn, exp, log, ones, stack


class PCENLayer(nn.Module):
    def __init__(self, num_bands,
                 s=0.025,
                 alpha=.8,
                 delta=10.,
                 r=.25,
                 eps=1e-6,
                 init_smoother_from_data=True):
        super(PCENLayer, self).__init__()
        self.log_s = nn.Parameter( log(ones((1,1,num_bands)) * s))
        self.log_alpha = nn.Parameter( log(ones((1,1,num_bands,1)) * alpha))
        self.log_delta = nn.Parameter( log(ones((1,1,num_bands,1)) * delta))
        self.log_r = nn.Parameter( log(ones((1,1,num_bands,1)) * r))
        self.eps = tensor(eps)
        self.init_smoother_from_data = init_smoother_from_data

    def forward(self, input): # expected input (batch, channel, freqs, time)
        init = input[:,:,:,0]  # initialize the filter with the first frame
        if not self.init_smoother_from_data:
            init = torch.zeros(init.shape)  # initialize with zeros instead

        filtered = [init]
        for iframe in range(1, input.shape[-1]):
            filtered.append( (1-exp(self.log_s)) * filtered[iframe-1] + exp(self.log_s) * input[:,:,:,iframe] )
        filtered = stack(filtered).permute(1,2,3,0)

        # stable reformulation due to Vincent Lostanlen; original formula was:
        alpha, delta, r = exp(self.log_alpha), exp(self.log_delta), exp(self.log_r)
        return (input / (self.eps + filtered)**alpha + delta)**r - delta**r
#        filtered = exp(-alpha * (log(self.eps) + log(1 + filtered / self.eps)))
#        return (input * filtered + delta)**r - delta**r


def create_mel_filterbank(sample_rate, frame_len, num_bands, min_freq, max_freq,
                          norm=True, crop=False):
    """
    Creates a mel filterbank of `num_bands` triangular filters, with the first
    filter starting at `min_freq` and the last one stopping at `max_freq`.
    Returns the filterbank as a matrix suitable for a dot product against
    magnitude spectra created from samples at a sample rate of `sample_rate`
    with a window length of `frame_len` samples. If `norm`, will normalize
    each filter by its area. If `crop`, will exclude rows that exceed the
    maximum frequency and are therefore zero.
    """
    # mel-spaced peak frequencies
    min_mel = 1127 * np.log1p(min_freq / 700.0)
    max_mel = 1127 * np.log1p(max_freq / 700.0)
    peaks_mel = torch.linspace(min_mel, max_mel, num_bands + 2)
    peaks_hz = 700 * (torch.expm1(peaks_mel / 1127))
    peaks_bin = peaks_hz * frame_len / sample_rate

    # create filterbank
    input_bins = (frame_len // 2) + 1
    if crop:
        input_bins = min(input_bins,
                         int(np.ceil(max_freq * frame_len /
                                     float(sample_rate))))
    x = torch.arange(input_bins, dtype=peaks_bin.dtype)[:, np.newaxis]
    l, c, r = peaks_bin[0:-2], peaks_bin[1:-1], peaks_bin[2:]
    # triangles are the minimum of two linear functions f(x) = a*x + b
    # left side of triangles: f(l) = 0, f(c) = 1 -> a=1/(c-l), b=-a*l
    tri_left = (x - l) / (c - l)
    # right side of triangles: f(c) = 1, f(r) = 0 -> a=1/(c-r), b=-a*r
    tri_right = (x - r) / (c - r)
    # combine by taking the minimum of the left and right sides
    tri = torch.min(tri_left, tri_right)
    # and clip to only keep positive values
    filterbank = torch.clamp(tri, min=0)

    # normalize by area
    if norm:
        filterbank /= filterbank.sum(0)

    return filterbank


class MelFilter(nn.Module):
    def __init__(self, sample_rate, winsize, num_bands, min_freq, max_freq):
        super(MelFilter, self).__init__()
        melbank = create_mel_filterbank(sample_rate, winsize, num_bands,
                                        min_freq, max_freq, crop=True)
        self.register_buffer('bank', melbank)

    def forward(self, x):
        x = x.transpose(-1, -2)  # put fft bands last
        x = x[..., :self.bank.shape[0]]  # remove unneeded fft bands
        x = x.matmul(self.bank)  # turn fft bands into mel bands
        x = x.transpose(-1, -2)  # put time last
        return x

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        result = super(MelFilter, self).state_dict(destination, prefix, keep_vars)
        # remove all buffers; we use them as cached constants
        for k in self._buffers:
            del result[prefix + k]
        return result

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # ignore stored buffers for backwards compatibility
        for k in self._buffers:
            state_dict.pop(prefix + k, None)
        # temporarily hide the buffers; we do not want to restore them
        buffers = self._buffers
        self._buffers = {}
        result = super(MelFilter, self)._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._buffers = buffers
        return result

class STFT(nn.Module):
    def __init__(self, winsize, hopsize, complex=False):
        super(STFT, self).__init__()
        self.winsize = winsize
        self.hopsize = hopsize
        self.register_buffer('window',
                             torch.hann_window(winsize, periodic=False))
        self.complex = complex

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        result = super(STFT, self).state_dict(destination, prefix, keep_vars)
        # remove all buffers; we use them as cached constants
        for k in self._buffers:
            del result[prefix + k]
        return result

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # ignore stored buffers for backwards compatibility
        for k in self._buffers:
            state_dict.pop(prefix + k, None)
        # temporarily hide the buffers; we do not want to restore them
        buffers = self._buffers
        self._buffers = {}
        result = super(STFT, self)._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._buffers = buffers
        return result

    def forward(self, x):
        x = x.unsqueeze(1)
        # we want each channel to be treated separately, so we mash
        # up the channels and batch size and split them up afterwards
        batchsize, channels = x.shape[:2]
        x = x.reshape((-1,) + x.shape[2:])
        # we apply the STFT
        x = torch.stft(x, self.winsize, self.hopsize, window=self.window,
                       center=False, return_complex=False)
        # we compute magnitudes, if requested
        if not self.complex:
            x = x.norm(p=2, dim=-1)
        # restore original batchsize and channels in case we mashed them
        x = x.reshape((batchsize, channels, -1) + x.shape[2:]) #if channels > 1 else x.reshape((batchsize, -1) + x.shape[2:])
        return x


HB_model = nn.Sequential(nn.Sequential(
    STFT(512, 64),
    MelFilter(11025, 512, 64, 100, 3000),
    PCENLayer(64),
  ),
  nn.Sequential(
    nn.Conv2d(1, 32, 3, bias=False),
    nn.BatchNorm2d(32),
    nn.LeakyReLU(0.01),
    nn.Conv2d(32, 32, 3,bias=False),
    nn.BatchNorm2d(32),
    nn.MaxPool2d(3),
    nn.LeakyReLU(0.01),
    nn.Conv2d(32, 32, 3, bias=False),
    nn.BatchNorm2d(32),
    nn.LeakyReLU(0.01),
    nn.Conv2d(32, 32, 3, bias=False),
    nn.BatchNorm2d(32),
    nn.LeakyReLU(0.01),
    nn.Conv2d(32, 64, (16, 3), bias=False),
    nn.BatchNorm2d(64),
    nn.MaxPool2d((1,3)),
    nn.LeakyReLU(0.01),
    nn.Dropout(p=.5),
    nn.Conv2d(64, 256, (1, 9), bias=False),  # for 80 bands
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.01),
    nn.Dropout(p=.5),
    nn.Conv2d(256, 64, 1, bias=False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.01),
    nn.Dropout(p=.5),
    nn.Conv2d(64, 1, 1, bias=False)
  )
 )

delphi_model = nn.Sequential(nn.Sequential(
    STFT(4096, 1024),
    MelFilter(96000, 4096, 128, 3000, 30000),
    PCENLayer(128),
 ),
 nn.Sequential(
  nn.Conv2d(1, 32, 3, bias=False),
  nn.BatchNorm2d(32),
  nn.LeakyReLU(0.01),
  nn.Conv2d(32, 32, 3,bias=False),
  nn.BatchNorm2d(32),
  nn.MaxPool2d(3),
  nn.LeakyReLU(0.01),
  nn.Conv2d(32, 32, 3, bias=False),
  nn.BatchNorm2d(32),
  nn.LeakyReLU(0.01),
  nn.Conv2d(32, 32, 3, bias=False),
  nn.BatchNorm2d(32),
  nn.LeakyReLU(0.01),
  nn.Conv2d(32, 64, (19, 3), bias=False),
  nn.BatchNorm2d(64),
  nn.MaxPool2d(3),
  nn.LeakyReLU(0.01),
  nn.Dropout(p=.5),
  nn.Conv2d(64, 256, (1, 9), bias=False),  # for 80 bands
  nn.BatchNorm2d(256),
  nn.LeakyReLU(0.01),
  nn.Dropout(p=.5),
  nn.Conv2d(256, 64, 1, bias=False),
  nn.BatchNorm2d(64),
  nn.LeakyReLU(0.01),
  nn.Dropout(p=.5),
  nn.Conv2d(64, 1, 1, bias=False),
  nn.MaxPool2d((6, 1))
  )
)
