from torch import nn
from frontend import STFT, MelFilter, PCENLayer, Log1p


class depthwise_separable_conv1d(nn.Module):
    def __init__(self, nin, nout, kernel, padding=0, stride=1):
        super(depthwise_separable_conv1d, self).__init__()
        self.depthwise = nn.Conv1d(nin, nin, kernel_size=kernel, padding=padding, stride=stride, groups=nin)
        self.pointwise = nn.Conv1d(nin, nout, kernel_size=1)
    def forward(self, x):
        out = self.depthwise(x.squeeze(1))
        out = self.pointwise(out)
        return out

class Dropout1d(nn.Module):
    def __init__(self, pdropout=.25):
        super(Dropout1d, self).__init__()
        self.dropout = nn.Dropout2d(pdropout)
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.dropout(x)
        return x.squeeze(-1)

PHYSETER_NFEAT = 128
PHYSETER_KERNEL = 7
BALAENOPTERA_NFEAT = 128
BALAENOPTERA_KERNEL = 5

get = {
    'physeter': {
        'weights': 'stft_depthwise_ovs_128_k7_r1.stdc',
        'fs': 50000,
        'archi': nn.Sequential(
            STFT(512, 256),
            MelFilter(50000, 512, 64, 2000, 25000),
            Log1p(trainable=True),
            depthwise_separable_conv1d(64, PHYSETER_NFEAT, PHYSETER_KERNEL, stride=2),
            nn.BatchNorm1d(PHYSETER_NFEAT),
            nn.LeakyReLU(),
            Dropout1d(),
            depthwise_separable_conv1d(PHYSETER_NFEAT, PHYSETER_NFEAT, PHYSETER_KERNEL, stride=2),
            nn.BatchNorm1d(PHYSETER_NFEAT),
            nn.LeakyReLU(),
            Dropout1d(),
            depthwise_separable_conv1d(PHYSETER_NFEAT, 1, PHYSETER_KERNEL, stride=2)
        ),
    },
    'balaenoptera': {
        'weights': 'dw_m128_brown_200Hzhps32_prod_w4_128_k5_r_sch97.stdc',
        'fs': 200,
        'archi': nn.Sequential(
            STFT(256, 32),
            MelFilter(200, 256, 128, 0, 100),
            Log1p(trainable=True),
            depthwise_separable_conv1d(128, BALAENOPTERA_NFEAT, kernel=BALAENOPTERA_KERNEL, padding=BALAENOPTERA_KERNEL//2),
            nn.BatchNorm1d(BALAENOPTERA_NFEAT),
            nn.LeakyReLU(),
            Dropout1d(),
            depthwise_separable_conv1d(BALAENOPTERA_NFEAT, BALAENOPTERA_NFEAT, kernel=BALAENOPTERA_KERNEL, padding=BALAENOPTERA_KERNEL//2),
            nn.BatchNorm1d(BALAENOPTERA_NFEAT),
            nn.LeakyReLU(),
            Dropout1d(),
            depthwise_separable_conv1d(BALAENOPTERA_NFEAT, 1, kernel=BALAENOPTERA_KERNEL, padding=BALAENOPTERA_KERNEL//2)
        )
    },
    'megaptera' : {
        'weights': 'sparrow_whales_train8C_2610_frontend2_conv1d_noaugm_bs32_lr.05_.stdc',
        'fs': 11025,
        'archi': nn.Sequential(
            nn.Sequential(
                STFT(512, 64),
                MelFilter(11025, 512, 64, 100, 3000),
                PCENLayer(64)
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
                nn.Conv2d(64, 256, (1, 9), bias=False),
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
    },
    'delphinid' : {
        'weights': 'sparrow_dolphin_train8_pcen_conv2d_noaugm_bs32_lr.005_.stdc',
        'fs': 96000,
        'archi': nn.Sequential(
            nn.Sequential(
                STFT(4096, 1024),
                MelFilter(96000, 4096, 128, 3000, 30000),
                PCENLayer(128)
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
                nn.Conv2d(64, 256, (1, 9), bias=False),
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
    },
    'orcinus': {
        'weights': 'train_fe76f_00085_85_0',
        'fs': 22050,
        'archi': nn.Sequential(
            nn.Sequential(
                STFT(1024, 128),
                MelFilter(22050, 1024, 80, 300, 11025),
                PCENLayer(80)
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
                nn.Dropout2d(p=.5),
                nn.Conv2d(64, 256, (1, 9), bias=False),  # for 80 bands
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.01),
                nn.Dropout2d(p=.5),
                nn.Conv2d(256, 64, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.Dropout2d(p=.5),
                nn.LeakyReLU(0.01),
                nn.Conv2d(64, 1, 1, bias=False),
            )
        )
    },
    'globicephala': {
        'weights': 'sparrow_dolphin_train8_pcen_conv2d_noaugm_bs32_lr_GLOBI.005_TRAIN_.stdc',
        'fs': 48000,
        'archi': nn.Sequential(
            nn.Sequential(
                STFT(2048, 512),
                MelFilter(48000, 2048, 128, 2000, 6000),
                PCENLayer(128)
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
                nn.Conv2d(64, 256, (1, 6), bias=False),  # for 80 bands
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.01),
                nn.Dropout(p=.5),
                nn.Conv2d(256, 64, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.01),
                nn.Dropout(p=.5),
                nn.Conv2d(64, 1, 1, bias=False),
                nn.MaxPool2d((6, 1)),
            )
        )
    }
}
