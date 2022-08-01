from torch import nn
from frontend import STFT, MelFilter, PCENLayer, Log1p



get = {
    'megaptera' : nn.Sequential(
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
    ),
    'delphinid' : nn.Sequential(
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
}
