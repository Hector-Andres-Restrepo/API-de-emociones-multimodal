"""Arquitectura PyTorch de Emo-AffectNet (ResNet50 estilo Keras/VGGFace2 + LSTM).

Los nombres de parámetros están alineados con los .pt publicados, verificados con
scripts/dump_keys.py sobre FER_static_ResNet50_AffectNet.pt (322 claves):
  stem:   conv_layer_s2_same, batch_norm1
  bloque: conv1, batch_norm1, conv2, batch_norm2, conv3, batch_norm3, i_downsample.{0,1}
  cabeza: fc1 (2048->512), fc2 (512->7)

Supuestos que los nombres/formas NO pueden confirmar (heredados del ResNet50 de Keras con el
que se entrenó el modelo original en TensorFlow):
  * el stride de cada bloque va en la primera conv 1x1 (y en la de i_downsample);
  * BatchNorm con eps=1e-3 (valor por defecto de Keras);
  * max-pool 3x3 stride 2 sin relleno ("valid" en Keras).
Si las probabilidades visuales parecieran incoherentes, revise estos tres puntos.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

BN_EPS = 1e-3


class Conv2dSame(nn.Conv2d):
    """Convolución con relleno 'same' al estilo TensorFlow (relleno extra a la derecha/abajo)."""

    def forward(self, x):
        ih, iw = x.shape[-2:]
        kh, kw = self.weight.shape[-2:]
        sh, sw = self.stride
        ph = max((math.ceil(ih / sh) - 1) * sh + kh - ih, 0)
        pw = max((math.ceil(iw / sw) - 1) * sw + kw - iw, 0)
        if ph or pw:
            x = F.pad(x, [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super().__init__()
        self.conv1 = Conv2dSame(in_channels, out_channels, 1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.conv2 = Conv2dSame(out_channels, out_channels, 3, stride=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.conv3 = Conv2dSame(out_channels, out_channels * self.expansion, 1, stride=1, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion, eps=BN_EPS)
        self.relu = nn.ReLU()
        self.i_downsample = i_downsample

    def forward(self, x):
        identity = x if self.i_downsample is None else self.i_downsample(x)
        out = self.relu(self.batch_norm1(self.conv1(x)))
        out = self.relu(self.batch_norm2(self.conv2(out)))
        out = self.batch_norm3(self.conv3(out))
        return self.relu(out + identity)


class EmoBackbone(nn.Module):
    """ResNet50 [3,4,6,3] + GAP + fc1(2048->512) + ReLU + fc2(512->7)."""

    def __init__(self, num_classes=7):
        super().__init__()
        self.in_channels = 64
        self.conv_layer_s2_same = Conv2dSame(3, 64, 7, stride=2, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64, eps=BN_EPS)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)  # sin relleno
        self.layer1 = self._make_layer(3, 64, 1)
        self.layer2 = self._make_layer(4, 128, 2)
        self.layer3 = self._make_layer(6, 256, 2)
        self.layer4 = self._make_layer(3, 512, 2)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def _make_layer(self, blocks, planes, stride):
        down = nn.Sequential(
            nn.Conv2d(self.in_channels, planes * Bottleneck.expansion, 1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * Bottleneck.expansion, eps=BN_EPS),
        )
        layers = [Bottleneck(self.in_channels, planes, down, stride)]
        self.in_channels = planes * Bottleneck.expansion
        layers += [Bottleneck(self.in_channels, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def extract_features(self, x):
        x = self.max_pool(self.relu(self.batch_norm1(self.conv_layer_s2_same(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)
        return self.fc1(x)  # 512-d antes de ReLU

    def forward(self, x):
        feat = F.relu(self.extract_features(x))  # esta es la entrada del LSTM
        return self.fc2(feat), feat


class EmoLSTM(nn.Module):
    """Dos LSTM apiladas (512->512, 512->256) + lineal 256->7; clasifica con el último paso.
    Verificado: sus 10 parámetros coinciden con FER_dinamic_LSTM_Aff-Wild2.pt."""

    def __init__(self, num_classes=7):
        super().__init__()
        self.lstm1 = nn.LSTM(512, 512, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):  # x: (B, W, 512)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return self.fc(x[:, -1, :])
