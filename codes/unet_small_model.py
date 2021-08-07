import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import models


# 3x3 konvolucioni sloj
def conv3x3(in_: int, out: int) -> nn.Module:
    return nn.Conv2d(in_, out, 3, padding=1)

# Konvolucioni sloj, pa ReLU funkcija
class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int) -> None:
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.activation(x)
        return x

# 1 Dekoder blok, redjacemo takvih nekoliko, u sebi ima konvolucioni sloj i ReLu aktivacionu f-ju kao i enkoder
class DecoderBlock(nn.Module):
    def __init__(
        self, in_channels: int, middle_channels: int, out_channels: int
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(
                middle_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

# Definisanje strukture cele neuralne mreze
class UNet11(nn.Module):
    def __init__(self, num_filters: int = 32, pretrained: bool = True) -> None:

        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # Mi zelimo da ucitamo pretrenirane tezine, pa je default na pretrained=True
        self.encoder = models.vgg13(pretrained=pretrained).features

        # Iz vgg11 izvlacimo odredjene slojeve i njihove tezine
        self.relu = self.encoder[1]
        self.conv1 = self.encoder[0]
        self.conv2s = self.encoder[5]
        self.conv2 = self.encoder[7]
        self.conv3s = self.encoder[10]
        self.conv3 = self.encoder[12]

        # Centralni sloj gde se spajaju enkoder i dekoder
        self.center = DecoderBlock(
            num_filters * 8, num_filters * 8, num_filters * 4
        )
        self.dec3 = DecoderBlock(
            num_filters * (8 + 4), num_filters * 8, num_filters * 2
        )
        self.dec2 = DecoderBlock(
            num_filters * (4 + 2), num_filters * 4, num_filters * 1
        )
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)

        # Poslednji sloj dekodera
        self.final = nn.Conv2d(num_filters, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Ovde redjamo redom pozive odgovarajucih slojeva za enkoder i dekoder
        conv1 = self.relu(self.conv1(x))
        conv2s = self.relu(self.conv2s(self.pool(conv1)))
        conv2 = self.relu(self.conv2(self.pool(conv2s)))
        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        conv3 = self.relu(self.conv3(conv3s))

        center = self.center(self.pool(conv3))

        dec3 = self.dec3(torch.cat([center, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.final(dec1)