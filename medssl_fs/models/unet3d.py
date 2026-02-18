
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, base_ch=32):
        super().__init__()
        ch = [base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16]
        self.enc1 = ConvBlock(in_channels, ch[0]); self.down1 = nn.Conv3d(ch[0], ch[0], 2, 2)
        self.enc2 = ConvBlock(ch[0], ch[1]);       self.down2 = nn.Conv3d(ch[1], ch[1], 2, 2)
        self.enc3 = ConvBlock(ch[1], ch[2]);       self.down3 = nn.Conv3d(ch[2], ch[2], 2, 2)
        self.enc4 = ConvBlock(ch[2], ch[3]);       self.down4 = nn.Conv3d(ch[3], ch[3], 2, 2)
        self.bottleneck = ConvBlock(ch[3], ch[4])
        self.up4 = nn.ConvTranspose3d(ch[4], ch[3], 2, 2); self.dec4 = ConvBlock(ch[4], ch[3])
        self.up3 = nn.ConvTranspose3d(ch[3], ch[2], 2, 2); self.dec3 = ConvBlock(ch[3], ch[2])
        self.up2 = nn.ConvTranspose3d(ch[2], ch[1], 2, 2); self.dec2 = ConvBlock(ch[2], ch[1])
        self.up1 = nn.ConvTranspose3d(ch[1], ch[0], 2, 2); self.dec1 = ConvBlock(ch[1], ch[0])
        self.out = nn.Conv3d(ch[0], out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x); x = self.down1(e1)
        e2 = self.enc2(x); x = self.down2(e2)
        e3 = self.enc3(x); x = self.down3(e3)
        e4 = self.enc4(x); x = self.down4(e4)
        x = self.bottleneck(x)
        x = self.up4(x); x = torch.cat([x, e4], dim=1); x = self.dec4(x)
        x = self.up3(x); x = torch.cat([x, e3], dim=1); x = self.dec3(x)
        x = self.up2(x); x = torch.cat([x, e2], dim=1); x = self.dec2(x)
        x = self.up1(x); x = torch.cat([x, e1], dim=1); x = self.dec1(x)
        return self.out(x)
