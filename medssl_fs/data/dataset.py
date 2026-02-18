
import random, numpy as np, nibabel as nib, torch
from torch.utils.data import Dataset

class NiftiPatchDataset(Dataset):
    """Text file with 'image_path,label_path' per line"""
    def __init__(self, list_file, patch_size=(96,96,96), foreground_ratio=0.5, normalize='zscore', seed=42):
        super().__init__()
        self.items = []
        with open(list_file) as f:
            for line in f:
                s=line.strip()
                if not s or s.startswith('#'): continue
                p=s.split(',')
                if len(p)==2: self.items.append((p[0], p[1]))
        self.ps = np.array(patch_size)
        self.fr = foreground_ratio
        self.normalize = normalize
        random.seed(seed)

    def _load(self, img_p, lbl_p):
        im = nib.load(img_p).get_fdata().astype(np.float32)
        la = nib.load(lbl_p).get_fdata().astype(np.int16)
        if self.normalize=='zscore':
            m, s = im.mean(), im.std() + 1e-8
            im = (im - m)/s
        return im, la

    def _rand_crop(self, vol, lbl):
        D,H,W = vol.shape; ps = self.ps
        if random.random()<self.fr and lbl.sum()>0:
            zs, ys, xs = np.where(lbl>0)
            i = random.randrange(len(zs)); cz, cy, cx = int(zs[i]), int(ys[i]), int(xs[i])
        else:
            cz, cy, cx = np.random.randint(0,D), np.random.randint(0,H), np.random.randint(0,W)
        z1 = max(0, min(cz - ps[0]//2, D-ps[0])); z2 = z1 + ps[0]
        y1 = max(0, min(cy - ps[1]//2, H-ps[1])); y2 = y1 + ps[1]
        x1 = max(0, min(cx - ps[2]//2, W-ps[2])); x2 = x1 + ps[2]
        return vol[z1:z2, y1:y2, x1:x2], lbl[z1:z2, y1:y2, x1:x2]

    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        im, la = self._load(*self.items[idx])
        im, la = self._rand_crop(im, la)
        return torch.from_numpy(im[None].copy()), torch.from_numpy(la.copy())
