
import argparse, numpy as np, nibabel as nib, torch
from medssl_fs.models.unet3d import UNet3D

def sliding_window(vol, model, patch=(96,96,96), overlap=0.5, device='cuda'):
    D,H,W = vol.shape; pd,ph,pw = patch
    sd = max(1, int(pd*(1-overlap))); sh = max(1, int(ph*(1-overlap))); sw = max(1, int(pw*(1-overlap)))
    prob = np.zeros((model.out.out_channels, D,H,W), dtype=np.float32)
    norm = np.zeros((1,D,H,W), dtype=np.float32)
    for z in range(0, max(1, D-pd+1), sd):
        for y in range(0, max(1, H-ph+1), sh):
            for x in range(0, max(1, W-pw+1), sw):
                p = vol[z:z+pd, y:y+ph, x:x+pw][None,None].astype(np.float32)
                with torch.no_grad():
                    t = torch.from_numpy(p).to(device)
                    logits = model(t)
                    pr = torch.softmax(logits, dim=1).cpu().numpy()[0]
                prob[:, z:z+pd, y:y+ph, x:x+pw] += pr
                norm[:, z:z+pd, y:y+ph, x:x+pw] += 1.0
    norm[norm==0] = 1.0
    return prob/norm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--classes', type=int, default=2)
    ap.add_argument('--base_ch', type=int, default=32)
    ap.add_argument('--patch', type=int, nargs=3, default=[96,96,96])
    args = ap.parse_args()

    img = nib.load(args.image); vol = img.get_fdata().astype(np.float32)
    vol = (vol - vol.mean())/(vol.std()+1e-8)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet3D(1, args.classes, args.base_ch).to(dev)
    model.load_state_dict(torch.load(args.checkpoint, map_location=dev))
    model.eval()

    prob = sliding_window(vol, model, tuple(args.patch), 0.5, dev)
    seg = prob.argmax(0).astype(np.uint8)
    nib.Nifti1Image(seg, img.affine, img.header).to_filename(args.out)
    print('Saved:', args.out)

if __name__ == '__main__':
    main()
