
import os, argparse, time, torch
from torch.utils.data import DataLoader
from medssl_fs.models.unet3d import UNet3D
from medssl_fs.data.dataset import NiftiPatchDataset
from medssl_fs.losses.dice_ce import DiceCELoss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_list', required=True)
    ap.add_argument('--val_list', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--patch', type=int, nargs=3, default=[96,96,96])
    ap.add_argument('--classes', type=int, default=2)
    ap.add_argument('--base_ch', type=int, default=32)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = NiftiPatchDataset(args.train_list, patch_size=args.patch)
    val_ds   = NiftiPatchDataset(args.val_list,   patch_size=args.patch)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=2)

    model = UNet3D(1, args.classes, args.base_ch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = DiceCELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=dev.type=='cuda')

    best = 1e9
    for ep in range(1, args.epochs+1):
        model.train(); t0=time.time(); tr=0.0
        for x,y in train_loader:
            x=x.to(dev, non_blocking=True); y=y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=dev.type=='cuda'):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tr += loss.item()
        tr /= max(1,len(train_loader))

        model.eval(); va=0.0
        with torch.no_grad():
            for x,y in val_loader:
                x=x.to(dev); y=y.to(dev)
                with torch.cuda.amp.autocast(enabled=dev.type=='cuda'):
                    logits = model(x)
                    loss = loss_fn(logits, y)
                va += loss.item()
        va /= max(1,len(val_loader))
        print(f"[{ep:03d}] train={tr:.4f}  val={va:.4f}  time={time.time()-t0:.1f}s")
        if va<best:
            best=va
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'unet3d_best.pth'))
    print('Best val loss:', best)

if __name__ == '__main__':
    main()
