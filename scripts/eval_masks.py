
import argparse, nibabel as nib
from medssl_fs.utils.metrics import dice_score, hd95

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--label', type=int, default=1)
    args = ap.parse_args()
    p = nib.load(args.pred).get_fdata().astype('int16')
    g = nib.load(args.gt).get_fdata().astype('int16')
    print('Dice:', round(dice_score(p,g,args.label),4), 'HD95:', round(hd95(p,g,args.label),2), 'vox')

if __name__ == '__main__':
    main()
