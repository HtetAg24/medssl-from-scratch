
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure

def dice_score(pred, gt, label=1):
    p = (pred == label).astype(np.uint8)
    g = (gt == label).astype(np.uint8)
    inter = (p & g).sum()
    denom = p.sum() + g.sum()
    return (2*inter/denom) if denom>0 else 1.0

def hd95(pred, gt, label=1):
    p = (pred == label).astype(bool)
    g = (gt == label).astype(bool)
    if p.sum()==0 and g.sum()==0: return 0.0
    if p.sum()==0 or g.sum()==0: return 1e6
    conn = generate_binary_structure(3,1)
    p_edge = p ^ binary_erosion(p, conn)
    g_edge = g ^ binary_erosion(g, conn)
    dt_p = distance_transform_edt(~p_edge)
    dt_g = distance_transform_edt(~g_edge)
    sds = np.concatenate([dt_p[g_edge], dt_g[p_edge]])
    return float(np.percentile(sds, 95))
