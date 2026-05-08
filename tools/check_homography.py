import numpy as np
from pathlib import Path
p = Path('data/homography_university_train/homographies.npz')
print('exists', p.exists())
with np.load(p, allow_pickle=True) as d:
    hom = d['homographies']
    valid = d['valid']
    dp = d['drone_paths']
    print('shape', hom.shape, 'valid_sum', int(valid.sum()), 'len_dp', len(dp))
    print('nan_count', int(np.isnan(hom).sum()), 'inf_count', int(np.isinf(hom).sum()))
    norms = [float(np.linalg.norm(hom[i])) for i in range(min(10, hom.shape[0]))]
    print('sample_norms', norms)
    # inspect a sample homography
    for i in range(min(5, hom.shape[0])):
        print(i, 'valid', bool(valid[i]), hom[i])
