from PIL import Image
import numpy as np, glob, os

for f in sorted(glob.glob('.runtime/_menu_shots/*.png')):
    im = np.asarray(Image.open(f).convert('L'), dtype=float)
    h, w = im.shape
    left = im[:, :w // 2]
    rows = left.mean(axis=1)
    thr = rows.mean() + rows.std() * 1.5
    bright = [i for i in range(h) if rows[i] > thr]
    blocks = []
    for i in bright:
        if blocks and i - blocks[-1][-1] <= 4:
            blocks[-1].append(i)
        else:
            blocks.append([i])
    print(os.path.basename(f), 'size', (w, h), '亮行块数', len(blocks))
    for b in blocks[:14]:
        print('    y %3d-%3d  峰值 %.0f  均值 %.0f' % (b[0], b[-1], max(rows[i] for i in b),
                                                      sum(rows[i] for i in b) / len(b)))
