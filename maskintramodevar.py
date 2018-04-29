#!/usr/bin/env python3
"""
Visualize Intra mode variant masks in AV1 codec

Copyright (c) 2018 yohhoy
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt


MAX_SB_SIZE = 128
II_DC_PRED = 0
II_V_PRED = 1
II_H_PRED = 2
II_SMOOTH_PRED = 3

Ii_Weights_1d = [
  60, 58, 56, 54, 52, 50, 48, 47, 45, 44, 42, 41, 39, 38, 37, 35, 34, 33, 32,
  31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 22, 21, 20, 19, 19, 18, 18, 17, 16,
  16, 15, 15, 14, 14, 13, 13, 12, 12, 12, 11, 11, 10, 10, 10,  9,  9,  9,  8,
  8,  8,  8,  7,  7,  7,  7,  6,  6,  6,  6,  6,  5,  5,  5,  5,  5,  4,  4,
  4,  4,  4,  4,  4,  4,  3,  3,  3,  3,  3,  3,  3,  3,  3,  2,  2,  2,  2,
  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,
  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1
]


# Intra Mode Variant Mask Process
def mask_intramodevariant(w, h, interintra_mode):
    sizeScale = MAX_SB_SIZE // max(h, w)
    Mask = np.zeros((h, w), int)
    for i, j in itertools.product(range(h), range(w)):
        if interintra_mode == II_V_PRED:
            Mask[i][j] = Ii_Weights_1d[i * sizeScale]
        elif interintra_mode == II_H_PRED:
            Mask[i][j] = Ii_Weights_1d[j * sizeScale]
        elif interintra_mode == II_SMOOTH_PRED:
            Mask[i][j] = Ii_Weights_1d[min(i, j) * sizeScale]
        else:
            Mask[i][j] = 32
    return Mask


w = h = 16

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(4, 4))
fig.suptitle('Intra mode variant masks')

II_Names = {
  II_DC_PRED: 'II_DC_PRED', II_V_PRED: 'II_V_PRED',
  II_H_PRED: 'II_H_PRED', II_SMOOTH_PRED: 'II_SMOOTH_PRED'
}
for interintra_mode in range(4):
    mask = mask_intramodevariant(w, h, interintra_mode)
    ax = axs[interintra_mode // 2][interintra_mode % 2]
    ax.set_title(II_Names[interintra_mode])
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)
    ax.imshow(mask, cmap='Greys_r', vmin=0, vmax=64)

plt.savefig('mask-intrmodevar.png')
