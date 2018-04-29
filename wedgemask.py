#!/usr/bin/env python3
"""
Visualize Wedge masks in AV1 codec

Copyright (c) 2018 yohhoy
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt


BLOCK_8X8 = 3
BLOCK_16X16 = 6
BLOCK_SIZES = 22
MASK_MASTER_SIZE = 64
WEDGE_TYPES = 16
WEDGE_HORIZONTAL = 0
WEDGE_VERTICAL = 1
WEDGE_OBLIQUE27 = 2
WEDGE_OBLIQUE63 = 3
WEDGE_OBLIQUE117 = 4
WEDGE_OBLIQUE153 = 5


def Clip3(x, y, z):
    return max(x, min(y, z))


Num_4x4_Blocks_Wide = [
  1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8,
  16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16
]
Num_4x4_Blocks_High = [
  1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16,
  8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4
]
Block_Width = [w4 * 4 for w4 in Num_4x4_Blocks_Wide]
Block_Height = [h4 * 4 for h4 in Num_4x4_Blocks_High]
Wedge_Master_Oblique_Odd = [
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  2,  6,  18,
  37, 53, 60, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
  64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64
]
Wedge_Master_Oblique_Even = [
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  4,  11, 27,
  46, 58, 62, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
  64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64
]
Wedge_Master_Vertical = [
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  7,  21,
  43, 57, 62, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
  64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64
]
Wedge_Codebook = [
    [
        [WEDGE_OBLIQUE27, 4, 4],  [WEDGE_OBLIQUE63, 4, 4],
        [WEDGE_OBLIQUE117, 4, 4], [WEDGE_OBLIQUE153, 4, 4],
        [WEDGE_HORIZONTAL, 4, 2], [WEDGE_HORIZONTAL, 4, 4],
        [WEDGE_HORIZONTAL, 4, 6], [WEDGE_VERTICAL, 4, 4],
        [WEDGE_OBLIQUE27, 4, 2],  [WEDGE_OBLIQUE27, 4, 6],
        [WEDGE_OBLIQUE153, 4, 2], [WEDGE_OBLIQUE153, 4, 6],
        [WEDGE_OBLIQUE63, 2, 4],  [WEDGE_OBLIQUE63, 6, 4],
        [WEDGE_OBLIQUE117, 2, 4], [WEDGE_OBLIQUE117, 6, 4],
    ],
    [
        [WEDGE_OBLIQUE27, 4, 4],  [WEDGE_OBLIQUE63, 4, 4],
        [WEDGE_OBLIQUE117, 4, 4], [WEDGE_OBLIQUE153, 4, 4],
        [WEDGE_VERTICAL, 2, 4],   [WEDGE_VERTICAL, 4, 4],
        [WEDGE_VERTICAL, 6, 4],   [WEDGE_HORIZONTAL, 4, 4],
        [WEDGE_OBLIQUE27, 4, 2],  [WEDGE_OBLIQUE27, 4, 6],
        [WEDGE_OBLIQUE153, 4, 2], [WEDGE_OBLIQUE153, 4, 6],
        [WEDGE_OBLIQUE63, 2, 4],  [WEDGE_OBLIQUE63, 6, 4],
        [WEDGE_OBLIQUE117, 2, 4], [WEDGE_OBLIQUE117, 6, 4],
    ],
    [
        [WEDGE_OBLIQUE27, 4, 4],  [WEDGE_OBLIQUE63, 4, 4],
        [WEDGE_OBLIQUE117, 4, 4], [WEDGE_OBLIQUE153, 4, 4],
        [WEDGE_HORIZONTAL, 4, 2], [WEDGE_HORIZONTAL, 4, 6],
        [WEDGE_VERTICAL, 2, 4],   [WEDGE_VERTICAL, 6, 4],
        [WEDGE_OBLIQUE27, 4, 2],  [WEDGE_OBLIQUE27, 4, 6],
        [WEDGE_OBLIQUE153, 4, 2], [WEDGE_OBLIQUE153, 4, 6],
        [WEDGE_OBLIQUE63, 2, 4],  [WEDGE_OBLIQUE63, 6, 4],
        [WEDGE_OBLIQUE117, 2, 4], [WEDGE_OBLIQUE117, 6, 4],
    ]
]
Wedge_Bits = [
  0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 0,
  0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0
]


def block_shape(bsize):
    w4 = Num_4x4_Blocks_Wide[bsize]
    h4 = Num_4x4_Blocks_High[bsize]
    if h4 > w4:
        return 0
    elif h4 < w4:
        return 1
    else:
        return 2


def get_wedge_direction(bsize, index):
    return Wedge_Codebook[block_shape(bsize)][index][0]


def get_wedge_xoff(bsize, index):
    return Wedge_Codebook[block_shape(bsize)][index][1]


def get_wedge_yoff(bsize, index):
    return Wedge_Codebook[block_shape(bsize)][index][2]


w = MASK_MASTER_SIZE
h = MASK_MASTER_SIZE
MasterMask = np.zeros((6, h, w), int)
for j in range(w):
    shift = MASK_MASTER_SIZE // 4
    for i in range(0, h, 2):
        MasterMask[WEDGE_OBLIQUE63][i][j] = Wedge_Master_Oblique_Even[Clip3(0, MASK_MASTER_SIZE - 1, j - shift)]
        shift -= 1
        MasterMask[WEDGE_OBLIQUE63][i + 1][j] = Wedge_Master_Oblique_Odd[Clip3(0, MASK_MASTER_SIZE - 1, j - shift)]
        MasterMask[WEDGE_VERTICAL][i][j] = Wedge_Master_Vertical[j]
        MasterMask[WEDGE_VERTICAL][i + 1][j] = Wedge_Master_Vertical[j]

for j, i in itertools.product(range(w), range(h)):
    msk = MasterMask[WEDGE_OBLIQUE63][i][j]
    MasterMask[WEDGE_OBLIQUE27][j][i] = msk
    MasterMask[WEDGE_OBLIQUE117][i][w - 1 - j] = 64 - msk
    MasterMask[WEDGE_OBLIQUE153][w - 1 - j][i] = 64 - msk
    MasterMask[WEDGE_HORIZONTAL][j][i] = MasterMask[WEDGE_VERTICAL][i][j]

WedgeMasks = np.zeros((BLOCK_SIZES, 2, WEDGE_TYPES, h, w), int)
for bsize in range(BLOCK_8X8, BLOCK_SIZES):
    if Wedge_Bits[bsize] > 0:
        w = Block_Width[bsize]
        h = Block_Height[bsize]
        for wedge in range(WEDGE_TYPES):
            d = get_wedge_direction(bsize, wedge)
            xoff = MASK_MASTER_SIZE // 2 - ((get_wedge_xoff(bsize, wedge) * w) >> 3)
            yoff = MASK_MASTER_SIZE // 2 - ((get_wedge_yoff(bsize, wedge) * h) >> 3)
            s = 0
            for i in range(w):
                s += MasterMask[d][yoff][xoff+i]
            for i in range(1, h):
                s += MasterMask[d][yoff+i][xoff]
            avg = (s + (w + h - 1) // 2) // (w + h - 1)
            flipSign = 1 if (avg < 32) else 0
            for j, i in itertools.product(range(w), range(h)):
                WedgeMasks[bsize][flipSign][wedge][i][j] = MasterMask[d][yoff+i][xoff+j]
                WedgeMasks[bsize][1-flipSign][wedge][i][j] = 64 - MasterMask[d][yoff+i][xoff+j]


# Wedge Mask Process
def mask_wedge(MiSize, wedge_index, wedge_sign=0):
    w = Block_Width[MiSize]
    h = Block_Height[MiSize]
    mask = np.zeros((w, h), int)
    for j, i in itertools.product(range(w), range(h)):
        mask[j, i] = WedgeMasks[MiSize][wedge_sign][wedge_index][i][j]
    return mask


fig, axs = plt.subplots(nrows=4, ncols=4)
fig.suptitle('Wedge masks')
fig.subplots_adjust(hspace=0.4)

for wedge_index in range(WEDGE_TYPES):
    mask = mask_wedge(BLOCK_16X16, wedge_index)
    ax = axs[wedge_index // 4][wedge_index % 4]
    ax.set_title(f'w={wedge_index}')
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)
    ax.imshow(mask, cmap='Greys_r', vmin=0, vmax=64)

plt.savefig('wedge-mask.png')
