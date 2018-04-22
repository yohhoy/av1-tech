#!/usr/bin/env python3
"""
Visualize Intra-predictions in AV1 codec

Copyright (c) 2018 yohhoy
"""
import itertools
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def Clip1(x):
    return max(0, min(255, x))


def Round2(x, n):
    if n == 0:
        return x
    return (x + (1 << (n - 1))) >> n


# index adjustment for AboveRow/LeftCol
def IDX(n):
    return n + 1


# PAETH_PRED: Basic Intra Prediction Process
def pred_peath(w, h, AboveRow, LeftCol):
    pred = np.zeros((h, w, 1), int)
    for i, j in itertools.product(range(h), range(w)):
        ar, lc, a0 = AboveRow[IDX(j)], LeftCol[IDX(i)], AboveRow[IDX(-1)]
        base = ar + lc - a0
        pLeft = np.abs(base - lc)
        pTop = np.abs(base - ar)
        pTopLeft = np.abs(base - a0)
        if pLeft <= pTop and pLeft <= pTopLeft:
            pred[i][j] = lc
        elif pTop <= pTopLeft:
            pred[i][j] = ar
        else:
            pred[i][j] = a0
    return pred


ANGLE_STEP = 3
Mode_To_Angle = [0, 90, 180, 45, 135, 113, 157, 203, 67, 0, 0, 0, 0]
Dr_Intra_Derivative = [
  0, 0, 0, 1023, 0, 0, 547, 0, 0, 372, 0, 0, 0, 0,
  273, 0, 0, 215, 0, 0, 178, 0, 0, 151, 0, 0, 132, 0, 0,
  116, 0, 0, 102, 0, 0, 0, 90, 0, 0, 80, 0, 0, 71, 0, 0,
  64, 0, 0, 57, 0, 0, 51, 0, 0, 45, 0, 0, 0, 40, 0, 0,
  35, 0, 0, 31, 0, 0, 27, 0, 0, 23, 0, 0, 19, 0, 0,
  15, 0, 0, 0, 0, 11, 0, 0, 7, 0, 0, 3, 0, 0
]


# Dnnn_PRED: Directional Intra Prediction Process
def pred_directional(w, h, AboveRow, LeftCol, mode, angleDelta):
    assert 1 <= mode <= 8
    pAngle = Mode_To_Angle[mode] + angleDelta * ANGLE_STEP
    upsampleAbove = upsampleLeft = 0

    # assume enable_intra_edge_filter == 0

    if 0 < pAngle < 90:
        dx = Dr_Intra_Derivative[pAngle]
    elif 90 < pAngle < 180:
        dx = Dr_Intra_Derivative[180 - pAngle]
    if 90 < pAngle < 180:
        dy = Dr_Intra_Derivative[pAngle - 90]
    elif 180 < pAngle < 270:
        dy = Dr_Intra_Derivative[270 - pAngle]

    pred = np.zeros((h, w, 1), int)
    if 0 < pAngle < 90:
        for i, j in itertools.product(range(h), range(w)):
            idx = (i + 1) * dx
            base = (idx >> (6 - upsampleAbove)) + (j << upsampleAbove)
            shift = ((idx << upsampleAbove) >> 1) & 0x1F
            maxBaseX = (w + h - 1) << upsampleAbove
            if base < maxBaseX:
                pred[i][j] = Clip1(Round2(AboveRow[IDX(base)] * (32 - shift) +
                                          AboveRow[IDX(base + 1)] * shift, 5))
            else:
                pred[i][j] = AboveRow[IDX(maxBaseX)]
    elif 90 < pAngle < 180:
        for i, j in itertools.product(range(h), range(w)):
            idx = (j << 6) - (i + 1) * dx
            base = idx >> (6 - upsampleAbove)
            if -(1 << upsampleAbove) <= base:
                shift = ((idx << upsampleAbove) >> 1) & 0x1F
                pred[i][j] = Clip1(Round2(AboveRow[IDX(base)] * (32 - shift) +
                                          AboveRow[IDX(base + 1)] * shift, 5))
            else:
                idx = (i << 6) - (j + 1) * dy
                base = idx >> (6 - upsampleLeft)
                shift = ((idx << upsampleLeft) >> 1) & 0x1F
                pred[i][j] = Clip1(Round2(LeftCol[IDX(base)] * (32 - shift) +
                                          LeftCol[IDX(base + 1)] * shift, 5))
    elif 180 < pAngle < 270:
        for i, j in itertools.product(range(h), range(w)):
            idx = (j + 1) * dy
            base = (idx >> (6 - upsampleLeft)) + (i << upsampleLeft)
            shift = ((idx << upsampleLeft) >> 1) & 0x1F
            maxBaseY = (w + h - 1) << upsampleLeft
            if base < maxBaseY:
                pred[i][j] = Clip1(Round2(LeftCol[IDX(base)] * (32 - shift) +
                                          LeftCol[IDX(base + 1)] * shift, 5))
            else:
                pred[i][j] = LeftCol[IDX(maxBaseY)]
    elif pAngle == 90:
        for i, j in itertools.product(range(h), range(w)):
            pred[i][j] = AboveRow[IDX(j)]
    elif pAngle == 180:
        for i, j in itertools.product(range(h), range(w)):
            pred[i][j] = LeftCol[IDX(i)]
    return pred


# DC_PRED: DC Intra Predication Process
def pred_DC(w, h, AboveRow, LeftCol):
    s = 0
    for k in range(h):
        s += LeftCol[IDX(k)]
    for k in range(w):
        s += AboveRow[IDX(k)]
    s += (w + h) >> 1
    avg = s // (w + h)
    pred = np.zeros((h, w, 1), int)
    pred[:] = avg
    return pred


Sm_Weights_Tx_4x4 = [255, 149,  85,  64]
Sm_Weights_Tx_8x8 = [255, 197, 146, 105,  73,  50,  37,  32]
Sm_Weights_Tx_16x16 = [
  255, 225, 196, 170, 145, 123, 102,  84,  68,  54,  43,  33,  26, 20, 17, 16]
Sm_Weights_Tx_32x32 = [
  255, 240, 225, 210, 196, 182, 169, 157, 145, 133, 122, 111, 101, 92, 83, 74,
  66,  59,  52,  45,  39,  34,  29,  25,  21,  17,  14,  12,  10,  9,  8,  8]
Sm_Weights_Tx_64x64 = [
  255, 248, 240, 233, 225, 218, 210, 203, 196, 189, 182, 176, 169, 163, 156,
  150, 144, 138, 133, 127, 121, 116, 111, 106, 101, 96, 91, 86, 82, 77, 73, 69,
  65, 61, 57, 54, 50, 47, 44, 41, 38, 35, 32, 29, 27, 25, 22, 20, 18, 16, 15,
  13, 12, 10, 9, 8, 7, 6, 6, 5, 5, 4, 4, 4]


def select_smWeights(n):
    if n == 4:
        return Sm_Weights_Tx_4x4
    elif n == 8:
        return Sm_Weights_Tx_8x8
    elif n == 16:
        return Sm_Weights_Tx_16x16
    elif n == 32:
        return Sm_Weights_Tx_32x32
    elif n == 64:
        return Sm_Weights_Tx_32x32


# SMOOTH_PRED: Smooth Intra Prediction Process
def pred_smooth(w, h, AboveRow, LeftCol):
    smWeightsX = select_smWeights(w)
    smWeightsY = select_smWeights(h)
    pred = np.zeros((h, w, 1), int)
    for i, j in itertools.product(range(h), range(w)):
        smoothPred = smWeightsY[i] * AboveRow[IDX(j)] + \
                     (256 - smWeightsY[i]) * LeftCol[IDX(h - 1)] + \
                     smWeightsX[j] * LeftCol[IDX(i)] + \
                     (256 - smWeightsX[j]) * AboveRow[IDX(w - 1)]
        pred[i][j] = Round2(smoothPred, 9)
    return pred


# SMOOTH_V_PRED: Smooth Intra Prediction Process
def pred_smooth_v(w, h, AboveRow, LeftCol):
    smWeights = select_smWeights(h)
    pred = np.zeros((h, w, 1), int)
    for i, j in itertools.product(range(h), range(w)):
        smoothPred = smWeights[i] * AboveRow[IDX(j)] + \
                     (256 - smWeights[i]) * LeftCol[IDX(h - 1)]
        pred[i][j] = Round2(smoothPred, 8)
    return pred


# SMOOTH_H_PRED: Smooth Intra Prediction Process
def pred_smooth_h(w, h, AboveRow, LeftCol):
    smWeights = select_smWeights(w)
    pred = np.zeros((h, w, 1), int)
    for i, j in itertools.product(range(h), range(w)):
        smoothPred = smWeights[j] * LeftCol[IDX(i)] + \
                     (256 - smWeights[j]) * AboveRow[IDX(w - 1)]
        pred[i][j] = Round2(smoothPred, 8)
    return pred


def draw_pred(pred, AboveRow, LeftCol, name):
    BSZ = 20
    sz = (LeftCol.shape[0] * BSZ, AboveRow.shape[0] * BSZ)
    img = Image.new('L', sz, (255))
    draw = ImageDraw.Draw(img)
    # draw pred[x][y]
    h, w = pred.shape[:2]
    for x, y in itertools.product(range(w), range(h)):
        rc = np.array([x + 1, y + 1, x + 2, y + 2]) * BSZ
        draw.rectangle(tuple(rc), fill=tuple(pred[y][x]))
    # draw AboveRow[x]
    for x in range(AboveRow.shape[0]):
        rc = np.array([x, 0, x + 1, 1]) * BSZ
        draw.rectangle(tuple(rc), fill=tuple(AboveRow[x]), outline=(0))
    # draw LeftCol[y]
    for y in range(LeftCol.shape[0]):
        rc = np.array([0, y, 1, y + 1]) * BSZ
        draw.rectangle(tuple(rc), fill=tuple(LeftCol[y]), outline=(0))
    img.save(name + '.png')


def line_gradation(sv, ev, step):
    return np.array([sv + (ev - sv) * k // (step - 1) for k in range(step)], int)


def line_cyclic(t, step):
    return np.array([[127 * (1 + math.cos(math.pi * k * 2 / t))] for k in range(step)], int)


w = h = 4

# gradation
anchor_bl = np.array([255])  # Bottom-Left
anchor_al = np.array([128])  # Above-Left
anchor_ar = np.array([0])    # Above-Right
AboveRow = line_gradation(anchor_al, anchor_ar, 1 + w + h)
LeftCol = line_gradation(anchor_al, anchor_bl, 1 + w + h)
# cyclic
#AboveRow = line_cyclic(w, 1 + w + h)
#LeftCol = line_cyclic(h, 1 + w + h)


# visualize Intra-predictions
pred = pred_DC(w, h, AboveRow, LeftCol)
draw_pred(pred, AboveRow, LeftCol, "intra-dc")        # mode = 0

for mode in range(1, 9):
    for deltaAngle in [-3, -2, -1, 0, 1, 2, 3]:
        pred = pred_directional(w, h, AboveRow, LeftCol, mode, deltaAngle)
        angle = Mode_To_Angle[mode] + deltaAngle * ANGLE_STEP
        draw_pred(pred, AboveRow, LeftCol, f'intra-d{angle:03d}')   # mode=1..8

pred = pred_smooth(w, h, AboveRow, LeftCol)
draw_pred(pred, AboveRow, LeftCol, "intra-smooth")    # mode=9

pred = pred_smooth_v(w, h, AboveRow, LeftCol)
draw_pred(pred, AboveRow, LeftCol, "intra-smooth-v")  # mode=10

pred = pred_smooth_h(w, h, AboveRow, LeftCol)
draw_pred(pred, AboveRow, LeftCol, "intra-smooth-h")  # mode=11

pred = pred_peath(w, h, AboveRow, LeftCol)
draw_pred(pred, AboveRow, LeftCol, "intra-peath")     # mode=12


# visualize directional Intra-predictions
fig, axs = plt.subplots(nrows=8, ncols=7, figsize=(6, 6))
fig.suptitle('Directional Intra Predictions')
fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05, wspace=0.05)
for ypos, mode in enumerate([3, 8, 1, 5, 4, 6, 2, 7]):  # 90, 180, 45, 135, 113, 157, 203, 67
    for delta in range(7):
        pred = pred_directional(w, h, AboveRow, LeftCol, mode, delta - 3)
        pred = np.reshape(pred, pred.shape[:2])
        ax = axs[ypos][delta]
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        ax.imshow(pred, cmap='Greys_r')
fig.savefig('intra-directional.png')
