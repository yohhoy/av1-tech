# av1-tech
Coding tools study in AOM AV1 video codec.

- `intrapred.py` visualize Intra-predictions
- `wedgemask.py` visualize Wedge masks


## Intra prediction
|`DC_PRED`|`V_PRED`|`H_PRED`|
|:-------:|:------:|:------:|
|![DC_PRED](image/intra-dc.png "intra-dc.png")|![V_PRED](image/intra-d090.png "intra-d090.png")|![H_PRED](image/intra-d180.png "intra-d180.png")|

|`D45_PRED`|`D135_PRED`|`D113_PRED`|
|:--------:|:---------:|:---------:|
|![D45_PRED](image/intra-d045.png "intra-d045.png")|![D135_PRED](image/intra-d135.png "intra-d135.png")|![D113_PRED](image/intra-d113.png "intra-d113.png")|

|`D157_PRED`|`D203_PRED`|`D67_PRED`|
|:---------:|:---------:|:--------:|
|![D157_PRED](image/intra-d157.png "intra-d157.png")|![D203_PRED](image/intra-d203.png "intra-d203.png")|![D67_PRED](image/intra-d067.png "intra-d067.png")|

|`SMOOTH_PRED`|`SMOOTH_V_PRED`|`SMOOTH_H_PRED`|
|:-----------:|:-------------:|:-------------:|
|![SMOOTH_PRED](image/intra-smooth.png "intra-smooth.png")|![SMOOTH_V_PRED](image/intra-smooth-v.png "intra-smooth-v.png")|![SMOOTH_H_PRED](image/intra-smooth-h.png "intra-smooth-h.png")|

|`PAETH_PRED`|
|:--------:|
|![PAETH_PRED](image/intra-paeth.png "intra-paeth.png")|


Directional Intra prediction (8 mode x 7 delta angle):
![Dnn_PRED](image/intra-directional.png "intra-directional.png")


## Wedge mask
![WedgeMask](image/wedge-mask.png "wedge-mask.png")


## License
MIT License
