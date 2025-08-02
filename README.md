# 2d image segmentation
Segformer implementation

## architecture

![Model Architecure](https://github.com/iag0g0mes/segment_net/blob/main/images/model.svg)

## backbones

#### Conv-based 

- resnet18
- mobilenetv3
- efficientnetb0
- deeplabv3_mobilenetv3

#### Transformer-based

- mobilevit
- deit3_small
- efficientformer
- levit
- segformerb0
- pitxs
- sam2_hiera

## FPN Aggregation 

- sum
- concat
- weighted_sum
- max_pool

## class-wise attention

- none
- spatial
- query
- class_channel
- se_channel

## decoder

- se_conv_interp
- depthwise_nn
- transformer

## loss

- dice
- focal_dice 
- cross_entropy
- focal_cross_entropy 
- lovasz_softmax
- boundary_dice
- hausdorff_dt_dice

## Datasets

- a2d2 [urban]
- rellis3d [off-road]
- rugd     [off-road]
- goose    [urban/off-road]
- bdd100k  [urban]
  
## run

```
  python run.py --cfg cfg/<nome do arquivo>.ini
```
