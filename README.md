# A simple demo for SAM+MMDetection

To achieve instance segmentation, I make a simple demo to show how to use [SAM](https://github.com/facebookresearch/segment-anything) with [MMDetection](https://github.com/open-mmlab/mmdetection)

I use the pretrained model of SAM and MMDetection, then merge them together. I only test it on RTMDet-l model, but it should be easy to use other models.

Only inputs with bs of 1 are currently supported, and evaluation on COCO have not been tested.

## Usage

1. Install mmdetection

    ``` bash
    pip install openmim
    mim install mmengine 'mmcv>=2.0.0' 'mmdet>=3.0.0'
    ```

2. Install segment-anything

    ``` bash
    pip install git+https://github.com/facebookresearch/segment-anything.git
    ```

3. Download the pretrained model vit-b and mmdet model, then use merge_sam_det to merge them.

    ``` bash
    python ./tools/merge_sam_det.py ./sam_vit_b_01ec64.pth ./rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth ./configs/rtm_l_sam_b.py ./merge.pth
    ```

4. Run the demo

    ``` bash
    python ./tools/demo.py ./000000001309.jpg configs/rtm_l_sam_b.py merge.pth
    ```

![result](/assets/result.png)

## Requirements

``` text
segment-anything
mmdetection 3.0.0
fire
```

## Citation

**Segment Anything**

```latex
@article{kirillov2023segany,
  title={Segment Anything}, 
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
