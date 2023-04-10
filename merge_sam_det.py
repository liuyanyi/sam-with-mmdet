from mmdet.apis import init_detector
from mmdet.utils import register_all_modules
import torch
from mmengine.runner import get_state_dict, save_checkpoint
import fire


def merge_sam_det(sam_ckpt: str = './sam_vit_b_01ec64.pth',
                  rtm_ckpt: str = './rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth',
                  config: str = 'configs/rtm_tiny_sam_b.py',
                  output: str = './test_convert.pth'):

    register_all_modules()
    mm_model = init_detector(config, device='cpu', palette='coco')

    sam_state_dict = torch.load(sam_ckpt, map_location='cpu')
    det_state_dict = torch.load(rtm_ckpt, map_location='cpu')

    # rebuild the weight dict
    mm_weight = get_state_dict(mm_model)

    mm_weight_dict = {}
    for k, v in mm_weight.items():
        prefix = k.split('.')[0]
        if prefix not in mm_weight_dict:
            mm_weight_dict[prefix] = {}
        mm_weight_dict[prefix][k] = v

    for k in mm_weight.keys():
        prefix, weight_name = k.split('.', 1)
        if prefix == 'backbone':
            mm_weight[k] = sam_state_dict['image_encoder.'+weight_name]
        elif prefix == 'prompt_encoder':
            mm_weight[k] = sam_state_dict['prompt_encoder.'+weight_name]
        elif prefix == 'mask_decoder':
            mm_weight[k] = sam_state_dict['mask_decoder.'+weight_name]
        elif prefix == 'prompt_generator':
            mm_weight[k] = det_state_dict['state_dict'][weight_name]
        else:
            pass

    save_checkpoint(mm_weight, output)


if __name__ == '__main__':
    fire.Fire(merge_sam_det)
