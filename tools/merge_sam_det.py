import os
import sys

import fire
import torch
from mmdet.apis import init_detector
from mmdet.utils import register_all_modules
from mmdet.evaluation import get_classes
from mmengine.runner import get_state_dict, save_checkpoint

# get absulute path of upper directory
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# add upper directory to sys.path. avoid import error
sys.path.append(path)


def merge_sam_det(sam_ckpt: str = 'sam_vit_h_4b8939.pth',
                #   det_config: str = '/workspace/mm_sam/projects/configs/vhr10/rtmdet-l.py',
                  det_ckpt: str = '/workspace/mm_sam/work_dirs/rtmdet-l/epoch_24.pth',
                  config: str = '/workspace/mm_sam/projects/configs/rtm_l_sam_h.py',
                  output: str = './huge.pth'):

    register_all_modules()
    mm_model = init_detector(config, device='cpu')
    # det_model = init_detector(det_config, checkpoint=det_ckpt, device='cpu')

    sam_state_dict = torch.load(sam_ckpt, map_location='cpu')
    det_state_dict = torch.load(det_ckpt, map_location='cpu')
    neck_identy_level = mm_model.neck.identy_level

    # rebuild the weight dict
    mm_weight = get_state_dict(mm_model)

    for k in mm_weight.keys():
        prefix, weight_name = k.split('.', 1)
        if prefix == 'backbone':
            mm_weight[k] = sam_state_dict['image_encoder.'+weight_name]
        elif prefix == 'neck':
            split_name = weight_name.split('.')

            if split_name[1] == str(neck_identy_level):
                convert_map = {
                    f'lateral_convs.{split_name[1]}.conv': 'neck.0.',
                    f'lateral_convs.{split_name[1]}.ln': 'neck.1.',
                    f'fpn_convs.{split_name[1]}.conv': 'neck.2.',
                    f'fpn_convs.{split_name[1]}.ln': 'neck.3.'
                }
                start = split_name[0] +'.'+ split_name[1] +'.'+ split_name[2]
                dest = convert_map[start]
                if dest is None:
                    continue
                mm_weight[k] = sam_state_dict['image_encoder.'+dest+split_name[3]]
        elif prefix == 'prompt_encoder':
            mm_weight[k] = sam_state_dict['prompt_encoder.'+weight_name]
        elif prefix == 'mask_decoder':
            mm_weight[k] = sam_state_dict['mask_decoder.'+weight_name]
        elif prefix == 'prompt_generator':
            mm_weight[k] = det_state_dict['state_dict'][weight_name]
        else:
            pass

    
    dataset_meta = {'classes': get_classes('coco')}
    if 'meta' in det_state_dict:
        dataset_meta = det_state_dict['meta']['dataset_meta']
    final_checkpoint = {
        'meta': {
            'dataset_meta': dataset_meta,
        },
        'state_dict': mm_weight
    }
    save_checkpoint(final_checkpoint, output)


if __name__ == '__main__':
    fire.Fire(merge_sam_det)
