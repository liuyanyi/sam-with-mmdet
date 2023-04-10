from mmdet.apis import init_detector
from mmdet.utils import register_all_modules
import torch
from mmengine.runner import weights_to_cpu, get_state_dict, save_checkpoint


def load_sam_checkpoint(sam_cpkt):
    """Load checkpoint from a file or URL.

    Args:
        sam_cpkt (str): The checkpoint file path or URL.

    Returns:
        dict: The loaded checkpoint.
    """
    checkpoint = torch.load(sam_cpkt, map_location='cpu')
    return checkpoint


if __name__ == '__main__':
    ckpt_sam_path = './sam_vit_b_01ec64.pth'
    out_file_path = './mm.pth'
    model_cfg_file = './configs/cfg.py'
    register_all_modules()
    mm_model = init_detector(model_cfg_file, device='cpu', palette='random')

    sam_ckpt = load_sam_checkpoint(ckpt_sam_path)

    print('sam_ckpt.keys():', sam_ckpt.keys())

    # 按模块区分
    sam_weight_dict = {}
    for k, v in sam_ckpt.items():
        prefix = k.split('.')[0]
        if prefix not in sam_weight_dict:
            sam_weight_dict[prefix] = {}
        sam_weight_dict[prefix][k] = v

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
            mm_weight[k] = sam_weight_dict['image_encoder']['image_encoder.'+weight_name]
        elif prefix == 'prompt_encoder':
            mm_weight[k] = sam_weight_dict['prompt_encoder']['prompt_encoder.'+weight_name]
        elif prefix == 'mask_decoder':
            mm_weight[k] = sam_weight_dict['mask_decoder']['mask_decoder.'+weight_name]
        else:
            pass


    save_checkpoint(mm_weight, './test_convert.pth')

