
from typing import Tuple, Union
from mmdet.models.detectors import BaseDetector
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.registry import MODELS
import numpy as np
from segment_anything.utils.amg import batch_iterator, batched_mask_to_box, calculate_stability_score
from torch import Tensor
import torch
from torch.nn import functional as F

from mmdet.models.task_modules import MlvlPointGenerator
from mmdet.structures.det_data_sample import SampleList
from mmengine.structures import InstanceData
from mmcv.ops import batched_nms


@MODELS.register_module()
class SAM(BaseDetector):

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 prompt_generator: OptConfigType = None,
                 prompt_encoder: OptConfigType = None,
                 mask_decoder: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)

        if prompt_generator is not None:
            self.prompt_generator = MODELS.build(prompt_generator)
            self.use_prompt_generator = True
        else:
            self.use_prompt_generator = False

        self.prompt_encoder = MODELS.build(prompt_encoder)
        self.mask_decoder = MODELS.build(mask_decoder)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> Union[dict, tuple]:
        raise NotImplementedError

    def generate_masks_by_points(self,
                                 image_embeddings: Tensor,
                                 points: Tensor,
                                 ori_shape: Tuple,
                                 resize_shape: Tuple) -> InstanceData:
        device = image_embeddings.device
        res = InstanceData(
            scores=torch.zeros(0, device=device),
            masks=torch.zeros(0, ori_shape[0], ori_shape[1], device=device),
            bboxes=torch.zeros(0, 4, device=device),
        )
        for (batch_points,) in batch_iterator(64, points):
            batch_labels = torch.ones(
                batch_points.shape[0], dtype=torch.int, device=device)
            batch_ipt = (batch_points[:, None, :], batch_labels[:, None])
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=batch_ipt,
                boxes=None,
                masks=None,
            )
            multimask_output = self.test_cfg.get("multimask_output", False)
            mask, iou = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            mask = self.postprocess_masks(mask, ori_shape, resize_shape)

            mask = mask.flatten(0, 1),
            iou = iou.flatten(0, 1),

            mask = mask[0]
            iou = iou[0]
            bbox = batched_mask_to_box(mask > 0.0)

            single_result = InstanceData(
                scores=iou,
                masks=mask > 0.0,
                bboxes=bbox,
            )

            if self.test_cfg.get("pred_iou_thresh", 0) > 0:
                keep_inds = iou > self.test_cfg.pred_iou_thresh
                single_result = single_result[keep_inds]

            stability_score = calculate_stability_score(
                mask[keep_inds], 0.0, self.test_cfg.stability_score_offset
            )
            if self.test_cfg.get("stability_score_thresh", 0) > 0:
                keep_mask = stability_score >= self.test_cfg.stability_score_thresh
                single_result = single_result[keep_mask]

            del mask
            del iou

            torch.cuda.empty_cache()

            res = res.cat([res, single_result])

        fake_label = torch.ones(
            res.bboxes.shape[0], dtype=torch.long, device=device)
        res.labels = fake_label
        return res

    def generate_masks_by_pred_instances(self,
                                         image_embeddings: Tensor,
                                         pred_instances: InstanceData,
                                         ori_shape: Tuple,
                                         resize_shape: Tuple) -> InstanceData:
        device = image_embeddings.device
        res = InstanceData(
            scores=torch.zeros(0, device=device),
            labels=torch.zeros(0, device=device).long(),
            masks=torch.zeros(0, ori_shape[0], ori_shape[1], device=device),
            bboxes=torch.zeros(0, 4, device=device),
        )

        for (pred_ins) in batch_iterator(64, pred_instances):
            labels = pred_ins[0].labels
            boxes = pred_ins[0].bboxes
            scores = pred_ins[0].scores

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=boxes,
                masks=None,
            )
            multimask_output = self.test_cfg.get("multimask_output", False)
            mask, iou = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            mask = self.postprocess_masks(mask, ori_shape, resize_shape)
            out_labels = labels.unsqueeze(1).expand(-1, mask.shape[1])
            out_scores = scores.unsqueeze(1).expand(-1, mask.shape[1])

            mask = mask.flatten(0, 1),
            labels = out_labels.flatten(0, 1)
            scores = out_scores.flatten(0, 1)
            iou = iou.flatten(0, 1),

            mask = mask[0]
            iou = iou[0]
            bbox = batched_mask_to_box(mask > 0.0)

            single_result = InstanceData(
                labels=labels,
                scores=scores,
                masks=mask > 0.0,
                bboxes=bbox,
            )

            if self.test_cfg.get("pred_iou_thresh", 0) > 0:
                keep_inds = iou > self.test_cfg.pred_iou_thresh
                single_result = single_result[keep_inds]

            stability_score = calculate_stability_score(
                mask[keep_inds], 0.0, self.test_cfg.stability_score_offset
            )
            if self.test_cfg.get("stability_score_thresh", 0) > 0:
                keep_mask = stability_score >= self.test_cfg.stability_score_thresh
                single_result = single_result[keep_mask]

            del mask
            del iou

            torch.cuda.empty_cache()

            res = res.cat([res, single_result])
        return res

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        x = self.extract_feat(batch_inputs)

        input_size = batch_data_samples[0].batch_input_shape
        ori_shape = batch_data_samples[0].ori_shape
        ori_shape_np = np.array(batch_data_samples[0].ori_shape, dtype=int)
        scale_factor_np = np.array(batch_data_samples[0].scale_factor)
        resize_shape = np.around(ori_shape_np * scale_factor_np).astype(int)

        resize_shape = tuple(resize_shape)

        results = None

        if self.use_prompt_generator:
            points = None
            # Generate Prompt Boxes
            # noneed to rescale, because sam will use the resized size, not the original size
            generated_prompt = self.prompt_generator.predict(batch_inputs,
                                                             batch_data_samples, rescale=False)
            pred_instances = generated_prompt[0].pred_instances
            results = self.generate_masks_by_pred_instances(
                x, pred_instances, ori_shape, resize_shape)
        else:
            # Generate points
            points_per_side = self.test_cfg.get("points_per_side", 32)
            if isinstance(points_per_side, tuple):
                pps_x, pps_y = points_per_side
            else:
                pps_x = points_per_side
                pps_y = points_per_side
            strides = input_size[0] // pps_x, input_size[1] // pps_y
            point_generator = MlvlPointGenerator(
                strides=[strides, ], offset=0.5)

            points = point_generator.grid_priors(
                [(pps_x, pps_y), ], device=x.device)
            points = points[0][:, :2]
            results = self.generate_masks_by_points(
                x, points, ori_shape, resize_shape)

        # Do nms if nms config is not None
        if self.test_cfg.nms is not None:
            _, keep_idxs = batched_nms(
                results.bboxes,
                results.scores,
                results.labels,
                self.test_cfg.nms)

            results = results[keep_idxs]

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, [results, ])
        return batch_data_samples

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        ori_shape: tuple,
        resize_shape: tuple,
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        # input_size = data_sample.batch_input_shape

        masks = F.interpolate(
            masks,
            (self.backbone.img_size, self.backbone.img_size),
            mode="bilinear",
            align_corners=False,
        )

        masks = masks[..., : resize_shape[0], : resize_shape[1]]
        masks = F.interpolate(masks, ori_shape,
                              mode="bilinear", align_corners=False)
        return masks

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def _forward(self, batch_inputs: Tensor, batch_data_samples=None):
        return super()._forward(batch_inputs, batch_data_samples)


def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)
