"""
This file contains specific functions for computing losses on the da_heads
file
"""

import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import consistency_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.poolers import Pooler
from ..utils import cat

class DALossComputation(object):
    """
    This class computes the DA loss.
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        self.pooler = pooler
        self.avgpool = nn.AvgPool2d(kernel_size=resolution, stride=resolution)
        
    def prepare_masks(self, targets):
        masks = []
        for targets_per_image in targets:
            is_source = targets_per_image.get_field('is_source')
            mask_per_image = is_source.new_ones(1, dtype=torch.uint8) if is_source.any() else is_source.new_zeros(1, dtype=torch.uint8)
            masks.append(mask_per_image)
        return masks


    @staticmethod
    def select_to_align_category_idx(ins_cls_labels, one_domain_ins_num, classes_num):
        sourcedomain_ins_cls_labels = ins_cls_labels[:int(one_domain_ins_num)]
        targetdomain_ins_cls_labels = ins_cls_labels[int(one_domain_ins_num):]

        # select_category_id = None
        # select_category_num = 0
        # for cat_i in range(1, classes_num+1):
        #     sd_cat_i_num = torch.sum(sourcedomain_ins_cls_labels==cat_i)
        #     td_cat_i_num = torch.sum(targetdomain_ins_cls_labels==cat_i)
        #     if (sd_cat_i_num > 0) and (td_cat_i_num > 0):
        #         if (sd_cat_i_num+td_cat_i_num)>select_category_num:
        #             select_category_num = sd_cat_i_num+td_cat_i_num
        #             select_category_id = cat_i

        # final_select_idx = ins_cls_labels==select_category_id




        #不管什么类，只对齐前景类
        final_select_idx = ins_cls_labels!=0

        return final_select_idx.cuda().type(torch.cuda.FloatTensor)


    def __call__(self, da_img, da_ins, da_img_consist, da_ins_consist, da_ins_labels, targets, ins_cls_labels):
        """
        Arguments:
            da_img (list[Tensor])
            da_img_consist (list[Tensor])
            da_ins (Tensor)
            da_ins_consist (Tensor)
            da_ins_labels (Tensor)
            targets (list[BoxList])

        Returns:
            da_img_loss (Tensor)
            da_ins_loss (Tensor)
            da_consist_loss (Tensor)
        """

        #### identify the source domain instances and target domain instances of the same lesion class that need to be aligned in this step

        # Foreground instances in the source and target domains
        select_ins_idx = self.select_to_align_category_idx(ins_cls_labels, len(ins_cls_labels)/2, 16)

        masks = self.prepare_masks(targets)
        masks = torch.cat(masks, dim=0)

        da_img_flattened = []
        da_img_labels_flattened = []
        # for each feature level, permute the outputs to make them be in the
        # same format as the labels. Note that the labels are computed for
        # all feature levels concatenated, so we keep the same representation
        # for the image-level domain alignment
        for da_img_per_level in da_img:
            N, A, H, W = da_img_per_level.shape
            da_img_per_level = da_img_per_level.permute(0, 2, 3, 1)
            da_img_label_per_level = torch.zeros_like(da_img_per_level, dtype=torch.float32)
            da_img_label_per_level[masks, :] = 1

            da_img_per_level = da_img_per_level.reshape(N, -1)
            da_img_label_per_level = da_img_label_per_level.reshape(N, -1)
            
            da_img_flattened.append(da_img_per_level)
            da_img_labels_flattened.append(da_img_label_per_level)
            
        da_img_flattened = torch.cat(da_img_flattened, dim=0)
        da_img_labels_flattened = torch.cat(da_img_labels_flattened, dim=0)
        
        da_img_loss = F.binary_cross_entropy_with_logits(
            da_img_flattened, da_img_labels_flattened
        )
        # da_ins_loss = F.binary_cross_entropy_with_logits(
        #     torch.squeeze(da_ins), da_ins_labels.type(torch.cuda.FloatTensor)
        # )


        # Convert the labeled data to float
        da_ins_labels = da_ins_labels.type(torch.cuda.FloatTensor)

        # When calculating the da_ins loss for an instance, only compute the BCE loss on the output nodes corresponding to its class. Nodes with weights not equal to 1 are excluded from the loss calculation.
        weights_ins_cls_one_hot = torch.zeros((ins_cls_labels.shape[0], 17)).cuda().scatter_(1,ins_cls_labels[...,None], 1)

        # The domain discrimination label (source or target) for each class of each instance is (17, 1).
        ins_cls_one_hot_label = weights_ins_cls_one_hot*da_ins_labels[...,None]

        # Align instances of the same class across different domains, ignoring alignment for the background class.
        da_ins_loss = 17*F.binary_cross_entropy_with_logits(da_ins*select_ins_idx[...,None], ins_cls_one_hot_label*select_ins_idx[:,None], weights_ins_cls_one_hot)


        # ####Align only specific classes
        # da_ins_loss = F.binary_cross_entropy_with_logits(
        #     torch.squeeze(da_ins)*select_ins_idx, da_ins_labels.type(torch.cuda.FloatTensor)*select_ins_idx.type(torch.cuda.FloatTensor)
        # )

        da_consist_loss = consistency_loss(da_img_consist, da_ins_consist, da_ins_labels, size_average=True)

        return da_img_loss, da_ins_loss, da_consist_loss

def make_da_heads_loss_evaluator(cfg):
    loss_evaluator = DALossComputation(cfg)
    return loss_evaluator
