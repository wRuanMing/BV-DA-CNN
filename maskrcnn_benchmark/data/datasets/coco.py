# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints


min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None, is_source= True
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms
        self.is_source = is_source

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)

        domain_labels = torch.ones_like(classes, dtype=torch.uint8) if self.is_source else torch.zeros_like(classes, dtype=torch.uint8)
        target.add_field("is_source", domain_labels)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data


class COCOSourceTargetDomainAdaptiveDatset(torchvision.datasets.coco.CocoSourceTargetDomainAdaptiveDetection):
    '''
    COCOSourceTargetDomainAdaptiveDatset,
    Each source domain image is randomly matched with a target domain image that contains bounding boxes of the same categories as those in the source domain image.
    For example: For the source domain image img001.jpg, which has bounding boxes of categories 1, 3, and 5, it will be randomly matched with an image from the target domain that also contains bounding boxes of categories 1, 3, and 5.
    '''
    def __init__(
        self, sourcedomain_ann_file, targetdomain_ann_file, sourcedomain_root, targetdomain_root, remove_images_without_annotations, sourcedomain_transforms=None, targetdomain_transforms=None):
        super(COCOSourceTargetDomainAdaptiveDatset, self).__init__(sourcedomain_root, targetdomain_root, sourcedomain_ann_file, targetdomain_ann_file)
        # sort indices for reproducible results
        self.sourcedomain_ids = sorted(self.sourcedomain_ids)
        self.targetdomain_ids = sorted(self.targetdomain_ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            # Remove redundant images from the source domain data.
            ids = []
            for img_id in self.sourcedomain_ids:
                ann_ids = self.sourcedomain_coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.sourcedomain_coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.sourcedomain_ids = ids

            # Remove redundant images from the target domain data.
            ids = []
            for img_id in self.targetdomain_ids:
                ann_ids = self.targetdomain_coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.targetdomain_coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.targetdomain_ids = ids

        self.sourcedomain_json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.sourcedomain_coco.getCatIds())
        }
        self.targetdomain_json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.targetdomain_coco.getCatIds())
        }

        self.sourcedomain_contiguous_category_id_to_json_id = {
            v: k for k, v in self.sourcedomain_json_category_id_to_contiguous_id.items()
        }
        self.targetdomain_contiguous_category_id_to_json_id = {
            v: k for k, v in self.targetdomain_json_category_id_to_contiguous_id.items()
        }

        self.sourcedomain_id_to_img_map = {k: v for k, v in enumerate(self.sourcedomain_ids)}
        self.targetdomain_id_to_img_map = {k: v for k, v in enumerate(self.targetdomain_ids)}

        self.sourcedomain_transforms = sourcedomain_transforms
        self.targetdomain_transforms = targetdomain_transforms

    def __getitem__(self, idx):
        sourcedomain_img, sourcedomain_anno, targetdomain_img, targetdomain_anno = super(COCOSourceTargetDomainAdaptiveDatset, self).__getitem__(idx)

        ######source domain
        # filter crowd annotations
        sourcedomain_anno = [obj for obj in sourcedomain_anno if obj["iscrowd"] == 0]

        sourcedomain_boxes = [obj["bbox"] for obj in sourcedomain_anno]
        sourcedomain_boxes = torch.as_tensor(sourcedomain_boxes).reshape(-1, 4)  # guard against no boxes
        sourcedomain_target = BoxList(sourcedomain_boxes, sourcedomain_img.size, mode="xywh").convert("xyxy")

        sourcedomain_classes = [obj["category_id"] for obj in sourcedomain_anno]
        sourcedomain_classes = [self.sourcedomain_json_category_id_to_contiguous_id[c] for c in sourcedomain_classes]
        sourcedomain_classes = torch.tensor(sourcedomain_classes)
        sourcedomain_target.add_field("labels", sourcedomain_classes)

        sourcedomain_masks = [obj["segmentation"] for obj in sourcedomain_anno]
        sourcedomain_masks = SegmentationMask(sourcedomain_masks, sourcedomain_img.size)
        sourcedomain_target.add_field("masks", sourcedomain_masks)

        sourcedomain_domain_labels = torch.ones_like(sourcedomain_classes, dtype=torch.uint8)
        sourcedomain_target.add_field("is_source", sourcedomain_domain_labels)

        if sourcedomain_anno and "keypoints" in sourcedomain_anno[0]:
            sourcedomain_keypoints = [obj["keypoints"] for obj in sourcedomain_anno]
            sourcedomain_keypoints = PersonKeypoints(sourcedomain_keypoints, sourcedomain_img.size)
            sourcedomain_target.add_field("keypoints", sourcedomain_keypoints)

        sourcedomain_target = sourcedomain_target.clip_to_image(remove_empty=True)

        if self.sourcedomain_transforms is not None:
            sourcedomain_img, sourcedomain_target = self.sourcedomain_transforms(sourcedomain_img, sourcedomain_target)

        ######target domain
        # filter crowd annotations
        targetdomain_anno = [obj for obj in targetdomain_anno if obj["iscrowd"] == 0]

        targetdomain_boxes = [obj["bbox"] for obj in targetdomain_anno]
        targetdomain_boxes = torch.as_tensor(targetdomain_boxes).reshape(-1, 4)  # guard against no boxes
        targetdomain_target = BoxList(targetdomain_boxes, targetdomain_img.size, mode="xywh").convert("xyxy")

        targetdomain_classes = [obj["category_id"] for obj in targetdomain_anno]
        targetdomain_classes = [self.targetdomain_json_category_id_to_contiguous_id[c] for c in targetdomain_classes]
        targetdomain_classes = torch.tensor(targetdomain_classes)
        targetdomain_target.add_field("labels", targetdomain_classes)

        targetdomain_masks = [obj["segmentation"] for obj in targetdomain_anno]
        targetdomain_masks = SegmentationMask(targetdomain_masks, targetdomain_img.size)
        targetdomain_target.add_field("masks", targetdomain_masks)

        targetdomain_domain_labels = torch.zeros_like(targetdomain_classes, dtype=torch.uint8)
        targetdomain_target.add_field("is_source", targetdomain_domain_labels)

        if targetdomain_anno and "keypoints" in targetdomain_anno[0]:
            targetdomain_keypoints = [obj["keypoints"] for obj in targetdomain_anno]
            targetdomain_keypoints = PersonKeypoints(targetdomain_keypoints, targetdomain_img.size)
            targetdomain_target.add_field("keypoints", targetdomain_keypoints)

        targetdomain_target = targetdomain_target.clip_to_image(remove_empty=True)

        if self.targetdomain_transforms is not None:
            targetdomain_img, targetdomain_target = self.targetdomain_transforms(targetdomain_img, targetdomain_target)

        return sourcedomain_img, sourcedomain_target, targetdomain_img, targetdomain_target, idx

    def get_img_info(self, index):
        sourcedomain_img_id = self.sourcedomain_id_to_img_map[index]
        sourcedomain_img_data = self.sourcedomain_coco.imgs[sourcedomain_img_id]

        # targetdomain_img_id = self.targetdomain_id_to_img_map[index]
        # targetdomain_img_data = self.targetdomain_coco.imgs[targetdomain_img_id]
        # return sourcedomain_img_data, targetdomain_img_data
        return sourcedomain_img_data