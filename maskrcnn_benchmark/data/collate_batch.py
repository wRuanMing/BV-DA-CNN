# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.structures.image_list import to_image_list


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    Domain-aware data organization:
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        
        sourcedomain_images = to_image_list(transposed_batch[0], self.size_divisible)
        targetdomain_images = to_image_list(transposed_batch[2], self.size_divisible)
        sourcedomain_targets = transposed_batch[1]
        targetdomain_targets = transposed_batch[3]
        img_ids = transposed_batch[4]
        return sourcedomain_images, targetdomain_images, sourcedomain_targets, targetdomain_targets, img_ids



# class BatchCollator(object):
#     """
#     From a list of samples from the dataset,
#     returns the batched images and targets.
#     This should be passed to the DataLoader
#     """

#     def __init__(self, size_divisible=0):
#         self.size_divisible = size_divisible

#     def __call__(self, batch):
#         transposed_batch = list(zip(*batch))
#         images = to_image_list(transposed_batch[0], self.size_divisible)
#         targets = transposed_batch[1]
#         img_ids = transposed_batch[2]
#         return images, targets, img_ids
