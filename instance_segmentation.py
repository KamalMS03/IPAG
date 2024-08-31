# Install Detectron2 for Windows
# https://medium.com/@yogeshkumarpilli/how-to-install-detectron2-on-windows-10-or-11-2021-aug-with-the-latest-build-v0-5-c7333909676f

from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import itertools as it
import torch

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

class Detectron2Model():
    def __init__(self, dataset='COCO', backbone='R_50') -> None:
        self.dataset = dataset
        self.backbone = backbone
        self.cfg = get_cfg()
        self.set_cfg()
        self.model = DefaultPredictor(self.cfg)

        self.pred_masks = None


    def set_yaml_file_name(self):
        lr_schedule = '3x'
        self.yaml_file = f"{self.dataset}-InstanceSegmentation/mask_rcnn_{self.backbone}_FPN_{lr_schedule}.yaml"

    def set_cfg(self):    
        self.cfg.MODEL.DEVICE = "cpu"
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        self.set_yaml_file_name()
        self.cfg.merge_from_file(model_zoo.get_config_file(self.yaml_file)) # 3x
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        # self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.yaml_file) # 3x
        try:
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.yaml_file) # 3x
        except:
            weight_link = f'https://dl.fbaipublicfiles.com/detectron2/{self.dataset}-InstanceSegmentation/'
            if self.dataset=='COCO':
                if self.backbone=='R_50': weight_link += 'mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl'
                if self.backbone=='X_101': weight_link += 'mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl'
        
            self.cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detectron2/LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/144219072/model_final_571f7c.pkl'
    
    def get_model(self):
        return self.model
    
    def predict(self, image):
        self.image = image
        outputs = self.model(image)
        self.instances = outputs["instances"]
        self.pred_masks = self.instances.pred_masks
        return self.pred_masks

class InstanceSegmenter(Detectron2Model):
    def __init__(self, dataset='COCO', backbone='R_50') -> None:
        super().__init__(dataset=dataset, backbone=backbone)

        self.combined_instances = None
        self.proposals = None

    def create_proposals(self, blur_ksize=100):
        proposals = list()
        
        image = self.image
        image = image.astype(int)

        masks = self.filter_masks()
        blurred = cv2.blur(image, (blur_ksize, blur_ksize))
        combined_instances = self.combine_instances(masks)

        for idx in range(len(combined_instances)):
            mask = combined_instances[idx]
            proposal = self.save_segment(idx, blurred, mask)
            proposals.append(proposal)

        self.proposals = proposals
        return proposals

    def filter_masks(self, thr=0.005):
        image_dim = self.image.shape[0], self.image.shape[1]
        masks = self.pred_masks

        filtered_masks = list()
        w, h = image_dim[0], image_dim[1]
        bg_mask = np.full((w, h), False)
        for mask in masks:
            bg_mask = np.logical_or(bg_mask, mask)
            mask_size = np.count_nonzero(mask)/(w*h)
            if mask_size >= thr: filtered_masks.append(mask)
        
        inversed_mask = np.logical_not(bg_mask)
        if type(inversed_mask).__module__ == np.__name__ : inversed_mask = torch.from_numpy(inversed_mask)
        
        filtered_masks.append(inversed_mask)

        # if there are small mask, we generate the mask list w/o a threshold
        if (len(filtered_masks)==1) and (len(masks)!=0): self.filter_masks(thr=0)
        return filtered_masks

    def combine_instances(self, masks=list()):
        combinations = self.combine_no_repeat(len(masks))
        combined_instances = self.union_masks(masks, combinations)
        self.combined_instances = combined_instances     
        return combined_instances

    def combine_no_repeat(self, no_masks=1):
        idxs = [i for i in range(no_masks)]
        combinations = list()
        counter_range=no_masks if no_masks==1 else no_masks-1 # execlude the combination that contains all masks, unless there is only one mask (full image, no instances)
        for counter in range(counter_range):
            length = counter+1
            combinations += list(it.combinations(idxs, length))
        return combinations

    def union_masks(self, masks=list(), combinations=list()):
        combined_instances = list()
        mask_shape = masks[0].shape
        for combination in combinations:
            combined_instance = np.full(mask_shape, False)
            for mask_idx in combination:
                combined_instance = np.logical_or(combined_instance, masks[mask_idx])
            combined_instances.append(combined_instance)
        return combined_instances

    def save_segment(self, idx, blurred, mask):
        image = self.image.copy()
        image_size = image.shape[0] * image.shape[1]
        mask_size = np.count_nonzero(mask)

        mask = mask.numpy().astype(bool)
        proposed_segment = blurred.copy()
        proposed_segment[mask] = image[mask]

        proposal = [idx, image_size, mask_size, proposed_segment, mask]

        return proposal