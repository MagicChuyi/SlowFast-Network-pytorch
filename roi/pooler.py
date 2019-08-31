from enum import Enum
import math
import torch
from torch import Tensor
from torch.nn import functional as F

from support.layer.roi_align import ROIAlign


class Pooler(object):

    class Mode(Enum):
        POOLING = 'pooling'
        ALIGN = 'align'

    OPTIONS = ['pooling', 'align']

    @staticmethod
    def apply(features: Tensor, proposal_bboxes: Tensor, proposal_batch_indices: Tensor, mode: Mode) -> Tensor:
        _, _, feature_map_height, feature_map_width = features.shape
        scale = 1 / 16
        output_size = (7, 7)
        # sure 2
        #print("proposal_batch_indices:",proposal_batch_indices)
        if mode == Pooler.Mode.POOLING:
            pool = []
            #print("debug_pooling:",proposal_batch_indices.shape)
            for (proposal_bbox, proposal_batch_index) in zip(proposal_bboxes, proposal_batch_indices):
                start_x = max(min(round(proposal_bbox[0].item() * scale), feature_map_width - 1), 0)      # [0, feature_map_width)
                start_y = max(min(round(proposal_bbox[1].item() * scale), feature_map_height - 1), 0)     # (0, feature_map_height]
                end_x = max(min(round(proposal_bbox[2].item() * scale) + 1, feature_map_width), 1)        # [0, feature_map_width)
                end_y = max(min(round(proposal_bbox[3].item() * scale) + 1, feature_map_height), 1)       # (0, feature_map_height]
                # sure 3
                #print("position:",start_x,start_y,end_x,end_y)
                h=end_y-start_y
                w=end_x-start_x
                if h<7:
                   change_h=math.ceil((7-h)/2)
                   start_y=max(start_y-change_h,0)
                   end_y=min(end_y+change_h,feature_map_height)
                if w<7:
                   change_w=math.ceil((7-w)/2)
                   start_x =max(start_x-change_w,0)
                   end_x = min(end_x+change_w,feature_map_width)
                # sure 4
                #print("changed_position:", start_x, start_y, end_x, end_y)
                roi_feature_map = features[proposal_batch_index, :, start_y:end_y, start_x:end_x]
                pool.append(F.adaptive_max_pool2d(input=roi_feature_map, output_size=output_size))
                shape=pool[-1].shape
            pool = torch.stack(pool, dim=0)
        elif mode == Pooler.Mode.ALIGN:
            pool = ROIAlign(output_size, spatial_scale=scale, sampling_ratio=0)(
                features,
                torch.cat([proposal_batch_indices.view(-1, 1).float(), proposal_bboxes], dim=1)
            )
        else:
            raise ValueError

        pool = F.max_pool2d(input=pool, kernel_size=2, stride=2)
        return pool

