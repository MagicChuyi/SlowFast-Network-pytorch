import os
from typing import Union, Tuple, List, Optional
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from config.eval_config import EvalConfig
from backbone.base import Base as BackboneBase
from bbox1 import BBox
import pandas as pd
from roi.pooler import Pooler
from rpn.region_proposal_network import RegionProposalNetwork
from support.layer.nms import nms


class Model(nn.Module):

    def __init__(self, backbone: BackboneBase, num_classes: int, pooler_mode: Pooler.Mode,
                 anchor_ratios: List[Tuple[int, int]], anchor_sizes: List[int],
                 rpn_pre_nms_top_n: int, rpn_post_nms_top_n: int,
                 anchor_smooth_l1_loss_beta: Optional[float] = None, proposal_smooth_l1_loss_beta: Optional[float] = None):
        super().__init__()
        self.features, hidden, num_features_out, num_hidden_out = backbone.features()
        # self._bn_modules = nn.ModuleList([it for it in self.features.modules() if isinstance(it, nn.BatchNorm3d)] +
        #                                  [it for it in hidden.modules() if isinstance(it, nn.BatchNorm3d)])

        # NOTE: It's crucial to freeze batch normalization modules for few batches training, which can be done by following processes
        #       (1) Change mode to `eval`
        #       (2) Disable gradient (we move this process into `forward`)
        # for bn_module in self._bn_modules:
        #     for parameter in bn_module.parameters():
        #         #print("discard_bn")
        #         parameter.requires_grad = False

        self.detection = Model.Detection(pooler_mode, hidden, num_hidden_out, num_classes, proposal_smooth_l1_loss_beta)

    def forward(self, image_batch: Tensor,
                gt_bboxes_batch: Tensor = None, gt_classes_batch: Tensor = None, detector_bboxes_batch: Tensor = None):
        # disable gradient for each forwarding process just in case model was switched to `train` mode at any time
        # for bn_module in self._bn_modules:
        #     bn_module.eval()
        fast_feature,slow_feature = self.features(image_batch)
        batch_size, _, _,image_height, image_width = image_batch.shape
        _, _, _,features_height, features_width = slow_feature.shape

        #anchor_bboxes = self.rpn.generate_anchors(image_width, image_height, num_x_anchors=features_width, num_y_anchors=features_height).to(features).repeat(batch_size, 1, 1)

        if self.training:
            proposal_classes, proposal_class_losses = self.detection.forward(fast_feature,slow_feature, detector_bboxes_batch, gt_classes_batch, gt_bboxes_batch)
            return  proposal_class_losses
        else:
            #on work
            detector_bboxes_batch = detector_bboxes_batch.squeeze(dim=0)
            proposal_classes = self.detection.forward(fast_feature,slow_feature,detector_bboxes_batch)
            #print("debug:",detector_bboxes_batch.shape,proposal_classes.shape)
            detection_bboxes, detection_classes, detection_probs = self.detection.generate_detections(detector_bboxes_batch, proposal_classes, image_width, image_height)
            return detection_bboxes, detection_classes, detection_probs

    def save(self, path_to_checkpoints_dir: str, step: int, optimizer: Optimizer, scheduler: _LRScheduler) -> str:
        path_to_checkpoint = os.path.join(path_to_checkpoints_dir, 'model-{}.pth'.format(step))
        checkpoint = {
            'state_dict': self.state_dict(),
            'step': step,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        torch.save(checkpoint, path_to_checkpoint)
        return path_to_checkpoint
    #
    def load(self, path_to_checkpoint: str, optimizer: Optimizer = None, scheduler: _LRScheduler = None) -> 'Model':
        checkpoint = torch.load(path_to_checkpoint)
        self.load_state_dict(checkpoint['state_dict'])

        # model_dict = self.state_dict()
        # pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}  # filter out unnecessary keys
        # model_dict.update(pretrained_dict)
        # self.load_state_dict(model_dict)
        # torch.nn.DataParallel(self).cuda()
        #step = checkpoint['step']
        step=0
        # if optimizer is not None:
        #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # if scheduler is not None:
        #     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return step

    class Detection(nn.Module):

        def __init__(self, pooler_mode: Pooler.Mode, hidden: nn.Module, num_hidden_out: int, num_classes: int, proposal_smooth_l1_loss_beta: float):
            super().__init__()
            self._pooler_mode = pooler_mode
            self.hidden = hidden
            self.num_classes = num_classes
            self._proposal_class = nn.Linear(num_hidden_out, num_classes)
            # self._proposal_transformer = nn.Linear(num_hidden_out, num_classes * 4)
            # self._proposal_smooth_l1_loss_beta = proposal_smooth_l1_loss_beta
            # self._transformer_normalize_mean = torch.tensor([0., 0., 0., 0.], dtype=torch.float)
            # self._transformer_normalize_std = torch.tensor([.1, .1, .2, .2], dtype=torch.float)
        #working
        def forward(self,fast_feature,slow_feature, proposal_bboxes: Tensor,
                    gt_classes_batch: Optional[Tensor] = None, gt_bboxes_batch: Optional[Tensor] = None) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]:
            batch_size = fast_feature.shape[0]
            fast_feature=nn.AvgPool3d(kernel_size=(fast_feature.shape[2], 1, 1))(fast_feature).squeeze(2)
            #print("fast_feature:",fast_feature.shape)
            slow_feature = nn.AvgPool3d(kernel_size=(slow_feature.shape[2], 1, 1))(slow_feature).squeeze(2)
            feature=torch.cat([fast_feature, slow_feature],dim=1)
            if not self.training:
                assert batch_size==1
                # a=torch.arange(end=batch_size, dtype=torch.long, device=proposal_bboxes.device).view(-1, 1)
                # b=a.repeat(1, proposal_bboxes.shape[1])
                ######################################
                ########  @ FATAL ERROR @  ###############
                ######################################
                proposal_batch_indices = torch.arange(end=batch_size, dtype=torch.long, device=proposal_bboxes.device).view(-1, 1).repeat(1, proposal_bboxes.shape[0])[0]
                # pool_f = Pooler.apply(fast_feature, proposal_bboxes.view(-1, 4), proposal_batch_indices.view(-1), mode=self._pooler_mode)
                # pool_s = Pooler.apply(slow_feature, proposal_bboxes.view(-1, 4), proposal_batch_indices.view(-1), mode=self._pooler_mode)
                # # 空间池化，拼接
                # pool_f = nn.AdaptiveAvgPool3d((1, pool_f.shape[3], pool_f.shape[4]))(pool_f)
                # pool_s = nn.AdaptiveAvgPool3d((1, pool_s.shape[3], pool_s.shape[4]))(pool_s)
                # pool_f = pool_f.squeeze(2)
                # pool_s = pool_s.squeeze(2)
                # pool = torch.cat([pool_f, pool_s], dim=1)
                pool = Pooler.apply(feature, proposal_bboxes, proposal_batch_indices, mode=Pooler.Mode.POOLING)
                hidden = self.hidden(pool)
                proposal_classes = self._proposal_class(hidden)
                #proposal_classes = proposal_classes.view(batch_size, -1, proposal_classes.shape[-1])
                return proposal_classes
            else:
                #过滤掉补充的0

                # find labels for each `proposal_bboxes`
                ious = BBox.iou(proposal_bboxes, gt_bboxes_batch)
                proposal_max_ious, proposal_assignments = ious.max(dim=2)
                #_, proposal_which = ious.max(dim=1)
                fg_masks = proposal_max_ious >= 0.85
                if len(fg_masks.nonzero()) > 0:
                    #fg_masks.nonzero()[:, 0]是在获取batch
                    proposal_bboxes=proposal_bboxes[fg_masks.nonzero()[:, 0], fg_masks.nonzero()[:, 1]]
                    batch_indices=fg_masks.nonzero()[:, 0]
                    labels=gt_classes_batch[fg_masks.nonzero()[:, 0], proposal_assignments[fg_masks]]
                else:
                    print('bbox warning')
                    fg_masks = proposal_max_ious >= 0.5
                    proposal_bboxes = proposal_bboxes[fg_masks.nonzero()[:, 0], fg_masks.nonzero()[:, 1]]
                    batch_indices = fg_masks.nonzero()[:, 0]
                    labels = gt_classes_batch[fg_masks.nonzero()[:, 0], proposal_assignments[fg_masks]]
                # pool_f_pre= Pooler.apply(fast_feature, proposal_bboxes, batch_indices,mode=self._pooler_mode)
                # pool_s_pre= Pooler.apply(slow_feature, proposal_bboxes,batch_indices, mode=self._pooler_mode)
                # #空间池化，拼接
                # pool_f_a = nn.AdaptiveAvgPool3d((1, pool_f_pre.shape[3], pool_f_pre.shape[4]))(pool_f_pre)
                # pool_s_a = nn.AdaptiveAvgPool3d((1, pool_s_pre.shape[3], pool_s_pre.shape[4]))(pool_s_pre)
                # pool_f_s = pool_f_a.squeeze(2)
                # pool_s_s = pool_s_a.squeeze(2)
                # pool=torch.cat([pool_f_s, pool_s_s], dim=1)
                pool = Pooler.apply(feature, proposal_bboxes, batch_indices, mode=Pooler.Mode.POOLING)
                #sure 5
                #print("cls_labels:",labels)
                hidden = self.hidden(pool)
                proposal_classes = self._proposal_class(hidden)
                proposal_class_losses = self.loss(proposal_classes, labels,batch_size,batch_indices)

                return proposal_classes, proposal_class_losses

        def loss(self, proposal_classes: Tensor,gt_proposal_classes: Tensor, batch_size,batch_indices) -> Tuple[Tensor, Tensor]:
            # assert np.any(np.isnan(np.array(proposal_classes)))==False
            # assert np.any(np.isnan(np.array(gt_proposal_classes))) == False
            cross_entropies = torch.zeros(batch_size, dtype=torch.float, device=proposal_classes.device).cuda()
            #batch_indices=torch.tensor(batch_indices,dtype=torch.float)
            for batch_index in range(batch_size):
                selected_indices = (batch_indices == batch_index).nonzero().view(-1)
                input=proposal_classes[selected_indices]
                target=gt_proposal_classes[selected_indices]
                if torch.numel(input)==0 or torch.numel(target)==0:
                    #print("Warning:None DATA:",batch_index)
                    continue
                assert torch.numel(input)==torch.numel(target)
                # print('input:',input)
                # print("input_sigmoid:", F.sigmoid(input))
                # print('target:',target)


                cross_entropy =F.multilabel_soft_margin_loss(input=proposal_classes[selected_indices],target=gt_proposal_classes[selected_indices],reduction="mean")

                # cross_entropy = F.binary_cross_entropy(input=F.sigmoid(proposal_classes[selected_indices]),
                #                                                target=gt_proposal_classes[selected_indices])
                torch.nn.MultiLabelSoftMarginLoss
                # print('cross_entropy:',cross_entropy)
                # print('cross_entropy:',cross_entropy)
                # cross_entropy = F.cross_entropy(input=proposal_classes[selected_indices],
                #                                 target=gt_proposal_classes[selected_indices])

                cross_entropies[batch_index] = cross_entropy
            return cross_entropies

        def generate_detections(self, proposal_bboxes: Tensor, proposal_classes: Tensor, image_width: int, image_height: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
            batch_size = proposal_bboxes.shape[0]
            #print("detection_bboxes:",proposal_bboxes)
            detection_bboxes = BBox.clip(proposal_bboxes, left=0, top=0, right=image_width, bottom=image_height)
            #print("detection_bboxes_clip:", detection_bboxes)
            detection_probs = F.sigmoid(proposal_classes)
            detection_zheng=detection_probs>=EvalConfig.KEEP
            all_detection_classes=[]
            all_detection_probs=[]
            for label,prob in zip(detection_zheng,detection_probs):
                detection_classes = []
                detection_p=[]
                for index,i in enumerate(label):
                    if i==1:
                        detection_classes.append(index)
                        detection_p.append(prob[index].item())
                all_detection_classes.append(detection_classes)
                all_detection_probs.append(detection_p)

            #print('all_detection_classes:',all_detection_classes)

            return detection_bboxes, all_detection_classes, all_detection_probs

