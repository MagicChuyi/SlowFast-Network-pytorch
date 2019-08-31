import argparse
import os
import time
import uuid
from collections import deque
from typing import Optional
from TF_logger import Logger
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader
from dataset.AVA_video_v2 import AVA_video
from backbone.base import Base as BackboneBase
from config.train_config import TrainConfig as Config
from dataset.base import Base as DatasetBase
from extention.lr_scheduler import WarmUpMultiStepLR
from logger import Logger as Log
from model import Model
from roi.pooler_ import Pooler
def _train(backbone_name, path_to_checkpoints_dir, path_to_resuming_checkpoint):
    logger = Logger('./logs')
    dataset=AVA_video(Config.TRAIN_DATA)

    dataloader = DataLoader(dataset, batch_size=4,
                            num_workers=8, collate_fn=DatasetBase.padding_collate_fn,pin_memory=True,shuffle=True)

    Log.i('Found {:d} samples'.format(len(dataset)))

    backbone = BackboneBase.from_name(backbone_name)()
    model=Model(
            backbone, dataset.num_classes(), pooler_mode=Config.POOLER_MODE,
            anchor_ratios=Config.ANCHOR_RATIOS, anchor_sizes=Config.ANCHOR_SIZES,
            rpn_pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=Config.RPN_POST_NMS_TOP_N,
            anchor_smooth_l1_loss_beta=Config.ANCHOR_SMOOTH_L1_LOSS_BETA, proposal_smooth_l1_loss_beta=Config.PROPOSAL_SMOOTH_L1_LOSS_BETA
        ).cuda()


    optimizer = optim.SGD(model.parameters(), lr=Config.LEARNING_RATE,
                          momentum=Config.MOMENTUM, weight_decay=Config.WEIGHT_DECAY)
    scheduler = WarmUpMultiStepLR(optimizer, milestones=Config.STEP_LR_SIZES, gamma=Config.STEP_LR_GAMMA,
                                  factor=Config.WARM_UP_FACTOR, num_iters=Config.WARM_UP_NUM_ITERS)

    step = 0
    time_checkpoint = time.time()
    losses = deque(maxlen=100)
    mean_losses = deque(maxlen=100)

    summary_writer = SummaryWriter(os.path.join(path_to_checkpoints_dir, 'summaries'))
    should_stop = False

    num_steps_to_display = Config.NUM_STEPS_TO_DISPLAY
    num_steps_to_snapshot = Config.NUM_STEPS_TO_SNAPSHOT
    num_steps_to_finish = Config.NUM_STEPS_TO_FINISH

    if path_to_resuming_checkpoint is not None:
        step = model.load(path_to_resuming_checkpoint, optimizer, scheduler)
        print("load from:",path_to_resuming_checkpoint)

    device_count = torch.cuda.device_count()
    assert Config.BATCH_SIZE % device_count == 0, 'The batch size is not divisible by the device count'
    Log.i('Start training with {:d} GPUs ({:d} batches per GPU)'.format(torch.cuda.device_count(),
                                                                        Config.BATCH_SIZE // torch.cuda.device_count()))

    while not should_stop:
        for n_iter, (_, image_batch, _, bboxes_batch, labels_batch,detector_bboxes_batch) in enumerate(dataloader):
            batch_size = image_batch.shape[0]
            image_batch = image_batch.cuda()
            bboxes_batch = bboxes_batch.cuda()
            labels_batch = labels_batch.cuda()
            detector_bboxes_batch=detector_bboxes_batch.cuda()
            #sure 1
            # print("bboxes_batch:",bboxes_batch)
            # print("detector_bboxes_batch:",detector_bboxes_batch)
            # print("labels_batch:",labels_batch)
            proposal_class_losses = \
                model.eval().forward(image_batch, bboxes_batch, labels_batch,detector_bboxes_batch)
            proposal_class_loss = proposal_class_losses.mean()

            loss = proposal_class_loss
            mean_loss=proposal_class_losses.mean()
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            scheduler.step()

            losses.append(loss.item())
            mean_losses.append(mean_loss.item())
            summary_writer.add_scalar('train/proposal_class_loss', proposal_class_loss.item(), step)
            summary_writer.add_scalar('train/loss', loss.item(), step)
            if n_iter % 10000 == 0:
                for name, param in model.named_parameters():
                    name = name.replace('.', '/')
                    if name.find("conv") >= 0:
                        summary_writer.add_histogram(name, param.data.cpu().numpy(), global_step=n_iter)
                        summary_writer.add_histogram(name + 'grad', param.grad.data.cpu().numpy(),
                                                     global_step=n_iter)
            #summary_writer.add_graph(model, (image_batch))
            step += 1

            if step == num_steps_to_finish:
                should_stop = True

            if step % num_steps_to_display == 0:
                elapsed_time = time.time() - time_checkpoint
                time_checkpoint = time.time()
                steps_per_sec = num_steps_to_display / elapsed_time
                samples_per_sec = batch_size * steps_per_sec
                eta = (num_steps_to_finish - step) / steps_per_sec / 3600
                avg_loss = sum(losses) / len(losses)
                avg_mean_loss=sum(mean_losses) / len(mean_losses)
                lr = scheduler.get_lr()[0]
                #Log.i('[Step {step}] Avg. Loss = {avg_loss:.6f}, Learning Rate = {lr:%.8f} ({samples_per_sec:.2f} samples/sec; ETA {eta:.1f} hrs)')
                print_string='[Step {0}] Avg. Loss = {avg_loss:.6f}, Learning Rate = {lr:.8f} ({samples_per_sec:.2f} samples/sec; ETA {eta:.1f} hrs)'\
                .format(step,avg_loss=avg_loss,lr=lr,samples_per_sec=samples_per_sec,eta=eta)
                print(print_string)
            if step % num_steps_to_snapshot == 0 or should_stop:
                path_to_checkpoint = model.save(path_to_checkpoints_dir, step, optimizer, scheduler)
                Log.i('Model has been saved to {}'.format(path_to_checkpoint))
            if should_stop:
                break
    Log.i('Done')


if __name__ == '__main__':
    def main():
        backbone_name = Config.BACKBONE_NAME
        path_to_outputs_dir = Config.PATH_TO_OUTPUTS_DIR
        path_to_resuming_checkpoint =Config.PATH_TO_RESUMEING_CHECKPOINT
        path_to_checkpoints_dir = os.path.join(path_to_outputs_dir, 'checkpoints-{:s}-{:s}'.format(
            backbone_name,time.strftime('%d%H%M')))
        path_to_checkpoints_dir="temp_4"
        os.makedirs(path_to_checkpoints_dir)
        Log.initialize(os.path.join(path_to_checkpoints_dir, 'train.log'))
        Log.i('Arguments:')
        # for k, v in vars(args).items():
        #     Log.i('\t{k} = {v}')
        # Log.i(Config.describe())
        _train(backbone_name, path_to_checkpoints_dir, path_to_resuming_checkpoint)

    main()
