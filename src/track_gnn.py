from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import torchvision

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch
import torchvision.transforms.functional as TF

from tracker.multitracker import GNNTracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.gsdt_dataset as datasets
from datasets.dataset.gsdt_dataset import JointDatasetGNN, letterbox

from tracking_utils.utils import mkdir_if_missing
from opts import opts


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},{conf},-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, track_confs in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id, track_conf in zip(tlwhs, track_ids, track_confs):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h,
                                          conf=track_conf)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30, conf_thres=0.3):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = GNNTracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    for i, (path, img, img0, p_img_path, p_img) in enumerate(dataloader):
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        if i == 0:
            p_boxes, init_img_path, p_img = dataloader.initialize(use_letter_box=opt.use_letter_box)
        else:
            p_boxes, p_img = prepare_prev_img(dataloader, online_targets, opt, p_img)

        if opt.use_roi_align:
            p_crops = p_boxes.clone()
            _, h, w = p_img.shape
            p_crops = p_crops.cuda()
            p_crops_lengths = [len(p_crops)]
            edge_index = create_inference_time_graph(opt, p_boxes, p_crops, p_img)

            # convert boxes from xyxy to normalized according to p_img dimensions
            p_crops[:, 0] = p_crops[:, 0] / w
            p_crops[:, 1] = p_crops[:, 1] / h
            p_crops[:, 2] = p_crops[:, 2] / w
            p_crops[:, 3] = p_crops[:, 3] / h
            online_targets = tracker.update(blob, img0, p_crops, p_crops_lengths, edge_index,
                                            gnn_output_layer=opt.inference_gnn_output_layer,
                                            p_imgs=p_img.unsqueeze(0).cuda(), conf_thres=conf_thres)
        else:
            p_crops = torchvision.ops.roi_align(
                input=p_img.unsqueeze(0),
                boxes=[p_boxes],
                output_size=opt.crop_size
            )
            p_crops = p_crops.cuda()
            p_crops_lengths = [len(p_crops)]

            edge_index = create_inference_time_graph(opt, p_boxes, p_crops, p_img)

            online_targets = tracker.update(blob, img0, p_crops, p_crops_lengths, edge_index,
                                            gnn_output_layer=opt.inference_gnn_output_layer, p_imgs=None, conf_thres=conf_thres)
        online_tlwhs = []
        online_ids = []
        online_confs = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            t_conf = t.score
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_confs.append(t_conf)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids, online_confs))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, scores=online_confs, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def create_inference_time_graph(opt, p_boxes, p_crops, p_img):
    if opt.graph_type == 'global':
        edge_index = [
            JointDatasetGNN.build_edge_index_full(
                len(p_crops),
                opt.default_backbone_feature_resolution[0] * opt.default_backbone_feature_resolution[1]
            ).cuda()
        ]
    elif opt.graph_type == 'local':
        p_boxes_centerwh = p_boxes.clone()
        _, h, w = p_img.shape
        # convert boxes from xyxy to cwh, normalized according to p_img dimensions
        p_boxes_centerwh[:, 0] = (p_boxes[:, 0] + p_boxes[:, 2]) / 2 / w
        p_boxes_centerwh[:, 1] = (p_boxes[:, 1] + p_boxes[:, 3]) / 2 / h
        p_boxes_centerwh[:, 2] = (p_boxes[:, 2] - p_boxes[:, 0]) / w
        p_boxes_centerwh[:, 3] = (p_boxes[:, 3] - p_boxes[:, 1]) / h
        dummy_frame_and_id = torch.zeros((p_boxes_centerwh.shape[0], 2))
        p_boxes_centerwh = torch.cat((dummy_frame_and_id, p_boxes_centerwh), dim=1)
        edge_index = [
            JointDatasetGNN.build_edge_index_local(
                p_labels=p_boxes_centerwh.numpy(),
                box_length=15,
                fm_height=opt.default_backbone_feature_resolution[0],
                fm_width=opt.default_backbone_feature_resolution[1]
            ).cuda()
        ]
    else:
        raise NotImplementedError
    return edge_index


def prepare_prev_img(dataloader, online_targets, opt, p_img):
    if len(online_targets) == 0:
        # if no detections exist
        p_boxes, init_img_path, p_img = dataloader.initialize(use_letter_box=opt.use_letter_box)
    else:
        # otherwise, read boxes from online_targets
        p_boxes0 = np.stack([t.tlwh for t in online_targets]).astype(np.float32).copy()
        if p_boxes0.shape[0] > opt.p_K:
            p_boxes0 = p_boxes0[np.random.choice(p_boxes0.shape[0], opt.p_K, replace=False)]
        # tlwh to xyxy format
        p_boxes0[:, 2] = p_boxes0[:, 2] + p_boxes0[:, 0]
        p_boxes0[:, 3] = p_boxes0[:, 3] + p_boxes0[:, 1]

        if not opt.use_letter_box:
            p_img = np.ascontiguousarray(p_img[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)
            p_img /= 255.0
            p_boxes = p_boxes0.copy()

            p_img = torch.from_numpy(p_img)
            p_boxes = torch.from_numpy(p_boxes)
        else:
            p_img, ratio, padw, padh = letterbox(p_img, height=dataloader.height, width=dataloader.width)

            # convert to the letter box coordinates
            p_boxes = p_boxes0.copy()
            p_boxes[:, 0] = ratio * p_boxes0[:, 0] + padw
            p_boxes[:, 1] = ratio * p_boxes0[:, 1] + padh
            p_boxes[:, 2] = ratio * p_boxes0[:, 2] + padw
            p_boxes[:, 3] = ratio * p_boxes0[:, 3] + padh

            p_img = TF.to_tensor(p_img)
            p_boxes = torch.from_numpy(p_boxes)
    return p_boxes, p_img


def main(opt, seq2conf, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        dataloader = datasets.LoadImagesGNN(osp.join(data_root, seq, 'img1'),
                                            osp.join(det_root, seq, 'det', 'det.txt'),
                                            opt.img_size,
                                            max_p_object=opt.p_K)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        conf = seq2conf[seq]
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate, conf_thres=conf)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -y -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    opt = opts().init()

    if not opt.val_mot16:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ETH-Bahnhof
                      ETH-Sunnyday
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte'''

        config_path = './lib/cfg/mot15.json'
        data_config = json.load(open(config_path))
        seq2conf = data_config['val_confs']
        data_root = f"{data_config['root']}/MOT15/images/train"
        det_root = f"{data_config['root']}/MOT15/images/train"
    else:
        seqs_str = '''MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13'''
        data_root = os.path.join(opt.data_dir, 'MOT16/train')
        config_path = './lib/cfg/mot16.json'
        data_config = json.load(open(config_path))
        seq2conf = data_config['val_confs']
    if opt.test_mot16:
        seqs_str = '''MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12
                      MOT16-14'''
        data_root = os.path.join(opt.data_dir, 'MOT16/test')
        config_path = './lib/cfg/mot16.json'
        data_config = json.load(open(config_path))
        seq2conf = data_config['test_confs']
    if opt.test_mot15:
        seqs_str = '''ADL-Rundle-1
                      ADL-Rundle-3
                      AVG-TownCentre
                      ETH-Crossing
                      ETH-Jelmoli
                      ETH-Linthescher
                      KITTI-16
                      KITTI-19
                      PETS09-S2L2
                      TUD-Crossing
                      Venice-1'''
        config_path = './lib/cfg/mot15.json'
        data_config = json.load(open(config_path))
        seq2conf = data_config['test_confs']
        data_root = f"{data_config['root']}/MOT15/images/test"
        det_root = f"{data_config['root']}/MOT15/images/test"
    if opt.test_mot17:
        seqs_str = '''MOT17-01-SDP
                      MOT17-03-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-14-SDP'''
        # data_root = os.path.join(opt.data_dir, 'MOT17/images/test')
        config_path = './lib/cfg/mot17.json'
        data_config = json.load(open(config_path))
        seq2conf = data_config['test_confs']
        data_root = f"{data_config['root']}/MOT17/images/test"
        det_root = f"{data_config['root']}/MOT17/images/test"
    if opt.val_mot17:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''

        config_path = './lib/cfg/mot17.json'
        data_config = json.load(open(config_path))
        seq2conf = data_config['val_confs']
        data_root = f"{data_config['root']}/MOT17/images/train"
        det_root = f"{data_config['root']}/MOT17/images/train"
    if opt.val_mot15:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ETH-Bahnhof
                      ETH-Sunnyday
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte'''
        config_path = './lib/cfg/mot15.json'
        data_config = json.load(open(config_path))
        seq2conf = data_config['val_confs']
        data_root = f"{data_config['root']}/MOT15/images/train"
        det_root = f"{data_config['root']}/MOT15/images/train"
    if opt.val_mot20:
        seqs_str = '''MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05
                      '''
        config_path = './lib/cfg/mot20.json'
        data_config = json.load(open(config_path))
        seq2conf = data_config['val_confs']
        data_root = f"{data_config['root']}/MOT20/images/train"
        det_root = f"{data_config['root']}/MOT20/images/train"
    if opt.test_mot20:
        seqs_str = '''MOT20-04
                      MOT20-06
                      MOT20-07
                      MOT20-08
                      '''
        # data_root = os.path.join(opt.data_dir, 'MOT20/images/test')
        config_path = './lib/cfg/mot20.json'
        data_config = json.load(open(config_path))
        seq2conf = data_config['test_confs']
        data_root = f"{data_config['root']}/MOT20/images/test"
        det_root = f"{data_config['root']}/MOT20/images/test"
    seqs = [seq.strip() for seq in seqs_str.split()]


    if not opt.eval_from_file_only:
        main(opt, seq2conf,
             data_root=data_root,
             det_root=det_root,
             seqs=seqs,
             exp_name=opt.exp_name,
             show_image=False,
             save_images=opt.save_images,
             save_videos=opt.save_videos)
    else:
        if opt.visualize_gt:
            from tracking_utils.utils import visualize_fp_fns_seq
            visualize_fp_fns_seq(opt.eval_result_dir, data_root, seqs, return_fn=False, return_fp=False)

        if opt.visualize_compare:
            from tracking_utils.utils import visualize_comparison
            visualize_comparison(dataroot=data_root,
                                 seq=opt.compare_seq,
                                 result_dir_1=opt.result_dir_1,
                                 result_dir_2=opt.result_dir_2,
                                 save_dir=os.path.join(data_root, '..', 'outputs', opt.exp_name),
                                 compile_images_only=opt.compile_images_only)
