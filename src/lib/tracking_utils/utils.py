import glob
import os
import os.path as osp
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import nms

import motmetrics as mm

# import maskrcnn_benchmark.layers.nms as nms
# Set printoptions
torch.set_printoptions(linewidth=1320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5

def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)


def float3(x):  # format floats to 3 decimals
    return float(format(x, '.3f'))


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, 'r')
    names = fp.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))



def plot_one_box(x, img, color=None, label=None, line_thickness=None):  # Plots one bounding box on image img
    tl = line_thickness or round(0.0004 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.03)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.03)
        torch.nn.init.constant_(m.bias.data, 0.0)


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)
    return y


def scale_coords(img_size, coords, img0_shape):
    # Rescale x1, y1, x2, y2 from 416 to image size
    gain_w = float(img_size[0]) / img0_shape[1]  # gain  = old / new
    gain_h = float(img_size[1]) / img0_shape[0]
    gain = min(gain_w, gain_h)
    pad_x = (img_size[0] - img0_shape[1] * gain) / 2  # width padding
    pad_y = (img_size[1] - img0_shape[0] * gain) / 2  # height padding
    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, 0:4] /= gain
    coords[:, :4] = torch.clamp(coords[:, :4], min=0)
    return coords


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Method originally from https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # lists/pytorch to numpy
    tp, conf, pred_cls, target_cls = np.array(tp), np.array(conf), np.array(pred_cls), np.array(target_cls)

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(np.concatenate((pred_cls, target_cls), 0))

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = sum(target_cls == c)  # Number of ground truth objects
        n_p = sum(i)  # Number of predicted objects

        if (n_p == 0) and (n_gt == 0):
            continue
        elif (n_p == 0) or (n_gt == 0):
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = np.cumsum(1 - tp[i])
            tpc = np.cumsum(tp[i])

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(tpc[-1] / (n_gt + 1e-16))

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(tpc[-1] / (tpc[-1] + fpc[-1]))

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    return np.array(ap), unique_classes.astype('int32'), np.array(r), np.array(p)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def bbox_iou(box1, box2, x1y1x2y2=False):
    """
    Returns the IoU of two bounding boxes
    """
    N, M = len(box1), len(box2)
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
    inter_rect_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
    inter_rect_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
    inter_rect_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
    b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1)).view(-1,1).expand(N,M)
    b2_area = ((b2_x2 - b2_x1) * (b2_y2 - b2_y1)).view(1,-1).expand(N,M)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)


def build_targets_max(target, anchor_wh, nA, nC, nGh, nGw):
    """
    returns nT, nCorrect, tx, ty, tw, th, tconf, tcls
    """
    nB = len(target)  # number of images in batch

    txy = torch.zeros(nB, nA, nGh, nGw, 2).cuda()  # batch size, anchors, grid size
    twh = torch.zeros(nB, nA, nGh, nGw, 2).cuda()
    tconf = torch.LongTensor(nB, nA, nGh, nGw).fill_(0).cuda()
    tcls = torch.ByteTensor(nB, nA, nGh, nGw, nC).fill_(0).cuda()  # nC = number of classes
    tid = torch.LongTensor(nB, nA, nGh, nGw, 1).fill_(-1).cuda() 
    for b in range(nB):
        t = target[b]
        t_id = t[:, 1].clone().long().cuda()
        t = t[:,[0,2,3,4,5]]
        nTb = len(t)  # number of targets
        if nTb == 0:
            continue

        #gxy, gwh = t[:, 1:3] * nG, t[:, 3:5] * nG
        gxy, gwh = t[: , 1:3].clone() , t[:, 3:5].clone()
        gxy[:, 0] = gxy[:, 0] * nGw
        gxy[:, 1] = gxy[:, 1] * nGh
        gwh[:, 0] = gwh[:, 0] * nGw
        gwh[:, 1] = gwh[:, 1] * nGh
        gi = torch.clamp(gxy[:, 0], min=0, max=nGw -1).long()
        gj = torch.clamp(gxy[:, 1], min=0, max=nGh -1).long()

        # Get grid box indices and prevent overflows (i.e. 13.01 on 13 anchors)
        #gi, gj = torch.clamp(gxy.long(), min=0, max=nG - 1).t()
        #gi, gj = gxy.long().t()

        # iou of targets-anchors (using wh only)
        box1 = gwh
        box2 = anchor_wh.unsqueeze(1)
        inter_area = torch.min(box1, box2).prod(2)
        iou = inter_area / (box1.prod(1) + box2.prod(2) - inter_area + 1e-16)

        # Select best iou_pred and anchor
        iou_best, a = iou.max(0)  # best anchor [0-2] for each target

        # Select best unique target-anchor combinations
        if nTb > 1:
            _, iou_order = torch.sort(-iou_best)  # best to worst

            # Unique anchor selection
            u = torch.stack((gi, gj, a), 0)[:, iou_order]
            # _, first_unique = np.unique(u, axis=1, return_index=True)  # first unique indices
            first_unique = return_torch_unique_index(u, torch.unique(u, dim=1))  # torch alternative
            i = iou_order[first_unique]
            # best anchor must share significant commonality (iou) with target
            i = i[iou_best[i] > 0.60]  # TODO: examine arbitrary threshold
            if len(i) == 0:
                continue

            a, gj, gi, t = a[i], gj[i], gi[i], t[i]
            t_id = t_id[i]
            if len(t.shape) == 1:
                t = t.view(1, 5)
        else:
            if iou_best < 0.60:
                continue
        
        tc, gxy, gwh = t[:, 0].long(), t[:, 1:3].clone(), t[:, 3:5].clone()
        gxy[:, 0] = gxy[:, 0] * nGw
        gxy[:, 1] = gxy[:, 1] * nGh
        gwh[:, 0] = gwh[:, 0] * nGw
        gwh[:, 1] = gwh[:, 1] * nGh

        # XY coordinates
        txy[b, a, gj, gi] = gxy - gxy.floor()

        # Width and height
        twh[b, a, gj, gi] = torch.log(gwh / anchor_wh[a])  # yolo method
        # twh[b, a, gj, gi] = torch.sqrt(gwh / anchor_wh[a]) / 2 # power method

        # One-hot encoding of label
        tcls[b, a, gj, gi, tc] = 1
        tconf[b, a, gj, gi] = 1
        tid[b, a, gj, gi] = t_id.unsqueeze(1)
    tbox = torch.cat([txy, twh], -1)
    return tconf, tbox, tid




def generate_anchor(nGh, nGw, anchor_wh):
    nA = len(anchor_wh)
    yy, xx =torch.meshgrid(torch.arange(nGh), torch.arange(nGw))
    xx, yy = xx.cuda(), yy.cuda()

    mesh = torch.stack([xx, yy], dim=0)                                              # Shape 2, nGh, nGw
    mesh = mesh.unsqueeze(0).repeat(nA,1,1,1).float()                                # Shape nA x 2 x nGh x nGw
    anchor_offset_mesh = anchor_wh.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, nGh,nGw) # Shape nA x 2 x nGh x nGw
    anchor_mesh = torch.cat([mesh, anchor_offset_mesh], dim=1)                       # Shape nA x 4 x nGh x nGw
    return anchor_mesh

def encode_delta(gt_box_list, fg_anchor_list):
    px, py, pw, ph = fg_anchor_list[:, 0], fg_anchor_list[:,1], \
                     fg_anchor_list[:, 2], fg_anchor_list[:,3]
    gx, gy, gw, gh = gt_box_list[:, 0], gt_box_list[:, 1], \
                     gt_box_list[:, 2], gt_box_list[:, 3]
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw/pw)
    dh = torch.log(gh/ph)
    return torch.stack([dx, dy, dw, dh], dim=1)

def decode_delta(delta, fg_anchor_list):
    px, py, pw, ph = fg_anchor_list[:, 0], fg_anchor_list[:,1], \
                     fg_anchor_list[:, 2], fg_anchor_list[:,3]
    dx, dy, dw, dh = delta[:, 0], delta[:, 1], delta[:, 2], delta[:, 3]
    gx = pw * dx + px
    gy = ph * dy + py
    gw = pw * torch.exp(dw)
    gh = ph * torch.exp(dh)
    return torch.stack([gx, gy, gw, gh], dim=1)

def decode_delta_map(delta_map, anchors):
    '''
    :param: delta_map, shape (nB, nA, nGh, nGw, 4)
    :param: anchors, shape (nA,4)
    '''
    nB, nA, nGh, nGw, _ = delta_map.shape
    anchor_mesh = generate_anchor(nGh, nGw, anchors) 
    anchor_mesh = anchor_mesh.permute(0,2,3,1).contiguous()              # Shpae (nA x nGh x nGw) x 4
    anchor_mesh = anchor_mesh.unsqueeze(0).repeat(nB,1,1,1,1)
    pred_list = decode_delta(delta_map.view(-1,4), anchor_mesh.view(-1,4))
    pred_map = pred_list.view(nB, nA, nGh, nGw, 4)
    return pred_map


def pooling_nms(heatmap, kernel=1):
    pad = (kernel -1 ) // 2
    hmax = F.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return keep * heatmap


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.2):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    output = [None for _ in range(len(prediction))]
    for image_i, pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        # Get score and class with highest confidence

        v = pred[:, 4] > conf_thres
        v = v.nonzero().squeeze()
        if len(v.shape) == 0:
            v = v.unsqueeze(0)

        pred = pred[v]

        # If none are remaining => process next image
        nP = pred.shape[0]
        if not nP:
            continue
        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])
        nms_indices = nms(pred[:, :4], pred[:, 4], nms_thres)
        det_max = pred[nms_indices]        

        if len(det_max) > 0:
            # Add max detections to outputs
            output[image_i] = det_max if output[image_i] is None else torch.cat((output[image_i], det_max))

    return output


def return_torch_unique_index(u, uv):
    n = uv.shape[1]  # number of columns
    first_unique = torch.zeros(n, device=u.device).long()
    for j in range(n):
        first_unique[j] = (uv[:, j:j + 1] == u).all(0).nonzero()[0]

    return first_unique


def strip_optimizer_from_checkpoint(filename='weights/best.pt'):
    # Strip optimizer from *.pt files for lighter files (reduced by 2/3 size)

    a = torch.load(filename, map_location='cpu')
    a['optimizer'] = []
    torch.save(a, filename.replace('.pt', '_lite.pt'))


def plot_results():
    # Plot YOLO training results file 'results.txt'
    # import os; os.system('wget https://storage.googleapis.com/ultralytics/yolov3/results_v1.txt')

    plt.figure(figsize=(14, 7))
    s = ['X + Y', 'Width + Height', 'Confidence', 'Classification', 'Total Loss', 'mAP', 'Recall', 'Precision']
    files = sorted(glob.glob('results*.txt'))
    for f in files:
        results = np.loadtxt(f, usecols=[2, 3, 4, 5, 6, 9, 10, 11]).T  # column 11 is mAP
        x = range(1, results.shape[1])
        for i in range(8):
            plt.subplot(2, 4, i + 1)
            plt.plot(x, results[i, x], marker='.', label=f)
            plt.title(s[i])
            if i == 0:
                plt.legend()


def visualize_fp_fns_seq(result_dir, dataroot, seqs, return_fp=True, return_fn=True):
    from tracking_utils.evaluation import Evaluator
    accs = []
    fp_info = []
    tp_info = []
    for seq in seqs:
        img_dir = osp.join(dataroot, seq, 'img1')
        output_dir = osp.join(f"{result_dir.replace('results', 'outputs')}", 'analyze', seq)
        os.makedirs(output_dir, exist_ok=True)
        evaluator = Evaluator(data_root=dataroot, seq_name=seq, data_type='mot', return_fp=return_fp,
                              return_fn=return_fn)

        if return_fp and return_fn:
            acc, fns, fps, matched, fp_confs = evaluator.eval_file(osp.join(result_dir, f"{seq}.txt"))
            accs.append(acc)

            # draw fp and fn
            for frame_id in fps.keys():

                fp = fps[frame_id]
                fp_conf = fp_confs[frame_id]
                fp[:, 2:] += fp[:, :2]
                fn = fns[frame_id]
                fn[:, 2:] += fn[:, :2]
                match = matched[frame_id]
                match['gt'][:, 2:] += match['gt'][:, :2]
                match['trk'][:, 2:] += match['trk'][:, :2]

                im = cv2.imread(osp.join(img_dir, f"{frame_id:06d}.jpg"))
                for box, conf in zip(fp, fp_conf):
                    im = cv2.rectangle(im, tuple(box[0:2].astype(int)), tuple(box[2:4].astype(int)), (0, 0, 255), 2)
                    im = cv2.putText(im, f"c:{conf:.3f}", (int(box[0]), int(box[1] + 30)), cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 255, 255),
                                     thickness=1)
                    height = box[3] - box[1]
                    im = cv2.putText(im, f"h:{height:.3f}", (int(box[0]), int(box[1] + 60)), cv2.FONT_HERSHEY_PLAIN,
                                     0.75, (0, 255, 255),
                                     thickness=1)
                    fp_info.append([height, conf])

                for box in fn:
                    im = cv2.rectangle(im, tuple(box[0:2].astype(int)), tuple(box[2:4].astype(int)), (0, 0, 0), 2)

                for gt, trk, conf in zip(match['gt'], match['trk'], match['trk_confs']):
                    im = cv2.rectangle(im, tuple(gt[0:2].astype(int)), tuple(gt[2:4].astype(int)), (0, 255, 0), 2)
                    im = cv2.rectangle(im, tuple(trk[0:2].astype(int)), tuple(trk[2:4].astype(int)), (255, 255, 0), 2)
                    im = cv2.putText(im, f"c:{conf:.3f}", (int(trk[0]), int(trk[1] + 30)), cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 255, 255),
                                     thickness=1)
                    height = trk[3] - trk[1]
                    im = cv2.putText(im, f"h:{height:.3f}", (int(trk[0]), int(trk[1] + 60)), cv2.FONT_HERSHEY_PLAIN,
                                     0.75, (0, 255, 255),
                                     thickness=1)
                    tp_info.append([height, conf])
                cv2.imwrite(osp.join(output_dir, f"{frame_id:06d}.jpg"), im)
        else:
            accs.append(evaluator.eval_file(osp.join(result_dir, f"{seq}.txt")))
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    fp_info = np.array(fp_info)
    tp_info = np.array(tp_info)
    import ipdb
    ipdb.set_trace()


def visualize_comparison(dataroot, seq, save_dir, result_dir_1, result_dir_2, compile_images_only=False):
    import pandas as pd
    from cython_bbox import bbox_overlaps as bbox_ious
    from tracker import matching
    from tracking_utils import visualization as vis

    res_1 = pd.read_csv(osp.join(result_dir_1, f"{seq}.txt"), sep=',', names=['frame', 'id', 'left', 'top', 'width', 'height', 'conf', 'misc1', 'misc2', 'misc3'])
    res_2 = pd.read_csv(osp.join(result_dir_2, f"{seq}.txt"), sep=',', names=['frame', 'id', 'left', 'top', 'width', 'height', 'conf', 'misc1', 'misc2', 'misc3'])
    img_root = osp.join(dataroot, seq, 'img1')
    save_root_1 = osp.join(osp.abspath(save_dir), osp.basename(result_dir_1), seq)
    os.makedirs(save_root_1, exist_ok=True)
    save_root_2 = osp.join(osp.abspath(save_dir), osp.basename(result_dir_2), seq)
    os.makedirs(save_root_2, exist_ok=True)

    if not compile_images_only:
        # for frame in np.unique(res_1['frame']):
        for frame in range(np.min(res_1['frame']), np.max(res_1['frame'])):
            # if frame == 250:
            #     break
            res_1_frame = res_1[res_1['frame'] == frame]
            res_2_frame = res_2[res_2['frame'] == frame]

            res_1_boxes_np = np.array(res_1_frame[['left', 'top', 'width', 'height']])
            res_1_boxes_np[:, 2:] += res_1_boxes_np[:, :2]
            res_2_boxes_np = np.array(res_2_frame[['left', 'top', 'width', 'height']])
            res_2_boxes_np[:, 2:] += res_2_boxes_np[:, :2]

            img1 = cv2.imread(osp.join(img_root, f"{frame:06d}.jpg"))
            img2 = cv2.imread(osp.join(img_root, f"{frame:06d}.jpg"))

            if res_1_boxes_np.shape[0] > 0 and res_2_boxes_np.shape[0] > 0:
                iou = bbox_ious(res_1_boxes_np, res_2_boxes_np)
                matches, u_res_1, u_res_2 = matching.linear_assignment(-iou, thresh=-0.7)

                if len(matches) > 0:
                    # Plot the matchings in normal colors
                    matches_res_1 = res_1_frame.iloc[matches[:, 0]]
                    matches_res_2 = res_2_frame.iloc[matches[:, 1]]
                    # matches_res_2['id'].iloc[matches[:, 1]] = matches_res_1['id'].iloc[matches[:, 0]]
                    online_ids = matches_res_1['id'].to_list()
                    # online_ids = matches_res_2['id'].to_list()

                    matches_res_1_np = np.array(matches_res_1[['left', 'top', 'width', 'height']])
                    matches_res_2_np = np.array(matches_res_2[['left', 'top', 'width', 'height']])

                    online_im_1 = vis.plot_tracking(img1, matches_res_1_np, online_ids,
                                                    frame_id=frame, fps=-1, line_thickness=1)

                    online_im_2 = vis.plot_tracking(img2, matches_res_2_np, online_ids,
                                                    frame_id=frame, fps=-1, line_thickness=1)
                else:
                    online_im_1 = img1
                    online_im_2 = img2
                # ipdb.set_trace()

                # Plot the unmatched in thicker colors
                unmatches_res_1 = res_1_frame.iloc[u_res_1]
                unmatches_res_1_np = np.array(unmatches_res_1[['left', 'top', 'width', 'height']])
                if len(unmatches_res_1) > 0:
                    if len(unmatches_res_1.shape) == 1:
                        unmatched_res_1_ids = [unmatches_res_1['id']]
                        scores = [unmatches_res_1['conf']]
                        unmatches_res_1_np = unmatches_res_1_np.reshape(1, -1)
                    else:
                        unmatched_res_1_ids = unmatches_res_1['id'].to_list()
                        scores = unmatches_res_1['conf'].to_list()

                    online_im_1 = vis.plot_tracking(online_im_1, unmatches_res_1_np, unmatched_res_1_ids,
                                                    frame_id=frame, fps=-1,
                                                    line_thickness_unmatched=4)
                    # ipdb.set_trace()
                    print(f"{len(unmatches_res_1)} unmatches found in frame {frame}")

                # Plot the unmatched in thicker colors
                unmatches_res_2 = res_2_frame.iloc[u_res_2]
                unmatches_res_2_np = np.array(unmatches_res_2[['left', 'top', 'width', 'height']])
                if len(unmatches_res_2) > 0:
                    if len(unmatches_res_2.shape) == 1:
                        unmatched_res_2_ids = [unmatches_res_2['id']]
                        scores = [unmatches_res_2['conf']]
                        unmatches_res_2_np = unmatches_res_2_np.reshape(1, -1)
                    else:
                        unmatched_res_2_ids = unmatches_res_2['id'].to_list()
                        scores = unmatches_res_2['conf'].to_list()

                    online_im_2 = vis.plot_tracking(online_im_2, unmatches_res_2_np, unmatched_res_2_ids,
                                                    frame_id=frame, fps=-1,
                                                    line_thickness_unmatched=4)

                cv2.imwrite(osp.join(save_root_1, f"{frame:06d}.jpg"), online_im_1)
                cv2.imwrite(osp.join(save_root_2, f"{frame:06d}.jpg"), online_im_2)
            else:
                if len(res_1_boxes_np) > 0 and len(res_2_boxes_np) == 0:
                    matches_res_1 = res_1_frame
                    matches_res_1_np = np.array(matches_res_1[['left', 'top', 'width', 'height']])
                    if len(matches_res_1.shape)  == 1:
                        online_ids = [matches_res_1['id']]
                        scores = [matches_res_1['conf']]
                    else:
                        online_ids = matches_res_1['id'].to_list()
                        scores = matches_res_1['conf'].to_list()
                    online_im_1 = vis.plot_tracking(img1, matches_res_1_np, online_ids,
                                                    frame_id=frame, fps=-1, line_thickness_unmatched=4)
                    cv2.imwrite(osp.join(save_root_1, f"{frame:06d}.jpg"), online_im_1)
                    cv2.imwrite(osp.join(save_root_2, f"{frame:06d}.jpg"), img2)
                if len(res_1_boxes_np) == 0 and len(res_2_boxes_np) > 0:
                    matches_res_2 = res_2_frame
                    matches_res_2_np = np.array(matches_res_2[['left', 'top', 'width', 'height']])
                    if len(matches_res_2.shape) == 1:
                        online_ids = [matches_res_2['id']]
                        scores = [matches_res_2['conf']]
                    else:
                        online_ids = matches_res_2['id'].to_list()
                        scores = matches_res_2['conf'].to_list()

                    online_im_2 = vis.plot_tracking(img2, matches_res_2_np, online_ids,
                                                    frame_id=frame, fps=-1, line_thickness_unmatched=4)

                    cv2.imwrite(osp.join(save_root_1, f"{frame:06d}.jpg"), img1)
                    cv2.imwrite(osp.join(save_root_2, f"{frame:06d}.jpg"), online_im_2)
                if len(res_1_boxes_np) == 0 and len(res_1_boxes_np) == 0:
                    cv2.imwrite(osp.join(save_root_1, f"{frame:06d}.jpg"), img1)
                    cv2.imwrite(osp.join(save_root_2, f"{frame:06d}.jpg"), img2)
    cmd_str = 'ffmpeg -framerate 8 -y -f image2 -i {}/%06d.jpg -vcodec libx264 -c:v copy {}'.format(save_root_1, osp.join(save_root_1, f'GNN_{seq}.avi'))
    os.system(cmd_str)
    cmd_str = 'ffmpeg -framerate 8 -y -f image2 -i {}/%06d.jpg -vcodec libx264 -c:v copy {}'.format(save_root_2, osp.join(save_root_2, f'noGNN_{seq}.avi'))
    os.system(cmd_str)
