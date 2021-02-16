from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar

from utils.utils import AverageMeter
import traceback


class GNNModelWithLossDistributed(torch.nn.Module):
    def __init__(self, model, loss, edge_regression=False, motion_model=False):
        super(GNNModelWithLossDistributed, self).__init__()
        self.model = model
        self.loss = loss
        self.edge_regression = edge_regression
        self.motion_model = motion_model

    def forward(self, batch):
        if self.edge_regression:
            if not self.motion_model:
                outputs = self.model(batch['input'], batch['p_crops'], batch['p_crops_lengths'], batch['edge_index'],
                                     batch['p_imgs'], batch['positive_edge_index'])
            else:
                outputs = self.model(batch['input'], batch['p_crops'], batch['p_crops_lengths'], batch['edge_index'],
                                     batch['p_imgs'], batch['positive_edge_index'], batch['p_motion'])
        else:
            outputs = self.model(batch['input'], batch['p_crops'], batch['p_crops_lengths'], batch['edge_index'],
                                 batch['p_imgs'])
        loss, loss_stats = self.loss(outputs, batch)
        # return outputs[-1], loss, loss_stats
        return loss, loss_stats


class GNNModelWithLossDistributedReturnGNNOutputs(torch.nn.Module):
    def __init__(self, model, loss, edge_regression=False, motion_model=False):
        super(GNNModelWithLossDistributedReturnGNNOutputs, self).__init__()
        self.model = model
        self.loss = loss
        self.edge_regression = edge_regression
        self.motion_model = motion_model

    def forward(self, batch):
        if self.edge_regression:
            if not self.motion_model:
                outputs = self.model(batch['input'], batch['p_crops'], batch['p_crops_lengths'], batch['edge_index'],
                                     batch['p_imgs'], batch['positive_edge_index'])
            else:
                outputs = self.model(batch['input'], batch['p_crops'], batch['p_crops_lengths'], batch['edge_index'],
                                     batch['p_imgs'], batch['positive_edge_index'], batch['p_motion'])
        else:
            outputs = self.model(batch['input'], batch['p_crops'], batch['p_crops_lengths'], batch['edge_index'],
                                 batch['p_imgs'])
        loss, loss_stats = self.loss[0](outputs[0], batch)
        for i in range(1, len(outputs)):
            if len(self.loss) == 1:
                loss_i, loss_stats_i = self.loss[0](outputs[i], batch)
            else:
                loss_i, loss_stats_i = self.loss[i](outputs[i], batch)
            # aggregate loss and loss0
            loss += loss_i
            for k, v in loss_stats_i.items():
                loss_stats[k] += v

        return loss, loss_stats


class BaseGNNTrainerDistributed(object):
    def __init__(
            self, opt, model_with_loss, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats = self._get_losses(opt)
        self.model_with_loss = model_with_loss


    def run_epoch(self, phase, epoch, data_loader, rank):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        if rank == 0:
            bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            try:
                if iter_id >= num_iters:
                    break
                data_time.update(time.time() - end)

                for k in batch:
                    if k != 'meta' and k != 'edge_index' and k != 'positive_edge_index' and k != 'img_path':
                        batch[k] = batch[k].to(device=opt.device, non_blocking=True)

                    if k == 'edge_index' or k == 'positive_edge_index':
                        for i in range(len(batch[k])):
                            batch[k][i] = batch[k][i].to(device=opt.device, non_blocking=True)
                loss, loss_stats = model_with_loss(batch)
                loss = loss.mean()
                if phase == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                batch_time.update(time.time() - end)
                end = time.time()
                if rank == 0:
                    Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                        epoch, iter_id, num_iters, phase=phase,
                        total=bar.elapsed_td, eta=bar.eta_td)
                    for l in avg_loss_stats:
                        if l in loss_stats:
                            avg_loss_stats[l].update(
                                loss_stats[l], batch['input'].size(0)
                            )
                            Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
                    if not opt.hide_data_time:
                        Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                                  '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
                    if opt.print_iter > 0:
                        if iter_id % opt.print_iter == 0:
                            print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
                    else:
                        bar.next()

                del loss, loss_stats, batch
            except Exception as e:
                print(traceback.print_exc())
                print(e)

                import ipdb
                ipdb.set_trace()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        if rank == 0:
            bar.finish()
            ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, opt):
        raise NotImplementedError

    def val(self, epoch, data_loader, rank):
        return self.run_epoch('val', epoch, data_loader, rank)

    def train(self, epoch, data_loader, rank):
        return self.run_epoch('train', epoch, data_loader, rank)