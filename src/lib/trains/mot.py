from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.decode import mot_decode
from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.utils import _sigmoid, _tranpose_and_gather_feat
from utils.post_process import ctdet_post_process

from .base_gnn_trainer_distributed import BaseGNNTrainerDistributed

class MotLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MotLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt
        self.emb_dim = opt.reid_dim
        self.nID = opt.nID
        self.classifier = nn.Linear(self.emb_dim, self.nID)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))

    def forward(self, outputs, batch, dataparallel=False):
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss = 0, 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                if opt.dense_wh:
                    mask_weight = batch['dense_wh_mask'].sum() + 1e-4
                    wh_loss += (
                                   self.crit_wh(output['wh'] * batch['dense_wh_mask'],
                                                batch['dense_wh'] * batch['dense_wh_mask']) /
                                   mask_weight) / opt.num_stacks
                else:
                    wh_loss += self.crit_reg(
                        output['wh'], batch['reg_mask'],
                        batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            if opt.id_weight > 0:
                id_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
                id_head = id_head[batch['reg_mask'] > 0].contiguous()
                id_head = self.emb_scale * F.normalize(id_head)
                id_target = batch['ids'][batch['reg_mask'] > 0]
                id_output = self.classifier(id_head).contiguous()
                id_loss += self.IDLoss(id_output, id_target)


        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss

        loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss + (self.s_det + self.s_id)
        loss *= 0.5

        if not dataparallel:
            loss_stats = {'loss': loss.mean().item(), 'hm_loss': hm_loss.mean().item(),
                          'wh_loss': wh_loss.mean().item(), 'off_loss': off_loss.mean().item(), 'id_loss': id_loss.mean().item()}
        else:
            loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'wh_loss': wh_loss, 'off_loss': off_loss,
                          'id_loss': id_loss}
        return loss, loss_stats


class MotLossWithEdgeRegression(torch.nn.Module):
    def __init__(self, opt):
        super(MotLossWithEdgeRegression, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.crit_edge = torch.nn.BCEWithLogitsLoss()
        self.opt = opt
        self.emb_dim = opt.reid_dim
        self.nID = opt.nID
        self.classifier = nn.Linear(self.emb_dim, self.nID)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        #self.TriLoss = TripletLoss()
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))

    def forward(self, outputs, batch, dataparallel=False):
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss, edge_loss = 0, 0, 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                if opt.dense_wh:
                    mask_weight = batch['dense_wh_mask'].sum() + 1e-4
                    wh_loss += (
                                   self.crit_wh(output['wh'] * batch['dense_wh_mask'],
                                                batch['dense_wh'] * batch['dense_wh_mask']) /
                                   mask_weight) / opt.num_stacks
                else:
                    wh_loss += self.crit_reg(
                        output['wh'], batch['reg_mask'],
                        batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            if opt.id_weight > 0:
                id_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
                id_head = id_head[batch['reg_mask'] > 0].contiguous()
                id_head = self.emb_scale * F.normalize(id_head)
                id_target = batch['ids'][batch['reg_mask'] > 0]
                id_output = self.classifier(id_head).contiguous()
                id_loss += self.IDLoss(id_output, id_target)
                #id_loss += self.IDLoss(id_output, id_target) + self.TriLoss(id_head, id_target)

            if opt.edge_reg_weight > 0:
                # TODO: compute the edge regression with BCELoss here
                if output['edge_preds'] is None or output['edge_labels'] is None:
                    edge_loss += torch.tensor(0.).to(hm_loss.device)
                elif len(output['edge_preds']) == 0 or len(output['edge_labels']) == 0:
                    edge_loss += torch.tensor(0.).to(hm_loss.device)
                else:
                    edge_loss += self.crit_edge(output['edge_preds'], output['edge_labels']) * opt.edge_reg_weight
                if torch.isnan(edge_loss):
                    print(output['edge_preds'])
                    print(output['edge_labels'])
                    import ipdb
                    ipdb.set_trace()
        #loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss + opt.id_weight * id_loss

        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss

        # loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss + (self.s_det + self.s_id)
        loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * (id_loss + edge_loss) + (
                    self.s_det + self.s_id)
        loss *= 0.5

        #print(loss, hm_loss, wh_loss, off_loss, id_loss)
        if not dataparallel:
            loss_stats = {'loss': loss.mean().item(), 'hm_loss': hm_loss.mean().item(),
                          'wh_loss': wh_loss.mean().item(), 'off_loss': off_loss.mean().item(),
                          'id_loss': id_loss.mean().item(), 'edge_loss': edge_loss.mean().item()}
        else:
            loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'wh_loss': wh_loss, 'off_loss': off_loss,
                          'id_loss': id_loss, 'edge_loss': edge_loss}
        return loss, loss_stats


class GNNMotTrainerDistributed(BaseGNNTrainerDistributed):
    def __init__(self, opt, model_with_loss, optimizer=None):
        super(GNNMotTrainerDistributed, self).__init__(opt, model_with_loss, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss', 'edge_loss']
        return loss_states

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = mot_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]