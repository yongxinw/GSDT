from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import re
import copy

from .networks.pose_dla_dcn import get_pose_net_with_gnn as get_dla_dcn_gnn

_model_factory = {
    'dlagnn': get_dla_dcn_gnn
}


def create_model(arch, heads, head_conv, **kwargs):
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    get_model = _model_factory[arch]
    model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv, **kwargs)
    return model


def load_model(model, model_path, optimizer=None, resume=False, lr=None, lr_step=None,
               distributed=False, copy_head_weights=True, load_modules=None, rank=0):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    if rank == 0:
        print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}
    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            if distributed:
                name = k.replace("module.", "").replace("model.", "")
            else:
                name = k[7:]
        else:
            name = k
        module = name.split(".")[0]
        if load_modules is not None:
            # if load_modules specified, only load params from this list
            if module in load_modules:
                state_dict[name] = state_dict_[k]
        else:
            # otherwise, load every parameter possible
            state_dict[name] = state_dict_[k]
    model_state_dict = model.state_dict()
    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you have correctly specified --arch xxx ' + \
          'or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                if rank == 0:
                    print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}. {}'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            if rank == 0:
                print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            if rank == 0:
                print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    if copy_head_weights:
        # Copy current heads parameters to heads at earlier gnn layers
        heads_state_dict = {}
        for k, v in model_state_dict.items():
            if k.startswith('hm_') or k.startswith('wh_') or k.startswith('reg_') or k.startswith('id_'):
                state_dict_look_up_key = re.sub(r'_\d+', '', k)
                heads_state_dict[k] = copy.deepcopy(model_state_dict[state_dict_look_up_key].clone())
                print(f"Copied weights from {state_dict_look_up_key} to {k}. "
                      f"You can ignore the related warning messages previously.")
        model.load_state_dict(heads_state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            if rank == 0:
                print('Resumed optimizer with start lr', start_lr)
        else:
            if rank == 0:
                print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)
