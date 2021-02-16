from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from .dataset.gsdt_dataset import JointDatasetGNN


def get_dataset(dataset, task):
  assert task == 'gnn_mot', f'Expected task to be gnn_mot, but got {task}'
  return JointDatasetGNN
