# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from common import *

feature_dim = 2869

model = dict(
    type='GFBModelGCN1',
    num_classes=num_classes,
    gfb_loss_weight=gfb_loss_weight,
    pre_trans=dict(
        type="mlp1D",
        num_layers=1,
        in_channels=feature_dim,
        h_channels=feature_dim,
        out_channels=feature_dim
    ),
    gfb_module=dict(
        type="GCN",
        in_dim=feature_dim,
        h_dim=feature_dim,
        num_layers=1,
        dropout=0.7,
    ),
    backbone=None,
)

data = set_dataset_type(data, 'EpicFutureLabelsGFBAug')
work_dir = get_workdir(__file__)