# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from common import *

model = dict(
    type='RNNModel',
    num_classes=num_classes,
    backbone=dict(
        type="mlp1D",
        num_layers=1,
        in_channels=2869,
        h_channels=2869,
        out_channels=2869,
    ),
    lstm=dict(
        type="lstm",
        num_layers=1,
        in_channels=2869,
        out_channels=2869,
        dropout=0.0,
    ),
    fc_indim=2869,
)

data = set_dataset_type(data, 'EpicFutureLabelsI3D')
work_dir = get_workdir(__file__)
