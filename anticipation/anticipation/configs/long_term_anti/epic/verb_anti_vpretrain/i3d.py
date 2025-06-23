# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from common import *

model = dict(
    type='I3DModel',
    num_classes=num_classes,
    backbone=dict(
        type="mlp1D",
        num_layers=2,
        in_channels=2869,
        h_channels=2869,
        out_channels=2869
    )
)

data = set_dataset_type(data, 'EpicFutureLabelsI3D')
work_dir = get_workdir(__file__)