"""
Teeth3DS Dataset for Dental Segmentation

Author: Generated for Sonata fine-tuning on Teeth3DS
"""

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class Teeth3DSDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment",
    ]
