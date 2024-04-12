import sys
sys.path.insert(0, '/root/my_code/neuralangelo')

import torch
import torch.nn as nn
from typing import Dict, OrderedDict

from projects.neuralangelo.utils.modules import NeuralSDF

class Transformer():
    pass

class SelfAttentionHeadLayer(nn.Module):
    pass

class OrdinaryNeuralSDF(NeuralSDF):
    def __init__(self, cfg_sdf):
        # filter cfg_sdf  for the part specific to me
        self.my_cfg = extract_specific_cfg(cfg_sdf)
        super().__init__(cfg_sdf)



