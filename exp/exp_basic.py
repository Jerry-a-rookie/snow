import os

import torch

from models import (
    GRU_CI,
    GRU_CD,
    GRU_CI_Snow,
    GRU_CD_Snow,
    NLinear,
    PatchTST,
    RLinear,
    SnowNet,
    TCN,
    TSMixer,
    XLinear,
    SegRNN,
    amplifier,
    cyclenet,
    iTransformer,
    Amplifier_Snow,
    CycleNet_Snow,
    NLinear_Snow,
    PatchTST_Snow,
    RLinear_Snow,
    SegRNN_Snow,
    TCN_Snow,
    TSMixer_Snow,
    XLinear_Snow,
    iTransformer_Snow,
    iTransformer_Snow_AllAttn,
    iTransformer_Snow_Attn1,
)


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'SnowNet': SnowNet,
            'NLinear': NLinear,
            'RLinear': RLinear,
            'PatchTST': PatchTST,
            'CycleNet': cyclenet,
            'cyclenet': cyclenet,
            'TCN': TCN,
            'tcn': TCN,
            'GRU': GRU_CD,
            'gru': GRU_CD,
            'GRU-CD': GRU_CD,
            'GRU_CD': GRU_CD,
            'GRU-CI': GRU_CI,
            'GRU_CI': GRU_CI,
            'TSMixer': TSMixer,
            'tsmixer': TSMixer,
            'XLinear': XLinear,
            'xlinear': XLinear,
            'SegRNN': SegRNN,
            'segrnn': SegRNN,
            'iTransformer': iTransformer,
            'itransformer': iTransformer,
            'Amplifier': amplifier,
            'amplifier': amplifier,
            'NLinear_Snow': NLinear_Snow,
            'RLinear_Snow': RLinear_Snow,
            'GRU_Snow': GRU_CD_Snow,
            'GRU-CD-Snow': GRU_CD_Snow,
            'GRU_CD_Snow': GRU_CD_Snow,
            'GRU-CI-Snow': GRU_CI_Snow,
            'GRU_CI_Snow': GRU_CI_Snow,
            'TCN_Snow': TCN_Snow,
            'CycleNet_Snow': CycleNet_Snow,
            'PatchTST_Snow': PatchTST_Snow,
            'TSMixer_Snow': TSMixer_Snow,
            'XLinear_Snow': XLinear_Snow,
            'Amplifier_Snow': Amplifier_Snow,
            'SegRNN_Snow': SegRNN_Snow,
            'iTransformer_Snow': iTransformer_Snow,
            'iTransformer_Snow_Attn1': iTransformer_Snow_Attn1,
            'iTransformer_Snow_AllAttn': iTransformer_Snow_AllAttn,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
