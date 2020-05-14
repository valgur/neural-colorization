dependencies = ['torch']

import torch.hub

from model import generator as _generator

_pretrained_url = 'https://github.com/zeruniverse/neural-colorization/releases/download/1.1/G.pth'


def generator(pretrained=False, progress=True, map_location=None):
    G = _generator()
    if pretrained:
        G.load_state_dict(torch.hub.load_state_dict_from_url(
            _pretrained_url, map_location=map_location, progress=progress, check_hash=False))
    return G
