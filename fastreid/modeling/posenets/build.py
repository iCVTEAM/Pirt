# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from ...utils.registry import Registry

POSENET_REGISTRY = Registry("POSENET")
POSENET_REGISTRY.__doc__ = """
Registry for posenets, which extract feature maps from images
The registered object must be a callable that accepts two arguments:
1. A :class:`detectron2.config.CfgNode`
2. A :class:`detectron2.layers.ShapeSpec`, which contains the input shape specification.
It must returns an instance of :class:`Posenet`.
"""

def build_posenet(cfg):
    """
    Build a posenet from `cfg.MODEL.POSENET.NAME`.
    Returns:
        an instance of :class:`posenet`
    """

    posenet_name = cfg.MODEL.POSENET.NAME
    posenet = POSENET_REGISTRY.get(posenet_name)(cfg)
    return posenet
