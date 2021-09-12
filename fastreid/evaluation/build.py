# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from ..utils.registry import Registry

EVALUATOR_REGISTRY = Registry("EVALUATOR")
EVALUATOR_REGISTRY.__doc__ = """
Registry for evaluators, which are used for different metric & methods
The registered object must be a callable that accepts two arguments:
1. A :class:`detectron2.config.CfgNode`
It must returns an instance of :class:`DefaultEvaluator`.
"""


def build_sp_evaluator(cfg, num_query, output_dir=None):
    """
    Build a evaluator from `cfg.TEST.EVALUATOR`.
    Returns:
        an instance of :class:`DefaultEvaluator`
    """

    evaluator_name = cfg.TEST.EVALUATOR
    evaluator = EVALUATOR_REGISTRY.get(evaluator_name)(cfg, num_query, output_dir)
    return evaluator
