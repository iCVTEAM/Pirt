# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
import logging
from collections import OrderedDict
from sklearn import metrics

import numpy as np
import torch
import torch.nn.functional as F

from .evaluator import DatasetEvaluator
from .rank import evaluate_rank
from .roc import evaluate_roc
from fastreid.utils import comm
from fastreid.utils.compute_dist import build_dist

logger = logging.getLogger(__name__)

from .build import EVALUATOR_REGISTRY

@EVALUATOR_REGISTRY.register()
class BaselineEvaluator(DatasetEvaluator):
    def __init__(self, cfg, num_query, output_dir=None):
        self.cfg = cfg
        self._num_query = num_query
        self._output_dir = output_dir

        self.features = []
        self.pids = []
        self.camids = []
        super().__init__()

    def reset(self):
        self.features = []
        self.pids = []
        self.camids = []
        self.model = None

    def process(self, inputs, outputs):
        self.pids.extend(inputs["targets"])
        self.camids.extend(inputs["camids"])
        self.features.append(outputs.cpu())

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            features = comm.gather(self.features)
            features = sum(features, [])

            pids = comm.gather(self.pids)
            pids = sum(pids, [])

            camids = comm.gather(self.camids)
            camids = sum(camids, [])

            # fmt: off
            if not comm.is_main_process(): return {}
            # fmt: on
        else:
            features = self.features
            pids = self.pids
            camids = self.camids

        features = torch.cat(features, dim=0)
        # query feature, person ids and camera ids
        query_features = features[:self._num_query]
        query_pids = np.asarray(pids[:self._num_query])
        query_camids = np.asarray(camids[:self._num_query])

        # gallery features, person ids and camera ids
        gallery_features = features[self._num_query:]
        gallery_pids = np.asarray(pids[self._num_query:])
        gallery_camids = np.asarray(camids[self._num_query:])

        self._results = OrderedDict()

        dist = build_dist(query_features, gallery_features, self.cfg.TEST.METRIC)

        if self.cfg.TEST.RERANK.ENABLED:
            logger.info("Test with rerank setting")
            k1 = self.cfg.TEST.RERANK.K1
            k2 = self.cfg.TEST.RERANK.K2
            lambda_value = self.cfg.TEST.RERANK.LAMBDA

            if self.cfg.TEST.METRIC == "cosine":
                query_features = F.normalize(query_features, dim=1)
                gallery_features = F.normalize(gallery_features, dim=1)

            rerank_dist = build_dist(query_features, gallery_features, metric="jaccard", k1=k1, k2=k2)
            dist = rerank_dist * (1 - lambda_value) + dist * lambda_value

        cmc, all_AP, all_INP = evaluate_rank(dist, query_pids, gallery_pids, query_camids, gallery_camids)

        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        for r in [1, 5, 10]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1]
        self._results['mAP'] = mAP
        self._results['mINP'] = mINP

        if self.cfg.TEST.ROC_ENABLED:
            scores, labels = evaluate_roc(dist, query_pids, gallery_pids, query_camids, gallery_camids)
            fprs, tprs, thres = metrics.roc_curve(labels, scores)

            for fpr in [1e-4, 1e-3, 1e-2]:
                ind = np.argmin(np.abs(fprs - fpr))
                self._results["TPR@FPR={:.0e}".format(fpr)] = tprs[ind]

        return copy.deepcopy(self._results)


@EVALUATOR_REGISTRY.register()
class PirtEvaluator(DatasetEvaluator):
    def __init__(self, cfg, num_query, output_dir=None):
        self.cfg = cfg
        self._num_query = num_query
        self._output_dir = output_dir

        self.pids = []
        self.camids = []

        self.features = []
        self.key_feat = []
        self.confs = []
        super().__init__()

    def reset(self):
        self.pids = []
        self.camids = []

        self.features = []
        self.key_feat = []
        self.confs = []

    def process(self, inputs, outputs):
        self.pids.extend(inputs["targets"])
        self.camids.extend(inputs["camids"])

        outs = outputs['outputs']
        global_feat, patch_feat, keys, confs = outs
        feat = torch.stack([global_feat, patch_feat], dim=-1)
        self.features.append(feat.cpu())
        keys = F.normalize(keys, dim=1, p=2)
        self.key_feat.append(keys.cpu())
        self.confs.append(confs.cpu())

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            features = comm.gather(self.features)
            features = sum(features, [])

            confs = comm.gather(self.confs)
            confs = sum(confs, [])

            key_feat = comm.gather(self.key_feat)
            key_feat = sum(key_feat, [])

            pids = comm.gather(self.pids)
            pids = sum(pids, [])

            camids = comm.gather(self.camids)
            camids = sum(camids, [])

            # fmt: off
            if not comm.is_main_process(): return {}
            # fmt: on
        else:
            features = self.features
            pids = self.pids
            camids = self.camids
            key_feat = self.key_feat
            confs = self.confs

        features = torch.cat(features, dim=0)
        key_feat = torch.cat(key_feat, dim=0)
        confs    = torch.cat(confs, dim=0)

        # query feature, person ids and camera ids
        query_features = features[:self._num_query]
        query_key_feat = key_feat[:self._num_query]
        query_confs = confs[:self._num_query]
        query_pids = np.asarray(pids[:self._num_query])
        query_camids = np.asarray(camids[:self._num_query])

        # gallery features, person ids and camera ids
        gallery_features = features[self._num_query:]
        gallery_key_feat = key_feat[self._num_query:]
        gallery_confs = confs[self._num_query:]
        gallery_pids = np.asarray(pids[self._num_query:])
        gallery_camids = np.asarray(camids[self._num_query:])

        self._results = OrderedDict()

        init_dist = build_dist(query_features[..., 0], gallery_features[..., 0], "cosine", qconf=None, gconf=None)
        pth_dist = build_dist(query_features[..., 1], gallery_features[..., 1], "cosine", qconf=None, gconf=None)

        N, M, P = query_features.shape[0], gallery_features.shape[0], query_key_feat.shape[1]
        index = np.argsort(init_dist.copy(), axis=1)
        dist = torch.ones(N, M) * 1000
        dist = dist.cuda()
        with torch.no_grad():
            for i in range(N):
                if (i + 1) % 500 == 0 or i + 1 == N:
                    logger.info("processing {} query images".format(i + 1))
                q = query_key_feat[i].float().cuda()
                qc = query_confs[i].cuda()
                gc = gallery_confs[i].cuda()
                c = (qc * gc).sqrt()
                num = min(100, M)
                for j in range(num):
                    g = gallery_key_feat[index[i, j]].float().cuda()
                    D = g - q
                    dist[i, index[i, j]] = (torch.pow(D, 2).sum(0).sqrt() * c).sum()
                    del g
                del q
                torch.cuda.empty_cache()

        dist = dist.cpu().numpy()
        dist = init_dist + pth_dist + dist

        if self.cfg.TEST.RERANK.ENABLED:
            logger.info("Test with rerank setting")
            k1 = self.cfg.TEST.RERANK.K1
            k2 = self.cfg.TEST.RERANK.K2
            lambda_value = self.cfg.TEST.RERANK.LAMBDA

            if self.cfg.TEST.METRIC == "cosine":
                query_features = F.normalize(query_features, dim=1)
                gallery_features = F.normalize(gallery_features, dim=1)

            rerank_dist = build_dist(query_features, gallery_features, metric="jaccard", k1=k1, k2=k2)
            dist = rerank_dist * (1 - lambda_value) + dist * lambda_value

        cmc, all_AP, all_INP = evaluate_rank(dist, query_pids, gallery_pids, query_camids, gallery_camids)
        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        for r in [1, 5, 10, 20]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1]
        self._results['mAP'] = mAP
        self._results['mINP'] = mINP

        return copy.deepcopy(self._results)

