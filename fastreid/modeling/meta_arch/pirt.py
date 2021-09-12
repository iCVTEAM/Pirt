# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch.nn.functional as F

from fastreid.layers.stripe_layer import *
from fastreid.utils.weight_init import *
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from fastreid.modeling.posenets import *
from .build import META_ARCH_REGISTRY

class HeatmapProcessor(object):

    def __init__(self, cfg):

        self.count = -1
        self.num = 0
        self.num_joints = cfg.MODEL.POSENET.NUM_JOINTS
        # group all overlap pixel into one
        self.groups = cfg.MODEL.POSENET.JOINTS_GROUPS
        self.smooth_lambda = cfg.MODEL.POSENET.SMOOTH

    def get_heatmap(self, maps):
        """
            maps: [b, p, h, w]
        """
        masks = F.adaptive_max_pool2d(maps, (24, 8))

        return masks, maps

    def mask_prosses(self, masks):

        mask_list = []
        for i in range(len(self.groups)):
            mask = masks[:, self.groups[i], :, :].max(dim=1, keepdim=True).values
            mask_list.append(mask)

        masks = torch.cat(mask_list, dim=1)
        b, p = masks.shape[:2]
        confs = masks.clone().view(b, p, -1).max(dim=-1).values

        return masks, confs

@META_ARCH_REGISTRY.register()
class Pirt(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))
        self.count = -1
        self.num = 0

        # backbone
        backbone = build_backbone(cfg)
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )

        # head
        self.heads = build_heads(cfg)

        # stripe attention
        self.block = StripeLayer(2048, 512, 24)
        self.block.apply(weights_init_kaiming)

        self.agg = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(2048, 8, 512, 0.1, 'relu'),
            num_layers=self._cfg.MODEL.TEL
        )
        self.agg.apply(weights_init_kaiming)

        self.conf_layer = nn.Sequential(
            nn.Conv1d(2048, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 1, 1),
        )

        # pose network
        self.pose_net = build_posenet(cfg)
        self.processor = HeatmapProcessor(cfg)

        for param in self.pose_net.parameters():
            param.requires_grad = False

    def compute_local_features(self, features, masks):
        b, c, h, w = features.shape
        p = masks.shape[1]

        feat_list = []
        masks = masks > 0.001
        for i in range(p):

            mask = masks[:, i, :, :].clone().reshape(b, 1, h, w).repeat(1, c, 1, 1).float()
            masked_feat = mask * features
            masked_feat = F.adaptive_avg_pool2d(masked_feat, (1, 1))
            feat_list.append(masked_feat.view(b, -1))

        return feat_list

    def part_process(self, feat_list):
        """
        feat_list [p:(b, c)]
        """
        feats = torch.stack(feat_list, dim=0)
        feat_list = []
        groups = self.processor.groups
        for i in range(len(groups)):
            part_feat = feats[groups[i], :, :].clone().contiguous()
            part_feat = rearrange(self.agg(part_feat), 'p b c -> b c p 1')
            part_feat = F.adaptive_avg_pool2d(part_feat, (1, 1))
            feat_list.append(part_feat.view(part_feat.shape[0], -1))

        return torch.stack(feat_list, dim=-1)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):

        self.count = (self.count + 1) % 1
        self.num = self.num + 1

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)
        b, c, h, w= features.shape
        with torch.no_grad():
            # shape: [b, p, h, w]
            # pose_features [b, 48, 96, 32]
            heat_maps, pose_features = self.pose_net(images)
            pose_masks, ori_masks = self.processor.get_heatmap(heat_maps)
            part_masks, confs = self.processor.mask_prosses(pose_masks)
            cob_masks = pose_masks.max(dim=1, keepdim=True).values

        features = self.block(features, cob_masks)

        feat_list = self.compute_local_features( features, pose_masks )
        # shape: [b, c, p]
        feats = self.part_process(feat_list)
        std_confs = torch.sigmoid(self.conf_layer(feats))
        confs = F.normalize(std_confs.squeeze() * confs, p=1, dim=1)
        feats = feats * confs.unsqueeze(1)

        stripe_features = F.avg_pool2d(features.clone(), kernel_size=(1, 8), stride=(1, 1)).reshape(b, c, -1)
        stripe_features = self.agg(rearrange(stripe_features, 'b c p -> p b c'))
        seatures = rearrange(stripe_features, 'p b c -> b c p 1')

        patch_features = F.adaptive_max_pool2d(features.clone(), (6, 4)).reshape(b, c, -1)
        patch_features = self.agg(rearrange(patch_features, 'b c p -> p b c'))
        peatures = rearrange(patch_features, 'p b c -> b c p 1')

        result = {}

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"].to(self.device)

            if targets.sum() < 0: targets.zero_()
            outputs = self.heads(seatures, peatures, feats, ori_masks, confs=confs)

            result['outputs'] = outputs
            result['targets'] = targets

        else:
            outputs = self.heads(seatures, peatures, feats, ori_masks, confs=confs)
            result['outputs'] = outputs

        return result

    def preprocess_image(self, batched_inputs):
        r"""
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):

            images = batched_inputs["images"].to(self.device)

        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs.to(self.device)

        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        images = images.sub(self.pixel_mean).div(self.pixel_std)
        return images

    def losses(self, outs):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        outputs           = outs["outputs"]
        gt_labels         = outs["targets"]

        # model predictions
        stp_outputs       = outputs['stp_outputs']
        pth_outputs       = outputs['pth_outputs']
        key_outputs       = outputs['key_outputs']
        key_feats         = outputs['key_feats']


        loss_dict = {}
        ce_eps = self._cfg.MODEL.LOSSES.CE.EPSILON
        ce_alpha = self._cfg.MODEL.LOSSES.CE.ALPHA
        ce_scale = self._cfg.MODEL.LOSSES.CE.SCALE
        loss_dict['loss_cls'] = cross_entropy_loss(
            stp_outputs,
            gt_labels,
            ce_eps,
            ce_alpha
        ) * ce_scale

        loss_dict['loss_pth'] = cross_entropy_loss(
            pth_outputs,
            gt_labels,
            ce_eps,
            ce_alpha
        ) * ce_scale

        p = key_feats.shape[-1]
        loss_dict['loss_kc'] = 0
        for i in range(p):
            loss_dict['loss_kc'] += cross_entropy_loss(
                key_outputs[:, :, i].clone().contiguous(),
                gt_labels,
                ce_eps,
                ce_alpha
            ) * ce_scale / p

        loss_dict['loss_kt'] = 0

        for i in range(p):
            loss_dict['loss_kt'] += triplet_loss(
                key_feats[:, :, i].clone().contiguous(),
                gt_labels,
                self._cfg.MODEL.LOSSES.TRI.MARGIN,
                self._cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                self._cfg.MODEL.LOSSES.TRI.HARD_MINING,
            ) * self._cfg.MODEL.LOSSES.TRI.SCALE / p

        return loss_dict