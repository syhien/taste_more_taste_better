import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from .vmamba import *
from .counting_head import CountingHead


class MAMBA4CC(nn.Module):
    def __init__(self, vmamba_path, num_classes):
        super().__init__()
        self.vmamba = VSSM(
            depths=[2, 2, 15, 2],
            dims=128,
            drop_path_rate=0.6,
            # ===================
            ssm_d_state=1,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_conv=3,
            ssm_conv_bias=False,
            forward_type="v3noz",
            # ===================
            mlp_ratio=4.0,
            # ===================
            downsample_version="v3",
            patchembed_version="v2",
            # norm_layer="ln2d",
        )
        checkpoint = torch.load(vmamba_path, "cpu", weights_only=False)
        self.vmamba.load_state_dict(checkpoint["model"])
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get("LN".lower(), None)

        self.vmamba.classifier = nn.Sequential(
            OrderedDict(
                norm=norm_layer(self.vmamba.num_features),  # B,H,W,C
                permute=(
                    Permute(0, 3, 1, 2)
                    if not self.vmamba.channel_first
                    else nn.Identity()
                ),
            )
        )
        self.vmamba.classifier.apply(self._init_weights)

        self.cls_head = nn.Sequential(
            OrderedDict(
                upsample=nn.Upsample(
                    scale_factor=4, mode="bilinear", align_corners=False
                ),
                conv1=nn.Conv2d(self.vmamba.num_features, 512, 1, 1),
                relu1=nn.ReLU(inplace=True),
                conv2=nn.Conv2d(512, num_classes, 1, 1),
            )
        )
        self.reg_head = nn.Sequential(
            OrderedDict(
                count=CountingHead(self.vmamba.num_features, 1),
            )
        )
        self.cls_head.apply(self._init_weights)
        self.reg_head.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def forward(self, x):
        x = self.vmamba(x)
        cls_score = self.cls_head(x)
        pred_den = self.reg_head(x)
        cls_score_max = cls_score.max(dim=1, keepdim=True)[0]
        cls_score = cls_score - cls_score_max
        return pred_den, cls_score


def mamba(num_classes):
    model = MAMBA4CC("vssm_base_0229_ckpt_epoch_237.pth", num_classes)
    return model
