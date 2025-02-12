# from jhammer.models import UNetPlusPlus, UNet, UNet3d, UnetCross, VisionTransformer, C2BAMUNet, UNetPlusPlus, CrossSliceAttentionUNet
from jhammer.models import VisionTransformer

# from jhammer.config import get_config
# from cfgs.args import parser
#
# args = parser.parse_args()
# cfg = get_config(args.cfg)

#
# def build_unet_plus_plus(cfg):
#     return UNetPlusPlus(in_channels=1, out_channels=2)

# def build_unet(cfg):
#     return UNet(in_channels=1, out_channels=2)

# def build_unet_cross(cfg):
#     return UnetCross(in_channels=3, out_channels=2)

# def build_unet3d(cfg):
#     return UNet3d(in_channels=1, out_channels=2)

# def build_C2BAMUNet(cfg):
#     return C2BAMUNet(input_channels = 1, num_classes=2, num_layers=5)

# def build_CrossSliceAttentionUNet(cfg):
#     return CrossSliceAttentionUNet(input_channels = 1, num_classes=2, num_layers=5)

def build_vit_base_16(cfg):
    # [512, 512, 64]
    return VisionTransformer(cfg, img_size = [512, 512, 64], num_classes = 2)

model_zoo = {
    # "unet++": build_unet_plus_plus,
    # "unet":build_unet,
    # "unet_cross":build_unet_cross,
    # "unet3d":build_unet3d,
    "vit_base_16":build_vit_base_16,
    # "C2BAMUNet":build_C2BAMUNet,
    # "CrossSliceAttentionUNet":build_CrossSliceAttentionUNet,
}
