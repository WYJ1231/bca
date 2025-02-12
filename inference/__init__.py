from inference.vanilla_inference import VanillaInference

inference_zoo = {
    'unet': VanillaInference,
    'vit_base_16': VanillaInference,
    'unet_cross': VanillaInference,
    'unet3d': VanillaInference,
    'unet++': VanillaInference,
    'C2BAMUNet': VanillaInference,
    'CrossSliceAttentionUNet': VanillaInference,
    'ganet': VanillaInference,
}
