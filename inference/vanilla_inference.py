import torch
from einops import rearrange

from jhammer.samplers import GridSampler

from inference.inference import Inference


class VanillaInference(Inference):
    def infer(self, batch, orientation):
        output = self.infer_3d_volume(batch, orientation)
        return output

    @torch.no_grad()
    def infer_3d_volume(self, batch, orientation):
        ct = batch["image"]
        ct = ct.squeeze()
        if orientation == "Transverse":
            ct = torch.einsum("hwb -> bhw", ct)
        elif orientation == "Coronal":
            ct = torch.einsum("hwb -> hwb", ct)
        else:
            ct = torch.einsum("hwb -> whb", ct)
        patch_size = (self.inference_size, ct.size(1), ct.size(2))
        sampler = GridSampler(ct, patch_size)
        output = []
        for patch in sampler:
            adjacent_slices = []
            for i in range(patch.size(0)):
                current_slice = patch[i]
                pre_slice = patch[i - 1] if i > 0 else patch[i]
                next_slice = patch[i + 1] if i < (patch.size(0) - 1) else patch[i]
                mul_slice = torch.stack([pre_slice, current_slice, next_slice],dim=0)
                adjacent_slices.append(mul_slice)
            patch = torch.stack(adjacent_slices)  # patch: torch.Size([b, 3, 512, 512])

            # # 3d
            # print("test_adjacent_slices:", patch.size())
            # patch = patch.unsqueeze(dim=1)
            # print("patch:", patch.size()) # BCHW
            # patch = rearrange(patch, "b c h w -> 1 c b h w")
            # print("patch_rearrange:", patch.size())  # BCHW

            # # 2d single_channel
            # patch = patch.unsqueeze(dim=1)
            # print("patch_size:", patch.size())
            # # output_patch, _, _ = self.model(patch.to(self.device))

            output_patch = self.model(patch.to(self.device))
            output_patch = torch.argmax(output_patch, dim=1).to(torch.uint8).cpu() # B D H W
            output.append(output_patch)

        output = sampler.restore(output)
        if orientation == "Transverse":
            output = torch.einsum("bhw -> hwb", output)
        elif orientation =="Coronal":
            output = torch.einsum("hwb -> hwb", output)
        else:
            output = torch.einsum("whb -> hwb", output)
        return output
