import torch
from torch.nn import Module
from torch.nn.functional import one_hot

from jhammer.reductions import Reduction


class DiceLoss(Module):
    def __init__(self,
                 reduction: str = "mean",
                 to_one_hot_y=False,
                 n_classes=-1):
        """
        Dice loss. The `input` is expected to be a `torch.Tensor` with the shape of `[BNHW(D)]`, where the first axis
        of `B` is batch size and the second axis of 'N' is the number of classes. If `to_one_hot_y` is True, the
        `target` is encoded by one-hot manner. `n_classes` is the total number of classes for one-hot encoding. Default
        is -1, the number of classes will be inferred as one greater than the largest class value in the target tensor.
        The `target` should be a shape of `[BHW(D)]` if `to_one_hot_y` is True, `[BNHW(D)]`, otherwise.

        Args:
            reduction (str, optional, default="mean"):
            to_one_hot_y (bool, optional, default=False):
            n_classes (int, optional, default=-1):
        """

        super().__init__()
        assert reduction is not None
        self.smooth_epsilon = 1e-5
        self.to_one_hot_y = to_one_hot_y
        self.n_classes = n_classes
        self.class_reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        assert len(input.shape) == len(target.shape) or len(input.shape) == len(target.shape) + 1
        if self.to_one_hot_y:
            target = target.to(torch.int64)
            target = one_hot(target, num_classes=self.n_classes)
            target_dims = torch.arange(0, len(target.shape)).tolist()
            new_target_dims = [0, target_dims[-1]] + target_dims[1:-1]
            target = torch.permute(target, dims=new_target_dims)
        reduce_axis_lst = torch.arange(2, len(input.shape)).tolist()
        intersection = torch.sum(input * target, dim=reduce_axis_lst)
        union = torch.sum(input + target, dim=reduce_axis_lst)
        f: torch.Tensor = 1 - (2. * intersection + self.smooth_epsilon) / (union + self.smooth_epsilon)
        # match self.class_reduction:
        #     case Reduction.MEAN.value:
        #         return torch.mean(f)
        #     case Reduction.SUM.value:
        #         return torch.sum(f)
        #     case _:
        #         raise ValueError(f"Unsupported reduction method {self.class_reduction}")
        if self.class_reduction == Reduction.MEAN.value:
            return torch.mean(f)
        elif self.class_reduction == Reduction.SUM.value:
            return torch.sum(f)
        else:
            raise ValueError(f"Unsupported reduction method {self.class_reduction}")

