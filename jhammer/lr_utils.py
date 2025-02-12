def update_lr(lr: float, optimizer):
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]
