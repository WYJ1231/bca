class Inference:
    def __init__(self, model, inference_size, device):
        self.model = model
        self.inference_size = inference_size
        self.device = device
        self.model.eval()

    def infer(self, batch):
        pass
