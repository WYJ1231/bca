from jhammer.distance_maps import distance_transform_sdf
from transforms import Transform


class SDF(Transform):
    def __init__(self, keys, normalize):
        """
        Compute signed distance function.

        Args:
            keys (str or sequence): Binary segmentation for computing SDF.
            normalize (bool): If `True`, normalize the SDF by min-max normalization.
        """

        super().__init__(keys)
        self.normalize = normalize

    def _call_fun(self, data, *args, **kwargs):
        for key in self.keys:
            segmentation = data[key]
            sdf = distance_transform_sdf(segmentation, self.normalize)
            data[f"{key}_SDF"] = sdf
        return data
