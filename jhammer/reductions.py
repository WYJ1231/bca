# # >python3.10
# from enum import StrEnum
#
#
# class Reduction(StrEnum):
#     """
#     Reduction policies.
#     """
#
#     MEAN = "mean"
#     SUM = "sum"
#     NONE = "none"

from enum import Enum

class Reduction(Enum):
    """
    Reduction policies.
    """

    MEAN = "mean"
    SUM = "sum"
    NONE = "none"