# Copyright (c) Tencent Inc. All rights reserved.
from torch import nn

from mmyolo.models.layers import ExpMomentumEMA
from mmyolo.registry import MODELS


@MODELS.register_module()
class IntegerSafeExpMomentumEMA(ExpMomentumEMA):
    """YOLO exponential EMA that synchronizes non-floating buffers.

    The upstream YOLO EMA averages floating-point state but leaves integer and
    boolean buffers unchanged after the first update. Counters used by model
    warmups must instead track the source model exactly.
    """

    def update_parameters(self, model: nn.Module) -> None:
        if self.steps == 0:
            for key, averaged in self.avg_parameters.items():
                source = self.src_parameters[key].data.to(averaged.device)
                averaged.data.copy_(source)
        elif self.steps % self.interval == 0:
            for key, averaged in self.avg_parameters.items():
                source = self.src_parameters[key].data.to(averaged.device)
                if averaged.dtype.is_floating_point:
                    self.avg_func(averaged.data, source, self.steps)
                else:
                    averaged.data.copy_(source)

        if not self.update_buffers:
            for averaged, source in zip(self.module.buffers(),
                                        self.src_buffers):
                averaged.data.copy_(source.data.to(averaged.device))
        self.steps += 1
