"""Samplers whose index stream survives a data-loader worker restart."""
from mmengine.dataset import InfiniteSampler
from mmengine.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class OVCInfiniteSampler(InfiniteSampler):
    """Keep the shared stream alive when an abandoned iterator is closed.

    InfiniteSampler delegates with ``yield from self.indices``. Closing that
    iterator propagates GeneratorExit into the shared stream, including when
    Mosaic switching discards the old DataLoader iterator. An explicit loop
    preserves the stream and its current random order across worker restarts.
    """

    def __iter__(self):
        for index in self.indices:
            yield index
