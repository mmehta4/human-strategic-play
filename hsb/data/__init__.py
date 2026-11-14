from .schema import Record
from . import arena, games2p2k, wlb

LOADERS = {"games2p2k": games2p2k.load, "arena": arena.load, "wlb": wlb.load}


def load(name: str, **kwargs) -> list[Record]:
    return LOADERS[name](**kwargs)
