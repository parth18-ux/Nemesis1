"""Publish payments to partitioned Redis Streams: partition = crc32(payer) mod N, so one worker owns each payer."""
from __future__ import annotations
import json, time, zlib


def partition_of(p: dict, n: int) -> int:
    return zlib.crc32(f'{p["from_bank"]}|{p["from_account"]}'.encode()) % n


def publish(r, payments, partitions: int, prefix: str = "", chunk: int = 2000, stop: bool = True):
    pl = r.pipeline(transaction=False); k = 0
    for p in payments:
        pl.xadd(f"{prefix}payments:{partition_of(p, partitions)}", {"p": json.dumps(p), "enq_ns": time.time_ns()})
        k += 1
        if k % chunk == 0:
            pl.execute()
    pl.execute()
    if stop:
        for i in range(partitions):
            r.xadd(f"{prefix}payments:{i}", {"stop": 1})
    return k
