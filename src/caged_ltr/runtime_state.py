"""Optional Redis state adapter with deterministic in-process fallback."""

from __future__ import annotations

import time
from collections import OrderedDict


class MemoryState:
    def __init__(self, max_entries: int = 1024) -> None:
        self._values: OrderedDict[str, tuple[float, object]] = OrderedDict()
        self.max_entries = max_entries

    def get(self, key: str):
        value = self._values.get(key)
        if value is None:
            return None
        expires, payload = value
        if expires and expires < time.time():
            self._values.pop(key, None)
            return None
        self._values.move_to_end(key)
        return payload

    def set(self, key: str, value, ttl_seconds: int = 300) -> None:
        self._values[key] = (time.time() + ttl_seconds if ttl_seconds else 0, value)
        self._values.move_to_end(key)
        while len(self._values) > self.max_entries:
            self._values.popitem(last=False)

    def incr(self, key: str, ttl_seconds: int = 60) -> int:
        current = int(self.get(key) or 0) + 1
        self.set(key, current, ttl_seconds)
        return current


class RedisState:
    def __init__(self, url: str) -> None:
        try:
            import redis
        except ImportError as error:
            raise RuntimeError("install redis to use RedisState") from error
        self.client = redis.Redis.from_url(url, decode_responses=True)
        self.client.ping()

    def get(self, key: str):
        return self.client.get(key)

    def set(self, key: str, value, ttl_seconds: int = 300) -> None:
        self.client.set(key, value, ex=ttl_seconds or None)

    def incr(self, key: str, ttl_seconds: int = 60) -> int:
        value = self.client.incr(key)
        if value == 1:
            self.client.expire(key, ttl_seconds)
        return int(value)


def build_state(url: str | None = None):
    if url:
        return RedisState(url)
    return MemoryState()
