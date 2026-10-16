import time
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class TTLCache(Generic[T]):
    def __init__(self, ttl_s: float, max_items: int = 2000):
        self.ttl_s = ttl_s
        self.max_items = max_items
        self._data: dict[Any, tuple[float, T]] = {}

    def get(self, key: Any) -> T | None:
        hit = self._data.get(key)
        if not hit:
            return None
        exp, value = hit
        if exp < time.monotonic():
            self._data.pop(key, None)
            return None
        return value

    def set(self, key: Any, value: T) -> None:
        if len(self._data) >= self.max_items:
            now = time.monotonic()
            for k in [k for k, (exp, _) in self._data.items() if exp < now]:
                self._data.pop(k, None)
            while len(self._data) >= self.max_items:
                self._data.pop(next(iter(self._data)))
        self._data[key] = (time.monotonic() + self.ttl_s, value)
