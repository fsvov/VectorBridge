import json
import os
from typing import Any, Optional

import redis


class RedisCache:
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.key_prefix = os.getenv("REDIS_KEY_PREFIX", "vectorbridge")
        self.default_ttl = int(os.getenv("REDIS_CACHE_TTL_SECONDS", "300"))
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = redis.Redis.from_url(
                self.redis_url, decode_responses=True,
                socket_connect_timeout=5, socket_timeout=5,
                retry_on_timeout=True, health_check_interval=30,
            )
        return self._client

    def _reset_client(self):
        """连接异常时重置客户端，下次调用重新建立连接。"""
        try:
            if self._client:
                self._client.close()
        except Exception:
            pass
        self._client = None

    def _key(self, key: str) -> str:
        return f"{self.key_prefix}:{key}"

    def _retry_once(self, fn):
        """执行 Redis 操作，超时或连接错误时重置连接并重试一次。"""
        try:
            return fn()
        except Exception:
            self._reset_client()
            try:
                return fn()
            except Exception:
                return None

    def get_json(self, key: str) -> Optional[Any]:
        def _op():
            value = self._get_client().get(self._key(key))
            return json.loads(value) if value else None
        return self._retry_once(_op)

    def set_json(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        def _op():
            payload = json.dumps(value, ensure_ascii=False)
            self._get_client().setex(self._key(key), ttl or self.default_ttl, payload)
        self._retry_once(_op)

    def delete(self, key: str) -> None:
        def _op():
            self._get_client().delete(self._key(key))
        self._retry_once(_op)

    def delete_pattern(self, pattern: str) -> None:
        def _op():
            full_pattern = self._key(pattern)
            keys = self._get_client().keys(full_pattern)
            if keys:
                self._get_client().delete(*keys)
        self._retry_once(_op)


cache = RedisCache()
