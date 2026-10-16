"""RabbitMQ for ai-service-search: one request queue, every search action on it."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Callable
from datetime import datetime
from typing import Any

import aio_pika

log = logging.getLogger(__name__)


def _json_log(obj: Any) -> str:
    def redact(value: Any) -> Any:
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for key, item in value.items():
                if key in ("image", "base64") and isinstance(item, str) and not item.startswith(("http://", "https://")) and len(item) > 120:
                    out[key] = f"<<base64:{len(item)} chars>>"
                else:
                    out[key] = redact(item)
            return out
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    return json.dumps(redact(obj), ensure_ascii=False, indent=2, default=str)


def _unique(*names: str | None) -> list[str]:
    out: list[str] = []
    for name in names:
        value = (name or "").strip()
        if value and value not in out:
            out.append(value)
    return out


class QueueHandler:
    def __init__(self) -> None:
        self.host = os.getenv("RABBITMQ_HOST", "localhost")
        self.port = int(os.getenv("RABBITMQ_PORT", "5672"))
        self.user = os.getenv("RABBITMQ_USER", "guest")
        self.password = os.getenv("RABBITMQ_PASS", os.getenv("RABBITMQ_PASSWORD", "guest"))
        self.prefetch_count = int(os.getenv("RABBITMQ_PREFETCH_COUNT", os.getenv("RABBITMQ_PREFETCH", "1")))
        self.consumer_timeout_seconds = int(os.getenv("RABBITMQ_CONSUMER_TIMEOUT", "300"))
        self.url = os.getenv("RABBITMQ_URL") or os.getenv("AMQP_URL") or ""

        self.request_queue = os.getenv("SEARCH_QUEUE_REQUEST", os.getenv("RABBITMQ_QUEUE", "AI_SEARCH_QUEUE_IN"))
        self.request_exchanges = _unique(
            os.getenv("RABBITMQ_EXCHANGE_IN"),
            os.getenv("SEARCH_REQUEST_EXCHANGE"),
            "AI_SEARCH_EXCHANGE_INPUT",
        )
        self.request_exchange = self.request_exchanges[0]
        self.result_queue = os.getenv("SEARCH_RESULT_QUEUE", os.getenv("RABBITMQ_RESULT_QUEUE", "AI_SEARCH_QUEUE_OUT"))
        self.result_exchanges = _unique(
            os.getenv("RABBITMQ_EXCHANGE_OUT"),
            os.getenv("SEARCH_RESULT_EXCHANGE"),
            "AI_SEARCH_EXCHANGE_RESULT",
        )
        self.result_exchange = self.result_exchanges[0]
        self.workflow_type = os.getenv("SEARCH_WORKFLOW", "process_search")
        self.dlq_routing_key = "search.dlq"

        self.connection: aio_pika.abc.AbstractRobustConnection | None = None
        self.channel: aio_pika.abc.AbstractChannel | None = None
        self.result_exchange_obj: aio_pika.abc.AbstractExchange | None = None
        self.result_exchange_objs: list[aio_pika.abc.AbstractExchange] = []

    def _amqp_url(self) -> str:
        if self.url.strip():
            return self.url.strip()
        return f"amqp://{self.user}:{self.password}@{self.host}:{self.port}/"

    async def connect(self) -> aio_pika.abc.AbstractQueue:
        connection_url = self._amqp_url()
        log.info("Connecting to RabbitMQ at %s:%s", self.host, self.port)
        self.connection = await aio_pika.connect_robust(connection_url)
        self.channel = await self.connection.channel()
        await self.channel.set_qos(prefetch_count=max(1, self.prefetch_count))

        bind_args = {"type": self.workflow_type, "x-match": "all"}
        request_exchanges = []
        for name in self.request_exchanges:
            ex = await self.channel.declare_exchange(name, aio_pika.ExchangeType.HEADERS, durable=True)
            request_exchanges.append(ex)
        self.result_exchange_objs = []
        for name in self.result_exchanges:
            ex = await self.channel.declare_exchange(name, aio_pika.ExchangeType.HEADERS, durable=True)
            self.result_exchange_objs.append(ex)
        self.result_exchange_obj = self.result_exchange_objs[0]

        dlx = await self.channel.declare_exchange("dlx_exchange", aio_pika.ExchangeType.DIRECT, durable=True)
        dlq = await self.channel.declare_queue(f"{self.request_queue}_DEAD", durable=True)
        await dlq.bind(dlx, routing_key=self.dlq_routing_key)

        request_queue = await self.channel.declare_queue(
            self.request_queue,
            durable=True,
            arguments={
                "x-dead-letter-exchange": "dlx_exchange",
                "x-dead-letter-routing-key": self.dlq_routing_key,
            },
        )
        for ex in request_exchanges:
            await request_queue.bind(ex, routing_key="", arguments=bind_args)
        log.info("bound %s ← %s (type=%s)", self.request_queue, ",".join(self.request_exchanges), self.workflow_type)

        result_queue = await self.channel.declare_queue(self.result_queue, durable=True)
        for ex in self.result_exchange_objs:
            await result_queue.bind(ex, routing_key="", arguments=bind_args)
        log.info(
            "RabbitMQ ready request=%s result=%s prefetch=%s timeout=%ss",
            self.request_queue,
            self.result_queue,
            self.prefetch_count,
            self.consumer_timeout_seconds,
        )
        return request_queue

    async def consume_messages_with_timeout_protection(self, message_handler: Callable[[dict[str, Any]], Any]) -> None:
        request_queue = await self.connect()
        assert self.channel is not None
        timeout_s = max(1, self.consumer_timeout_seconds)

        async def process_message(message: aio_pika.IncomingMessage) -> None:
            try:
                data = json.loads(message.body.decode())
                if not isinstance(data, dict):
                    data = {}
                request_id = data.get("requestId") or data.get("request_id") or "unknown"
                log.info(
                    "JSON nhận request_id=%s action=%s exchange=%s\n%s",
                    request_id,
                    data.get("action"),
                    message.exchange,
                    _json_log(data),
                )
                started = datetime.now()
                async with message.process(requeue=False, ignore_processed=True):
                    try:
                        result = await asyncio.wait_for(message_handler(data), timeout=timeout_s)
                    except asyncio.TimeoutError:
                        elapsed = (datetime.now() - started).total_seconds()
                        log.error("timeout request_id=%s after %.1fs", request_id, elapsed)
                        result = {
                            "ok": False,
                            "is_success": False,
                            "requestId": request_id,
                            "request_id": request_id,
                            "status": 504,
                            "error": f"Processing timeout after {timeout_s}s",
                        }
                    if isinstance(result, dict):
                        result.setdefault("requestId", request_id)
                        result.setdefault("request_id", request_id)
                    await self.send_result(result if isinstance(result, dict) else {"ok": True, "is_success": True, "result": result})
                    log.info("done request_id=%s in %.2fs", request_id, (datetime.now() - started).total_seconds())
            except Exception:
                log.exception("message failed")
                await message.reject(requeue=False)

        await request_queue.consume(process_message)
        log.info("consuming %s", self.request_queue)
        await asyncio.Future()

    async def send_result(self, result: dict[str, Any]) -> None:
        if not self.result_exchange_obj:
            log.warning("result exchange not ready")
            return
        result = dict(result)
        result.setdefault("is_success", bool(result.get("ok")))
        result.setdefault("timestamp", datetime.now().isoformat())
        message = aio_pika.Message(
            json.dumps(result, ensure_ascii=False).encode(),
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            correlation_id=str(result.get("requestId") or result.get("request_id") or ""),
            headers={"type": self.workflow_type},
        )
        targets = self.result_exchange_objs or ([self.result_exchange_obj] if self.result_exchange_obj else [])
        for ex in targets:
            await ex.publish(message, routing_key="")
        log.info(
            "JSON trả request_id=%s is_success=%s status=%s → %s\n%s",
            result.get("requestId") or result.get("request_id"),
            result.get("is_success"),
            result.get("status"),
            ",".join(self.result_exchanges),
            _json_log(result),
        )

    async def close(self) -> None:
        if self.connection and not self.connection.is_closed:
            await self.connection.close()
            log.info("RabbitMQ connection closed")
