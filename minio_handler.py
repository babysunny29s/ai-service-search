"""MinIO for search: download a query photo (and optional upload). Credentials from env only."""

from __future__ import annotations

import asyncio
import logging
import os
from io import BytesIO
from typing import Optional
from urllib.parse import unquote, urlsplit

from minio import Minio

log = logging.getLogger(__name__)

_FOLDER = os.getenv("MINIO_SEARCH_FOLDER", "search")
_instance: MinioHandler | None = None


class MinioHandler:
    def __init__(self) -> None:
        self.endpoint = os.getenv("MINIO_ENDPOINT", "").strip()
        self.access_key = os.getenv("MINIO_ACCESS_KEY", "").strip()
        self.secret_key = os.getenv("MINIO_SECRET_KEY", "").strip()
        self.bucket_name = os.getenv("MINIO_BUCKET_NAME", "ai-inhouse")
        self.secure = os.getenv("MINIO_SECURE", "true").lower() == "true"
        self.folder = _FOLDER
        self.client = self._init_client()

    def _init_client(self) -> Optional[Minio]:
        if not (self.endpoint and self.access_key and self.secret_key):
            log.info("MinIO disabled — set MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY")
            return None
        try:
            client = Minio(self.endpoint, self.access_key, self.secret_key, secure=self.secure)
            if not client.bucket_exists(self.bucket_name):
                client.make_bucket(self.bucket_name)
            log.info("MinIO ready bucket=%s", self.bucket_name)
            return client
        except Exception as e:
            log.error("MinIO init failed: %s", e)
            return None

    def is_available(self) -> bool:
        return self.client is not None

    def object_name(self, ref: str) -> str:
        """Accept object key or a full MinIO URL."""
        ref = (ref or "").strip()
        if not ref:
            return ""
        if "://" in ref:
            path = unquote(urlsplit(ref).path).lstrip("/")
            prefix = self.bucket_name.rstrip("/") + "/"
            if path.startswith(prefix):
                path = path[len(prefix) :]
            return path
        return ref.lstrip("/")

    def download(self, ref: str) -> bytes:
        if not self.client:
            raise RuntimeError("MinIO is not configured")
        name = self.object_name(ref)
        if not name:
            raise ValueError("empty MinIO object name")
        response = self.client.get_object(self.bucket_name, name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    async def download_async(self, ref: str) -> bytes:
        return await asyncio.to_thread(self.download, ref)

    def upload(self, filepath: str, request_id: str, folder: str | None = None) -> tuple[str, bool]:
        if not self.client:
            log.warning("MinIO client not available")
            return "", False
        try:
            ext = os.path.splitext(filepath)[1]
            object_name = f"{folder or self.folder}/{request_id}{ext}"
            with open(filepath, "rb") as f:
                self.client.put_object(
                    self.bucket_name,
                    object_name,
                    f,
                    os.stat(filepath).st_size,
                    content_type=self._content_type(ext),
                )
            scheme = "https" if self.secure else "http"
            url = f"{scheme}://{self.endpoint}/{self.bucket_name}/{object_name}"
            log.info("uploaded %s", url)
            return url, True
        except Exception as e:
            log.error("MinIO upload failed: %s", e)
            return "", False

    async def upload_async(self, filepath: str, request_id: str, folder: str | None = None) -> tuple[str, bool]:
        return await asyncio.to_thread(self.upload, filepath, request_id, folder)

    def upload_bytes(self, data: bytes, request_id: str, ext: str = ".jpg", content_type: str = "image/jpeg") -> tuple[str, str, bool]:
        """Upload image bytes. Returns (public url, object name, ok)."""
        if not self.client:
            return "", "", False
        ext = ext if ext.startswith(".") else f".{ext}"
        object_name = f"{self.folder}/{request_id}{ext}"
        try:
            self.client.put_object(self.bucket_name, object_name, BytesIO(data), len(data), content_type=content_type)
            scheme = "https" if self.secure else "http"
            return f"{scheme}://{self.endpoint}/{self.bucket_name}/{object_name}", object_name, True
        except Exception as e:
            log.error("MinIO upload failed: %s", e)
            return "", "", False

    def remove(self, object_name: str) -> None:
        if not self.client or not object_name:
            return
        try:
            self.client.remove_object(self.bucket_name, object_name)
        except Exception as e:
            log.warning("MinIO delete %s failed: %s", object_name, e)

    def _content_type(self, ext: str) -> str:
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(ext.lower(), "application/octet-stream")


def get_minio() -> MinioHandler:
    global _instance
    if _instance is None:
        _instance = MinioHandler()
    return _instance
