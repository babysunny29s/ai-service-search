"""In-process job store (MVP). Swap for Redis/DB when running multiple workers."""

from __future__ import annotations

import asyncio
import secrets
import time

from app.core.models import Job, JobStatus, SearchInput

RETENTION_S = 3600
MAX_JOBS = 1000


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def create(self, inp: SearchInput) -> Job:
        self._gc()
        job = Job(searchId=f"search_{secrets.token_hex(8)}", input=inp)
        self._jobs[job.searchId] = job
        return job

    def get(self, search_id: str) -> Job | None:
        return self._jobs.get(search_id)

    def attach(self, job: Job, task: asyncio.Task) -> None:
        self._tasks[job.searchId] = task
        task.add_done_callback(lambda _: self._tasks.pop(job.searchId, None))

    async def shutdown(self) -> None:
        for t in list(self._tasks.values()):
            t.cancel()

    def _gc(self) -> None:
        now = time.time()
        for sid, job in list(self._jobs.items()):
            done = job.status in (JobStatus.complete, JobStatus.failed)
            if done and now - job.updatedAt.timestamp() > RETENTION_S:
                self._jobs.pop(sid, None)
        while len(self._jobs) >= MAX_JOBS:
            oldest = min(self._jobs.values(), key=lambda j: j.updatedAt)
            self._jobs.pop(oldest.searchId, None)
