"""Preserve every saved checkpoint locally and publish it atomically in FIFO order.

The spool is a temporary recovery cache. Training is not complete until finish()
has published every checkpoint to the explicitly configured primary directory.
"""
from __future__ import annotations
from collections import deque
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import time

from scripts.train.resume_study import file_sha, save_checkpoint, write


class CheckpointPublisher:
    def __init__(self, output, spool, source_sha256):
        self.output, self.spool = Path(output).resolve(), Path(spool).resolve()
        assert self.output != self.spool and not self.spool.is_relative_to(self.output)
        self.spool.mkdir(parents=True, exist_ok=True)
        self._lock = (self.spool / 'publisher.lock').open('a')
        fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = dict(output=str(self.output), source_sha256=source_sha256)
        config_path = self.spool / 'config.json'
        if config_path.exists():
            assert json.loads(config_path.read_text()) == config
        else:
            write(config_path, config)
        self.jobs = []
        self._pending, self._published = deque(), {}
        self._condition = threading.Condition()
        self._error, self._active, self._stop = None, None, False
        for path in sorted(self.spool.glob('[0-9]*/job.json')):
            job = json.loads(path.read_text())
            relative = Path(job['relative_destination'])
            assert not relative.is_absolute() and '..' not in relative.parts
            assert job['source_sha256'] == source_sha256
            self.jobs.append(job)
            receipt = path.parent / 'published.json'
            if receipt.exists():
                sent = json.loads(receipt.read_text())
                assert sent['sha256'] == job['sha256'] and sent['id'] == job['id']
                self._published[job['relative_destination']] = sent
            else:
                self._pending.append(job)
        directories = [int(path.name) for path in self.spool.iterdir() if path.is_dir() and path.name.isdigit()]
        self._next = max(directories, default=-1) + 1
        self.source_sha256 = source_sha256
        self._thread = threading.Thread(target=self._worker, daemon=True, name='checkpoint-publication')
        self._thread.start()

    def check(self):
        if self._error is not None:
            raise RuntimeError('Checkpoint publication failed; recoverable files remain in the spool') from self._error

    def save_checkpoint(self, path, source, model, optimizer, scheduler, epoch, metadata):
        self.check()
        relative = Path(path).resolve().relative_to(self.output)
        directory = self.spool / f'{self._next:08d}'
        directory.mkdir()
        local = directory / 'checkpoint.pth'
        save_checkpoint(local, source, model, optimizer, scheduler, epoch, metadata,
                        recorded_output_dir=str(Path(path).parent))
        job = dict(id=self._next, relative_destination=str(relative), local_path=str(local),
                   sha256=file_sha(local), bytes=local.stat().st_size, epoch=epoch,
                   updates=metadata['updates'], source_sha256=self.source_sha256)
        write(directory / 'job.json', job)
        self._next += 1
        with self._condition:
            self.jobs.append(job)
            self._pending.append(job)
            self._condition.notify_all()

    def latest_local_checkpoint(self):
        jobs = [job for job in self.jobs if job['relative_destination'] == 'last_resume.pth']
        if not jobs:
            return None
        job = max(jobs, key=lambda value: value['id'])
        assert file_sha(job['local_path']) == job['sha256']
        return Path(job['local_path'])

    def state(self):
        with self._condition:
            return dict(queued=len(self._pending), active=self._active,
                        saved_versions=len(self.jobs), error=repr(self._error) if self._error else None)

    def _copy(self, job):
        destination = self.output / job['relative_destination']
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + f'.publishing_{job["id"]:08d}')
        digest = hashlib.sha256()
        started = time.monotonic()
        try:
            with Path(job['local_path']).open('rb') as source, temporary.open('wb') as target:
                for block in iter(lambda: source.read(8 << 20), b''):
                    target.write(block)
                    digest.update(block)
                target.flush()
                os.fsync(target.fileno())
            assert digest.hexdigest() == job['sha256']
            assert temporary.stat().st_size == job['bytes']
            temporary.replace(destination)
        except BaseException:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        receipt = dict(**job, seconds=time.monotonic() - started, published_unix=time.time())
        write(Path(job['local_path']).parent / 'published.json', receipt)
        return receipt

    def _worker(self):
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._pending or self._stop)
                    if not self._pending:
                        return
                    job = self._pending.popleft()
                    self._active = job['id']
                receipt = self._copy(job)
                with self._condition:
                    self._published[job['relative_destination']] = receipt
                    self._active = None
                    self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._error, self._active = error, None
                self._condition.notify_all()
            write(self.spool / 'error.json', dict(error=repr(error), failed_unix=time.time()))

    def finish(self):
        try:
            with self._condition:
                while self._pending or self._active is not None:
                    self.check()
                    self._condition.wait(timeout=1)
                self.check()
                self._stop = True
                self._condition.notify_all()
            self._thread.join()
            result = dict(status='complete', source_sha256=self.source_sha256,
                          published_versions=len(self.jobs), targets=self._published,
                          spool=str(self.spool), completed_unix=time.time())
            write(self.output / 'checkpoint_publication_complete.json', result)
            return result
        finally:
            if self._error is not None:
                self._thread.join()
            self._lock.close()
