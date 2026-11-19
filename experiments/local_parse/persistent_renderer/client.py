"""CPU-only persistent transport for the original fitted PDF render helper."""

from contextlib import suppress
import hashlib
import math
import os
from pathlib import Path
import select
import socket
import struct
import subprocess
import threading
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
REFERENCE = ROOT / 'nrl-curator-same-pixels-20261118'
NRL = ROOT / 'nrl-rtx-2865-profile-20261118'
PYTHON = NRL / 'runtime/bin/python'
HELPER = REFERENCE / 'curator_render_helper.py'
KERNELS = REFERENCE / 'shared-prepare-images.py'
PINS = {
    HELPER: 'f84dc5186d7a6848f4473d707ce4ad3def103fcf73055ed2da5724e0dcf01dc3',
    KERNELS: '9afa18e95a5eb23f646eeb78400810eee1fc72846e8c45b4715925c806e6e398',
}
MAX_FRAME_BYTES = 256 * 1024**2
HEADER = struct.Struct('!4sBQQ')
MAGIC = b'PFR1'
READY, REQUEST, RESPONSE, ERROR, STOP, STOPPED = range(6)
POLL_S = .05
TERM_GRACE_S = .25
REAP_S = 2


class RendererCancelled(Exception):
    """The owning actor requested cancellation. This is not a render failure."""


class RendererProtocolError(RuntimeError):
    """An owned helper failed to preserve its request/reply protocol."""


class RendererRangeError(RuntimeError):
    """The upstream helper raised before producing a complete range."""


def verify_pins():
    for path, pinned in PINS.items():
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != pinned:
            raise ValueError(f'Pinned fitted renderer changed: {path}')


def child_environment():
    return dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(NRL / 'source/nemo_retriever/src'),
                OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                PYTHONDONTWRITEBYTECODE='1')


def frame_head(kind, request_id, size):
    if not 0 <= size <= MAX_FRAME_BYTES:
        raise RendererProtocolError(f'Frame exceeds configured {MAX_FRAME_BYTES}-byte bound: {size}')
    return HEADER.pack(MAGIC, kind, request_id, size)


def parse_head(payload):
    magic, kind, request_id, size = HEADER.unpack(payload)
    if magic != MAGIC or size > MAX_FRAME_BYTES:
        raise RendererProtocolError('Invalid or oversized fitted renderer frame')
    return kind, request_id, size


class PersistentFittedRenderer:
    """One client per actor. Each call admits one whole fitted render range."""

    def __init__(self, on_event=None):
        verify_pins()
        self._command = [str(PYTHON), '-B', '-s', str(HERE / 'worker.py')]
        self._lock = threading.Lock()
        self._closing = threading.Event()
        self._closed = False
        self._process = None
        self._socket = None
        self._next_id = 0
        self._generation = 0
        self._worker_birth = None
        self._on_event = on_event

    @property
    def worker_pid(self):
        return self._process.pid if self._process is not None else None

    @property
    def generation(self):
        return self._generation

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _remaining(self, deadline, timeout_s, cancel, *, closing=False):
        if not closing and (self._closing.is_set() or cancel is not None and cancel.is_set()):
            raise RendererCancelled('Parent cancelled fitted range rendering')
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(self._command, timeout_s)
        return remaining

    def _transfer(self, buffer, *, write, deadline, timeout_s, cancel, closing=False):
        sock = self._socket
        if sock is None:
            raise RendererProtocolError('No active fitted renderer connection')
        view = memoryview(buffer)
        poller = select.poll()
        poller.register(sock, select.POLLOUT if write else select.POLLIN)
        try:
            while view:
                wait = min(POLL_S, self._remaining(deadline, timeout_s, cancel, closing=closing))
                if not poller.poll(math.ceil(wait * 1000)):
                    continue
                try:
                    nbytes = sock.send(view) if write else sock.recv_into(view)
                except BlockingIOError:
                    continue
                if nbytes == 0:
                    raise RendererProtocolError('Fitted renderer closed the connection during a frame')
                view = view[nbytes:]
        except (ConnectionError, OSError) as error:
            raise RendererProtocolError(f'Fitted renderer socket transfer failed: {error}') from error
        finally:
            view.release()

    def _send(self, kind, request_id, payload, deadline, timeout_s, cancel=None, *, closing=False):
        self._transfer(frame_head(kind, request_id, len(payload)), write=True, deadline=deadline,
                       timeout_s=timeout_s, cancel=cancel, closing=closing)
        if payload:
            self._transfer(payload, write=True, deadline=deadline, timeout_s=timeout_s,
                           cancel=cancel, closing=closing)

    def _receive(self, deadline, timeout_s, cancel=None, *, closing=False):
        head = bytearray(HEADER.size)
        self._transfer(head, write=False, deadline=deadline, timeout_s=timeout_s, cancel=cancel, closing=closing)
        kind, request_id, size = parse_head(head)
        body = bytearray(size)
        if size:
            self._transfer(body, write=False, deadline=deadline, timeout_s=timeout_s, cancel=cancel, closing=closing)
        return kind, request_id, bytes(body)

    def _start(self, deadline, timeout_s, cancel):
        self._remaining(deadline, timeout_s, cancel)
        peer, worker = socket.socketpair()
        try:
            self._process = subprocess.Popen(
                [*self._command, str(worker.fileno()), str(os.getpid())], pass_fds=(worker.fileno(),),
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env=child_environment(),
            )
        except BaseException:
            peer.close()
            raise
        finally:
            worker.close()
        self._socket = peer
        peer.setblocking(False)
        self._generation += 1
        readiness = self._receive(deadline, timeout_s, cancel)
        if readiness[:2] == (ERROR, 0):
            raise RendererProtocolError('Fitted renderer startup failed: ' + readiness[2].decode(errors='replace'))
        if readiness != (READY, 0, b'pidfd'):
            raise RendererProtocolError('Fitted renderer failed its readiness handshake')
        with open(f'/proc/{self._process.pid}/stat', encoding='utf-8') as stream:
            self._worker_birth = int(stream.read().rsplit(') ', 1)[1].split()[19])
        self._event('persistent_renderer_started', self._process)

    def _event(self, kind, proc, **fields):
        if self._on_event is not None and self._worker_birth is not None:
            self._on_event(kind, renderer_pid=proc.pid, renderer_birth_ticks=self._worker_birth,
                           renderer_generation=self._generation, owner_watch='pidfd', **fields)

    def _discard(self):
        sock, self._socket = self._socket, None
        if sock is not None:
            with suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            sock.close()
        proc = self._process
        if proc is not None:
            if proc.poll() is None:
                with suppress(ProcessLookupError):
                    proc.terminate()
                try:
                    proc.wait(timeout=TERM_GRACE_S)
                except subprocess.TimeoutExpired:
                    with suppress(ProcessLookupError):
                        proc.kill()
                    try:
                        proc.wait(timeout=REAP_S)
                    except subprocess.TimeoutExpired as error:
                        raise RendererProtocolError(f'Owned fitted renderer PID {proc.pid} did not reap') from error
            self._process = None
            self._event('persistent_renderer_reaped', proc)
            self._worker_birth = None

    def render(self, request, *, timeout_s, cancel=None):
        """Return the unchanged one-shot helper response payload for one range."""
        if type(request) is not bytes or not 0 < len(request) <= MAX_FRAME_BYTES:
            raise ValueError('Fitted renderer requires one bounded, nonempty upstream request payload')
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('Fitted renderer requires a finite positive timeout')
        deadline = time.monotonic() + timeout_s
        self._remaining(deadline, timeout_s, cancel)
        if not self._lock.acquire(blocking=False):
            raise RendererProtocolError('One active range per fitted renderer is required')
        try:
            self._remaining(deadline, timeout_s, cancel)
            if self._process is not None and self._process.poll() is not None:
                self._discard()
            if self._process is None:
                self._start(deadline, timeout_s, cancel)
            self._next_id += 1
            request_id = self._next_id
            self._event('persistent_renderer_range_begin', self._process, range_ordinal=request_id)
            self._send(REQUEST, request_id, request, deadline, timeout_s, cancel)
            status, replied_id, content = self._receive(deadline, timeout_s, cancel)
            self._remaining(deadline, timeout_s, cancel)
            if replied_id != request_id or status not in (RESPONSE, ERROR):
                raise RendererProtocolError('Fitted renderer reply identity or type differed')
            if status == ERROR:
                raise RendererRangeError('Shared raster helper failed: ' + content.decode(errors='replace'))
            self._event('persistent_renderer_range_ready', self._process, range_ordinal=request_id)
            return content
        except BaseException:
            # An interrupted frame cannot be safely re-used. Discard it before
            # returning control to the stock retry or cancellation policy.
            self._discard()
            raise
        finally:
            self._lock.release()

    def close(self):
        """Cancel any active range and synchronously reap the owned worker."""
        self._closing.set()
        if not self._lock.acquire(timeout=REAP_S + TERM_GRACE_S + 1):
            raise RendererProtocolError('Timed out waiting for fitted renderer range teardown')
        try:
            if self._closed:
                return
            proc = self._process
            if proc is not None and proc.poll() is None and self._socket is not None:
                try:
                    limit = time.monotonic() + TERM_GRACE_S
                    self._send(STOP, 0, b'', limit, TERM_GRACE_S, closing=True)
                    if self._receive(limit, TERM_GRACE_S, closing=True) == (STOPPED, 0, b''):
                        proc.wait(timeout=TERM_GRACE_S)
                except (RendererProtocolError, subprocess.TimeoutExpired):
                    pass
            self._discard()
            self._closed = True
        finally:
            self._lock.release()
