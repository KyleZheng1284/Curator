"""Reuse the pinned fitted renderer behind a private inherited socket."""

import ctypes
import gc
from io import BytesIO
import os
import runpy
import select
import socket
import sys
import threading
from types import SimpleNamespace
import weakref

from client import (ERROR, HEADER, HELPER, KERNELS, MAX_FRAME_BYTES, READY, REQUEST,
                    RESPONSE, STOP, STOPPED, RendererProtocolError, frame_head,
                    parse_head, verify_pins)


def read_exact(sock, size, *, clean_eof=False):
    result = bytearray(size)
    cursor = 0
    while cursor < size:
        received = sock.recv_into(memoryview(result)[cursor:])
        if not received:
            if clean_eof and cursor == 0:
                return None
            raise EOFError('Fitted renderer received a truncated parent frame')
        cursor += received
    return bytes(result)


def send_frame(sock, kind, request_id, body=b''):
    sock.sendall(frame_head(kind, request_id, len(body)))
    if body:
        sock.sendall(body)


def load_helper():
    verify_pins()
    helper = runpy.run_path(str(HELPER))['main']
    bindings = runpy.run_path(str(KERNELS))

    def reuse(path):
        if path != str(KERNELS):
            raise ValueError(f'Unexpected fitted renderer binding: {path}')
        return bindings

    helper.__globals__['runpy'] = SimpleNamespace(run_path=reuse)
    return helper


def render(helper, payload):
    original_in, original_out = sys.stdin, sys.stdout
    reply = BytesIO()
    try:
        sys.stdin = SimpleNamespace(buffer=BytesIO(payload))
        sys.stdout = SimpleNamespace(buffer=reply)
        helper()
    finally:
        sys.stdin, sys.stdout = original_in, original_out
    return reply.getvalue()


def diagnostic(error):
    return f'{type(error).__name__}: {error}'.encode(errors='replace')[-2000:]


def reset_pdfium_library():
    """Match stock process-scoped PDFium state without reloading Python imports."""
    from pypdfium2 import _library_scope

    # Stock helper ownership normally settles by reference counting. Only
    # collect if a PDFium wrapper survives; do not scan the whole Python heap
    # on every short range. Never destroy the library with live C handles.
    def outstanding():
        return [info.func.__name__ for info in weakref.finalize._registry.values()
                if getattr(info.func, '__module__', '').startswith('pypdfium2')]

    active = outstanding()
    if active:
        gc.collect()
        active = outstanding()
    if active:
        raise RendererProtocolError(f'Live PDFium handles prevent library reset: {active}')
    _library_scope.destroy_lib()
    _library_scope.init_lib()


def watch_owner(owner_pid):
    """Use a process pidfd because the spawning Page64 thread is short-lived."""
    if os.getppid() != owner_pid:
        raise ProcessLookupError('The fitted renderer owner has already exited')
    # The pinned Python 3.12.11 lacks the Python binding on this host.
    libc = ctypes.CDLL(None, use_errno=True)
    open_pidfd = libc.pidfd_open
    open_pidfd.argtypes = [ctypes.c_int, ctypes.c_uint]
    open_pidfd.restype = ctypes.c_int
    owner_fd = open_pidfd(owner_pid, 0)
    if owner_fd < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    if os.getppid() != owner_pid:
        os.close(owner_fd)
        raise ProcessLookupError('The fitted renderer owner exited during identity validation')

    def exit_with_owner():
        poller = select.poll()
        poller.register(owner_fd, select.POLLIN)
        poller.poll()
        # An abruptly killed Ray worker cannot complete the framed shutdown.
        os._exit(0)

    threading.Thread(target=exit_with_owner, name='fitted-render-owner', daemon=True).start()


def serve(fd, owner_pid):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('The persistent fitted renderer requires no GPU visibility')
    with socket.socket(fileno=fd) as pipe:
        try:
            watch_owner(owner_pid)
            helper = load_helper()
        except Exception as error:
            send_frame(pipe, ERROR, 0, diagnostic(error))
            return
        send_frame(pipe, READY, 0, b'pidfd')
        previous = 0
        served = False
        while head := read_exact(pipe, HEADER.size, clean_eof=True):
            kind, request_id, size = parse_head(head)
            if (kind, request_id, size) == (STOP, 0, 0):
                send_frame(pipe, STOPPED, 0)
                return
            if kind != REQUEST or request_id <= previous or size == 0:
                raise RendererProtocolError('Invalid persistent fitted renderer request order')
            previous = request_id
            payload = read_exact(pipe, size)
            try:
                if served:
                    reset_pdfium_library()
                result = render(helper, payload)
                if len(result) > MAX_FRAME_BYTES:
                    raise RendererProtocolError('Fitted range response exceeds the frame limit')
            except Exception as error:
                # Preserve the upstream exception type and message. Per-page
                # failures are already retained by the upstream helper itself.
                send_frame(pipe, ERROR, request_id, diagnostic(error))
            else:
                send_frame(pipe, RESPONSE, request_id, result)
            finally:
                # The next range must not keep this range's PDF or crops alive.
                payload = None
                result = None
                served = True


if __name__ == '__main__':
    try:
        serve(int(sys.argv[1]), int(sys.argv[2]))
    except (BrokenPipeError, ConnectionResetError):
        # The owner cancelled and closed its private socket.
        pass
