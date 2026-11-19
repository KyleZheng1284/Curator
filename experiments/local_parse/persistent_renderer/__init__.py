"""Opt-in persistent fitted renderer wiring for the isolated Page64 study."""

import atexit

from nemo_curator import _parse35_stage_trace as trace

from .client import (
    HELPER, MAX_FRAME_BYTES, PYTHON, PersistentFittedRenderer, RendererCancelled, child_environment,
)


def render(stage, request, timeout_s, cancel):
    if cancel is None:
        raise ValueError('The persistent fitted renderer requires the E03 actor-owned cancellation context')
    if len(request) > MAX_FRAME_BYTES:
        from experiments.local_parse.early_admission import cancellable_helper

        trace.emit('persistent_renderer_oversize_fallback', request_bytes=len(request),
                   frame_limit_bytes=MAX_FRAME_BYTES)
        command = [str(PYTHON), '-B', '-s', str(HELPER)]
        return cancellable_helper(command, request, child_environment(), timeout_s, cancel).stdout
    client = getattr(stage, '_persistent_fitted_client', None)
    if client is None:
        client = PersistentFittedRenderer(on_event=trace.emit)
        stage._persistent_fitted_client = client
        atexit.register(client.close)
    try:
        return client.render(request, timeout_s=timeout_s, cancel=cancel)
    except RendererCancelled as error:
        from experiments.local_parse.early_admission import EarlyAdmissionCancelled
        raise EarlyAdmissionCancelled('Cancelled fitted render range') from error


def teardown(stage):
    client = getattr(stage, '_persistent_fitted_client', None)
    if client is not None:
        client.close()
        atexit.unregister(client.close)
        stage._persistent_fitted_client = None
