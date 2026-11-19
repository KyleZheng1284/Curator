"""Optional ordering of exact frozen Page64 parent tasks after their construction."""

import hashlib
import json
import math
import os
from pathlib import Path
import re

from nemo_curator import _parse35_stage_trace as trace


PATH_ENV = 'PARSE35_PAGE64_PRIORITY_PATH'
SHA_ENV = 'PARSE35_PAGE64_PRIORITY_SHA256'
MODE_ENV = 'PARSE35_PAGE64_PRIORITY_MODE'
ENVIRONMENT = (PATH_ENV, SHA_ENV, MODE_ENV)
SCHEMA = 'parse35.page64-parent-priority.v1'
MODES = ('original', 'longest_first')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def membership_sha256(parent):
    # Keep the PDF chunks and each exact ordered physical-page selection.
    entries = (json.loads(item) for item in parent.data)
    members = [[row['file_name'], row['source_version'], row['selected_page_indices']] for row in entries]
    return sha(json.dumps(members, sort_keys=True, ensure_ascii=True, separators=(',', ':')).encode())


def load_priority(path, expected_sha, manifest_sha):
    data = Path(path).read_bytes()
    require(type(expected_sha) is str and sha(data) == expected_sha, 'The pinned Page64 priority file changed')
    priority = json.loads(data)
    require(priority['schema'] == SCHEMA and priority['input_manifest_sha256'] == manifest_sha
            and isinstance(priority['parents'], list) and priority['parents'],
            'The Page64 priorities do not belong to this exact ordered input')
    records = {}
    for row in priority['parents']:
        index, seconds, membership = (row[key] for key in ('partition_index', 'predicted_render_s', 'membership_sha256'))
        require(type(index) is int and index >= 0 and index not in records
                and type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0
                and type(membership) is str and re.fullmatch(r'[0-9a-f]{64}', membership),
                f'Invalid or duplicate Page64 priority: {index!r}')
        records[index] = row
    return records


def configured():
    values = [os.environ.get(key) for key in ENVIRONMENT]
    if not any(values):
        return None
    require(all(values) and values[2] in MODES and Path(values[0]).is_absolute()
            and os.environ.get('CURATOR_STUDY_RASTER') == 'matched'
            and os.environ.get('PARSE35_EARLY_ADMISSION') == '1'
            and os.environ.get('PARSE35_PERSISTENT_FITTED_RENDERER') == '1',
            'Page64 priority requires an exact plan in the persistent fitted E03 recipe')
    return values


def order_parents(tasks, manifest, pages_per_task, config):
    require(pages_per_task == 64, 'Historical priorities support only original Page64 membership')
    path, expected_sha, mode = config
    priorities = load_priority(path, expected_sha, digest(manifest))
    require(set(priorities) == set(range(len(tasks))), 'The priority plan must cover every Page64 parent')
    for index, task in enumerate(tasks):
        row = priorities[index]
        require(task._metadata['partition_index'] == index and membership_sha256(task) == row['membership_sha256'],
                f'Page64 membership differs from the previously timed parent {index}')
    ordered = tasks
    if mode == 'longest_first':
        ordered = sorted(tasks, key=lambda task: (
            -priorities[task._metadata['partition_index']]['predicted_render_s'],
            task._metadata['partition_index'],
        ))
    for position, task in enumerate(ordered):
        index = task._metadata['partition_index']
        row = priorities[index]
        trace.emit('page64_parent_submission', partition_index=index, submission_index=position,
                   predicted_render_s=row['predicted_render_s'], membership_sha256=row['membership_sha256'],
                   priority_sha256=expected_sha, priority_mode=mode)
    return ordered
