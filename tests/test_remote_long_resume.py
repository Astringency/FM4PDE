from pathlib import Path
import socket

import pytest

from scripts.train.launch_long_resume_eval import training_process_reaped
from scripts.train.launch_remote_long_resume import canonical_completion, translated


def test_remote_process_requires_a_completed_wait():
    terminal = dict(host=socket.gethostname() + '-remote', child_pid=1, exit_code=0)
    assert not training_process_reaped(terminal)
    assert training_process_reaped(dict(terminal, wait_returned=True))


def test_remote_completion_preserves_model_identity_and_maps_paths():
    original = dict(completed_epochs=50, updates=35200, last_sha256='a' * 64,
                    checkpoint_candidates=['/mounted/pretrained/study/p/checkpoints/005.pth'],
                    best_fm_checkpoint='/mounted/pretrained/study/p/best_fm_resume.pth')
    mapped = canonical_completion(original, Path('/mounted/pretrained'), Path('/canonical/pretrained'))
    assert mapped['checkpoint_candidates'] == ['/canonical/pretrained/study/p/checkpoints/005.pth']
    assert mapped['best_fm_checkpoint'] == '/canonical/pretrained/study/p/best_fm_resume.pth'
    assert mapped['last_sha256'] == original['last_sha256']
    assert mapped['updates'] == 35200 and mapped['completed_epochs'] == 50
    assert original['best_fm_checkpoint'].startswith('/mounted/')
    with pytest.raises(ValueError):
        translated('/unrelated/model.pth', Path('/mounted'), Path('/canonical'))
