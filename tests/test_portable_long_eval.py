import json
import os
import socket
import subprocess
import sys

import pytest

from scripts.train.launch_portable_long_eval import sha, terminal_ready, verified_proof


def test_local_training_receipt_does_not_override_actual_live_process():
    host = socket.gethostname()
    assert not terminal_ready(dict(exit_code=0, child_pid=os.getpid(), wait_returned=True), host)
    child = subprocess.Popen([sys.executable, '-c', 'pass'])
    assert child.wait() == 0
    assert terminal_ready(dict(exit_code=0, child_pid=child.pid), host)


def test_remote_training_requires_matching_host_and_completed_wait():
    host = socket.gethostname() + '-remote'
    terminal = dict(exit_code=0, child_pid=1, host=host)
    assert not terminal_ready(terminal, host)
    assert terminal_ready(dict(terminal, wait_returned=True), host)
    assert not terminal_ready(dict(exit_code=0, child_pid=1, wait_returned=True), host)
    with pytest.raises(AssertionError):
        terminal_ready(dict(terminal, host='different-host', wait_returned=True), host)


def test_training_proof_binds_dispatch_and_completed_state(tmp_path):
    assert not verified_proof(tmp_path, 'dispatch')
    terminal = tmp_path / 'exit.json'
    complete = tmp_path / 'training_complete.json'
    proof = tmp_path / 'portable_training_exit_proof.json'
    terminal.write_text(json.dumps(dict(exit_code=0, child_pid=123)))
    complete.write_text(json.dumps(dict(completed_epochs=50, updates=35200)))
    def bind():
        proof.write_text(json.dumps(dict(training_reaped=True, dispatch_sha256='dispatch',
            terminal_sha256=sha(terminal), training_complete_sha256=sha(complete))))
    bind()
    assert verified_proof(tmp_path, 'dispatch')
    with pytest.raises(AssertionError):
        verified_proof(tmp_path, 'other-dispatch')
    complete.write_text(json.dumps(dict(completed_epochs=49, updates=34496)))
    with pytest.raises(AssertionError):
        verified_proof(tmp_path, 'dispatch')
    bind()
    with pytest.raises(AssertionError):
        verified_proof(tmp_path, 'dispatch')
