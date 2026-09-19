"""Task-scoped SSH reuse; server197 provides the verified route to server216."""
import subprocess


def command(host):
    assert host in ('server197', 'server216')
    suffix = host.removeprefix('server')
    args = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            # Shared masters belong to long-lived transfer sessions. A short-lived
            # inspection command must never own a shared transfer connection.
            '-o', 'ControlMaster=no', '-o', 'ControlPersist=no',
            '-o', f'ControlPath=/tmp/fm4pde_ema_20260919_{suffix}.sock']
    if host == 'server216':
        args += ['-J', 'server197']
    return args + [host]


def ssh(host, remote_command, **kwargs):
    return subprocess.run(command(host) + [remote_command], check=True, **kwargs)


def pipe(source_host, source_command, target_host, target_command):
    source = subprocess.Popen(command(source_host) + [source_command], stdout=subprocess.PIPE)
    target = subprocess.Popen(command(target_host) + [target_command], stdin=source.stdout)
    source.stdout.close()
    target_code, source_code = target.wait(), source.wait()
    if target_code or source_code:
        raise RuntimeError(f'Transfer failed source={source_code} target={target_code}')
