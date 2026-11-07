"""Checkpoint saving and process monitoring."""
import os
import errno
from pathlib import Path
import signal
import subprocess
import time

FATAL_KERNEL_TAINT = (1 << 4) | (1 << 5) | (1 << 7)  # MCE, bad page, kernel died/oops


class KernelHealthError(RuntimeError):
    pass


def kernel_taint():
    path=Path('/proc/sys/kernel/tainted')
    return int(path.read_text()) if path.exists() else None


def assert_kernel_healthy():
    value=kernel_taint()
    if value is not None and value & FATAL_KERNEL_TAINT:
        raise KernelHealthError(f'Host kernel reported fatal memory/oops taint {value}. Stop new experiments and recover the host; this is not a model error.')
    return value


def drop_file_cache(descriptor):
    """Advise eviction of clean file pages only; never discard file data/blocks."""
    if not hasattr(os,'posix_fadvise'):return False
    try:
        os.posix_fadvise(descriptor,0,0,os.POSIX_FADV_DONTNEED)
        return True
    except OSError as exc:
        if exc.errno in (errno.EINVAL,errno.ENOSYS,errno.EOPNOTSUPP):return False
        raise


def atomic_torch_save(state, destination, log=lambda message: None):
    import torch
    assert_kernel_healthy()
    destination=Path(destination)
    temporary=destination.with_name(destination.name+'.tmp')
    started=time.perf_counter()
    log(f'Checkpoint {destination.name}: writing and syncing data')
    with temporary.open('wb') as stream:
        torch.save(state,stream)
        stream.flush()
        os.fsync(stream.fileno())
        released=drop_file_cache(stream.fileno())
    log(f'Checkpoint {destination.name}: clean-cache eviction hint accepted={released}')
    assert_kernel_healthy()
    log(f'Checkpoint {destination.name}: committing filename')
    os.replace(temporary,destination)
    if os.name=='posix':
        descriptor=os.open(destination.parent,os.O_RDONLY | getattr(os,'O_DIRECTORY',0))
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    elapsed=time.perf_counter()-started
    log(f'Checkpoint {destination.name}: committed in {elapsed:.3f}s')
    return elapsed


def stop_owned_process(process):
    """Never wait indefinitely for a process stuck in uninterruptible kernel I/O."""
    for sig in (signal.SIGTERM,getattr(signal,'SIGKILL',9)):
        if process.poll() is not None:return True
        try:
            if os.name=='posix':os.killpg(process.pid,sig)
            elif sig==signal.SIGTERM:process.terminate()
            else:process.kill()
        except ProcessLookupError:return True
        try:process.wait(timeout=1);return True
        except subprocess.TimeoutExpired:pass
    return process.poll() is not None


def run_monitored(command, *, timeout=None, **kwargs):
    """Use a file/inherited stdout, not PIPE; monitor the host while the child runs."""
    assert_kernel_healthy()
    if os.name=='posix':kwargs.setdefault('start_new_session',True)
    process=subprocess.Popen(command,**kwargs)
    started=time.monotonic()
    while process.poll() is None:
        try:assert_kernel_healthy()
        except KernelHealthError as exc:
            exited=stop_owned_process(process)
            raise KernelHealthError(f'{exc} Child pid={process.pid}, exited={exited}. An uninterruptible process requires host recovery.') from exc
        if timeout is not None and time.monotonic()-started > timeout:
            exited=stop_owned_process(process)
            error=subprocess.TimeoutExpired(command,timeout)
            error.child_still_alive=not exited
            raise error
        time.sleep(.25)
    assert_kernel_healthy()
    return subprocess.CompletedProcess(command,process.returncode)
