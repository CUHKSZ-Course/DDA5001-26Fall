"""Checkpoint saving and process monitoring."""
import os
import errno
from pathlib import Path
import signal
import subprocess
import threading
import time
from contextlib import contextmanager

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


class RunInterrupted(KeyboardInterrupt):
    def __init__(self, signum=signal.SIGINT):
        self.signum = signum
        super().__init__(f'Run interrupted by signal {signum}')


@contextmanager
def interruption_signals(ignore=False):
    """Allow main-thread launchers to clean up on INT/TERM/HUP, then restore handlers."""
    previous = {}
    def interrupted(signum, frame):
        raise RunInterrupted(signum)
    if threading.current_thread() is threading.main_thread():
        for name in ('SIGINT', 'SIGTERM', 'SIGHUP'):
            sig = getattr(signal, name, None)
            if sig is not None:
                previous[sig] = signal.signal(sig, signal.SIG_IGN if ignore else interrupted)
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def owned_process_alive(process):
    parent_alive = process.poll() is None
    if os.name != 'posix':
        return parent_alive
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return parent_alive
    # Linux orphan zombies cannot be signalled; they hold no GPU/worker resources.
    proc = Path('/proc')
    if proc.is_dir():
        for entry in proc.iterdir():
            if not entry.name.isdigit(): continue
            try:
                fields = (entry/'stat').read_text().rsplit(')', 1)[1].split()
                if int(fields[2]) == process.pid and fields[0] not in ('Z', 'X'):
                    return True
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
        return parent_alive
    return True


def stop_owned_process(process):
    """Never wait indefinitely for a process stuck in uninterruptible kernel I/O."""
    for sig in (signal.SIGTERM,getattr(signal,'SIGKILL',9)):
        if not owned_process_alive(process):return True
        try:
            if os.name=='posix':os.killpg(process.pid,sig)
            elif sig==signal.SIGTERM:process.terminate()
            else:process.kill()
        except ProcessLookupError:pass
        deadline = time.monotonic() + 1
        while owned_process_alive(process) and time.monotonic() < deadline:
            time.sleep(.05)
    return not owned_process_alive(process)


def run_monitored(command, *, timeout=None, cancel_event=None, **kwargs):
    """Use a file/inherited stdout, not PIPE; monitor the host while the child runs."""
    assert_kernel_healthy()
    if cancel_event is not None and cancel_event.is_set(): raise RunInterrupted()
    if os.name=='posix':
        if kwargs.get('start_new_session') is False:
            raise ValueError('Monitored children require their own session for safe group cleanup')
        kwargs['start_new_session']=True
    process = None
    with interruption_signals():
        try:
            process=subprocess.Popen(command,**kwargs)
            started=time.monotonic()
            while process.poll() is None:
                assert_kernel_healthy()
                if cancel_event is not None and cancel_event.is_set(): raise RunInterrupted()
                if timeout is not None and time.monotonic()-started > timeout:
                    raise subprocess.TimeoutExpired(command,timeout)
                time.sleep(.25)
            # A parent can exit before its compiler workers. Clean the owned group.
            with interruption_signals(ignore=True):
                if not stop_owned_process(process):
                    raise RuntimeError('Owned child process group did not stop')
            assert_kernel_healthy()
            return subprocess.CompletedProcess(command,process.returncode)
        except BaseException as exc:
            if process is not None:
                with interruption_signals(ignore=True):
                    exc.child_still_alive = not stop_owned_process(process)
            raise
