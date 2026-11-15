#!/usr/bin/env python3
"""Execute one MBPP candidate in a credential-free, unprivileged chroot.

The parent prepares a read-only Python runtime. The fresh worker chroots, closes
inherited descriptors, drops groups/uid/gid, applies resource limits and kernel
seccomp rules before evaluating any generated text. No container engine needed.
"""
from __future__ import annotations
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import select
import shutil
import signal
import stat
import subprocess
import sys
import sysconfig
import tempfile
import time

VERSION=2
_RESULT_FD=1
SAFE_ENV={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','PYTHONDONTWRITEBYTECODE':'1',
          'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
          'MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
DENIED='socket socketpair connect bind listen accept accept4 sendto recvfrom sendmsg recvmsg sendmmsg recvmmsg shutdown setsockopt getsockopt clone clone3 fork vfork execve execveat ptrace process_vm_readv process_vm_writev process_madvise pidfd_open pidfd_getfd pidfd_send_signal kill tkill tgkill mount umount2 pivot_root chroot unshare setns bpf perf_event_open keyctl add_key request_key open_by_handle_at name_to_handle_at io_uring_setup io_uring_enter io_uring_register userfaultfd reboot swapon swapoff kexec_load kexec_file_load init_module finit_module delete_module syslog acct quotactl sethostname setdomainname'.split()
# Chroot does not isolate the kernel's IPC namespaces. Deny creation/access to
# persistent IPC objects, including raw libc/syscall bypasses of Python APIs.
DENIED += 'shmget shmat shmdt shmctl semget semop semtimedop semctl msgget msgsnd msgrcv msgctl ipc mq_open mq_unlink mq_timedsend mq_timedreceive mq_notify mq_getsetattr'.split()
# The trusted worker arms its timer before loading the filter. Candidate code
# cannot cancel it, replace/block SIGALRM, or establish competing kernel timers.
DENIED += 'setitimer alarm rt_sigaction rt_sigprocmask timer_create timer_settime timer_delete timerfd_create timerfd_settime'.split()

def sha(path):
    out=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8<<20),b''):out.update(block)
    return out.hexdigest()

def copy_readonly(source,destination):
    """Copy only runtime files; do not expose symlinks or writable hardlinks."""
    if source.is_dir():
        destination.mkdir(parents=True,exist_ok=True)
        for child in source.iterdir():
            if child.name in ('site-packages','__pycache__','tests','test','idlelib','ensurepip','tkinter'):continue
            copy_readonly(child,destination/child.name)
        destination.chmod(0o555)
    elif source.is_file():
        destination.parent.mkdir(parents=True,exist_ok=True)
        info=source.stat()
        if info.st_uid==0 and not info.st_mode & (stat.S_IWGRP|stat.S_IWOTH):
            try:os.link(source.resolve(),destination);return
            except OSError:pass
        shutil.copyfile(source,destination);destination.chmod(0o444)

def prepare(root):
    if os.geteuid()!=0:raise RuntimeError('The sandbox launcher needs chroot/setuid privileges')
    root=Path(root).resolve()
    marker=root/'runtime.json'
    if marker.exists():
        value=json.loads(marker.read_text())
        if value['version']!=VERSION or value['python']!=sys.version:raise RuntimeError('Sandbox runtime mismatch')
        for name,expected in value['runtime_files'].items():
            path=root/name
            if path.is_symlink() or not path.resolve().is_relative_to(root):raise RuntimeError('Invalid sandbox runtime path')
            observed=path.stat()
            if (observed.st_uid!=0 or observed.st_mode & (stat.S_IWGRP|stat.S_IWOTH) or
                observed.st_size!=expected['bytes'] or sha(path)!=expected['sha256']):
                raise RuntimeError(f'Sandbox runtime content or ownership changed: {name}')
        return value
    if root.exists() and any(root.iterdir()):raise RuntimeError('Refusing an incomplete or nonempty sandbox root')
    root.mkdir(parents=True,exist_ok=True)
    standard=Path(sysconfig.get_path('stdlib'))
    copy_readonly(standard,root/'python')
    packages=Path(sysconfig.get_path('purelib'))
    included={}
    for name in ('numpy','numpy.libs','scipy','scipy.libs','sympy','mpmath'):
        source=packages/name
        if source.exists():copy_readonly(source,root/'packages'/name);included[name]=str(source)
    # Compiled stdlib modules resolve their ordinary runtime libraries here.
    for pattern in ('libpthread.so*','librt.so*','libdl.so*','libm.so*','libc.so*',
                    'libgcc_s.so*','libstdc++.so*','libgfortran.so*','libquadmath.so*',
                    'libz.so*','libbz2.so*','liblzma.so*','libffi.so*','libssl.so*','libcrypto.so*'):
        for source in Path('/usr/lib/x86_64-linux-gnu').glob(pattern):
            target=root/'lib/x86_64-linux-gnu'/source.name
            if not target.exists():copy_readonly(source,target)
    for directory in ('tmp','packages','lib','lib/x86_64-linux-gnu'):
        (root/directory).mkdir(parents=True,exist_ok=True)
    (root/'tmp').chmod(0o1777)
    value=dict(version=VERSION,python=sys.version,stdlib=str(standard),packages=included,
        runtime_files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p))
                       for p in sorted(root.rglob('*')) if p.is_file()},
        isolation=dict(chroot=True,uid=65534,gid=65534,supplementary_groups=[],
                       network='seccomp socket/network syscalls denied',process_creation='seccomp denied',
                       no_new_privileges=True,credentials='minimal fixed environment; no home/workspace/proc mounts'))
    marker.write_text(json.dumps(value,indent=2)+'\n');marker.chmod(0o444)
    root.chmod(0o555)
    return value

def seccomp():
    lib=ctypes.CDLL('libseccomp.so.2',use_errno=True)
    lib.seccomp_init.argtypes=[ctypes.c_uint32];lib.seccomp_init.restype=ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];lib.seccomp_syscall_resolve_name.restype=ctypes.c_int
    lib.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
    lib.seccomp_load.argtypes=[ctypes.c_void_p];lib.seccomp_release.argtypes=[ctypes.c_void_p]
    context=lib.seccomp_init(0x7fff0000)  # SCMP_ACT_ALLOW
    if not context:raise OSError('seccomp_init failed')
    for name in DENIED:
        number=lib.seccomp_syscall_resolve_name(name.encode())
        if number>=0 and lib.seccomp_rule_add(context,0x00050000|errno.EPERM,number,0):
            raise OSError(f'seccomp rule failed: {name}')
    return lib,context

def worker(root,timeout):
    # Imports and payload IO precede isolation; no generated code is run here.
    import contextlib
    import io
    payload=json.loads(sys.stdin.buffer.read())
    global _RESULT_FD
    result_fd=os.dup(1);_RESULT_FD=result_fd
    null_fd=os.open('/dev/null',os.O_RDWR)
    for target in (0,1,2):os.dup2(null_fd,target)
    if null_fd>2:os.close(null_fd)
    os.closerange(3,result_fd)
    os.closerange(result_fd+1,resource.getrlimit(resource.RLIMIT_NOFILE)[0])
    lib,context=seccomp()
    libc=ctypes.CDLL(None,use_errno=True)
    os.environ.clear();os.environ.update(SAFE_ENV)
    os.chroot(root);os.chdir('/tmp')
    os.setgroups([]);os.setresgid(65534,65534,65534);os.setresuid(65534,65534,65534)
    if libc.prctl(38,1,0,0,0):raise OSError('PR_SET_NO_NEW_PRIVS failed')
    for what,value in ((resource.RLIMIT_AS,2<<30),(resource.RLIMIT_CPU,math.ceil(timeout)+1),
                       (resource.RLIMIT_FSIZE,1<<20),(resource.RLIMIT_NOFILE,32),
                       (resource.RLIMIT_NPROC,0),(resource.RLIMIT_CORE,0)):
        resource.setrlimit(what,(value,value))
    sys.path[:]=['/python','/python/lib-dynload','/packages']
    sys.dont_write_bytecode=True
    class CandidateTimeout(Exception):pass
    def expired(*args):raise CandidateTimeout()
    class WriteOnly(io.StringIO):
        def read(self,*args):raise OSError('input unavailable')
        def readline(self,*args):raise OSError('input unavailable')
        def readlines(self,*args):raise OSError('input unavailable')
        def write(self,text):return len(text)  # output is discarded, never accumulated
    stream=WriteOnly()
    signal.signal(signal.SIGALRM,expired)
    outcome=dict(passed=False,kind='failed',isolation_applied=True)
    begin=time.monotonic()
    try:
        signal.setitimer(signal.ITIMER_REAL,timeout)
        if lib.seccomp_load(context):raise OSError('seccomp_load failed')
        lib.seccomp_release(context)
        # Parent starts its independent deadline only after actual isolation.
        # A candidate cannot retract this already-written trusted startup line.
        os.write(result_fd,b'{"stage":"ready","isolation_applied":true}\n')
        with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream),contextlib.redirect_stdin(stream) if hasattr(contextlib,'redirect_stdin') else contextlib.nullcontext():
            sys.stdin=stream
            exec(payload['program'],{})
        outcome.update(passed=True,kind='passed')
    except CandidateTimeout:outcome['kind']='timed_out'
    except BaseException as error:
        outcome.update(kind='failed',exception_type=type(error).__name__,exception_message=str(error)[:2000])
    # Timer changes are now denied, including trusted cleanup. The process exits
    # immediately after reporting; the parent independently enforces wall time.
    outcome['elapsed_sec']=time.monotonic()-begin
    os.write(result_fd,json.dumps(outcome).encode());os.close(result_fd)

def execute(root,program,timeout=3.0):
    root=Path(root).resolve()
    if not (root/'runtime.json').is_file():raise RuntimeError('Prepare the sandbox first')
    scratch=root/'tmp'
    for path in scratch.iterdir():
        if path.is_dir() and not path.is_symlink():shutil.rmtree(path)
        else:path.unlink()
    command=[sys.executable,'-I',str(Path(__file__).resolve()),'--worker',str(root),str(timeout)]
    begin=time.monotonic()
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        env=SAFE_ENV,close_fds=True,bufsize=0)
    try:
        remaining=memoryview(json.dumps({'program':program}).encode())
        while remaining:
            written=process.stdin.write(remaining)
            remaining=remaining[written:]
        process.stdin.close();process.stdin=None
        if not select.select([process.stdout],[],[],6.)[0]:
            process.kill();process.communicate()
            return dict(passed=False,kind='sandbox_error',reason='startup_timeout')
        ready_line=process.stdout.readline()
        try:ready=json.loads(ready_line)
        except ValueError:ready={}
        if ready.get('stage')!='ready' or ready.get('isolation_applied') is not True:
            process.kill();tail,_=process.communicate()
            return dict(passed=False,kind='sandbox_error',reason='isolation_startup_failed',
                diagnostic=(ready_line+tail)[:1000].decode(errors='replace'))
        deadline=time.monotonic()+timeout
        output=bytearray()
        try:
            while True:
                remaining=deadline-time.monotonic()
                if remaining<=0 or not select.select([process.stdout],[],[],remaining)[0]:
                    raise subprocess.TimeoutExpired(command,timeout)
                chunk=os.read(process.stdout.fileno(),4096)
                if not chunk:
                    process.wait(timeout=max(0.001,deadline-time.monotonic()))
                    break
                output.extend(chunk)
                if len(output)>65536:
                    process.kill();process.communicate()
                    return dict(passed=False,kind='candidate_output_limit',isolation_applied=True,
                        result_bytes_limit=65536,elapsed_sec=time.monotonic()-begin)
        except subprocess.TimeoutExpired:
            process.kill();process.communicate()
            return dict(passed=False,kind='timed_out',timeout_enforcer='parent_wall_clock',
                isolation_applied=True,elapsed_sec=time.monotonic()-begin)
        if process.returncode:
            return dict(passed=False,kind='worker_signal' if process.returncode<0 else 'candidate_exit',
                returncode=process.returncode,isolation_applied=True,elapsed_sec=time.monotonic()-begin)
    finally:
        if process.poll() is None:
            process.kill();process.wait()
        for stream in (process.stdin,process.stdout,process.stderr):
            if stream is not None:stream.close()
    try:value=json.loads(bytes(output))
    except ValueError:return dict(passed=False,kind='candidate_exit',reason='missing_or_invalid_result',isolation_applied=True)
    if (not isinstance(value,dict) or value.get('isolation_applied') is not True
            or type(value.get('passed')) is not bool
            or value.get('kind') not in ('passed','failed','timed_out')
            or value['passed'] != (value['kind']=='passed')):
        return dict(passed=False,kind='candidate_output_format',isolation_applied=True)
    return value

if __name__=='__main__':
    if len(sys.argv)==4 and sys.argv[1]=='--worker':
        try:worker(sys.argv[2],float(sys.argv[3]))
        except BaseException as error:
            os.write(_RESULT_FD,json.dumps(dict(kind='sandbox_error',exception_type=type(error).__name__,error=str(error))).encode())
            raise SystemExit(2)
    else:raise SystemExit('Use the separate score_mbpp.py entrypoint')
