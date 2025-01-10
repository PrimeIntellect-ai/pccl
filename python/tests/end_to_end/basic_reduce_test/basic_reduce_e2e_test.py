import os
import subprocess
import sys
import time
from typing import List, Optional, Dict, Union, IO


def launch_py_process(
        script_path: str,
        args: List[str],
        env_vars: Optional[Dict[str, str]] = None
) -> subprocess.Popen:
    """
    Launches a Python process with optional environment variables and stdout forwarding.

    :param script_path: Path to the Python script to execute.
    :param args: List of arguments to pass to the script.
    :param env_vars: Dictionary of environment variables to set for the process.
    :return: A Popen object for the launched process.
    """
    # Merge the current environment with the new variables (if any)
    env = {**dict(os.environ), **(env_vars or {})}

    cmd = [sys.executable, script_path] + args
    print(f"Launching process: {cmd}")
    return subprocess.Popen(
        cmd,
        env=env
    )


def test_basic_reduce():
    peer_script_path = os.path.join(os.path.dirname(__file__), 'peer.py')
    master_script_path = os.path.join(os.path.dirname(__file__), 'master.py')

    # launch master node
    master_process = launch_py_process(master_script_path, [], {'PCCL_LOG_LEVEL': 'INFO'})
    print(f"Launched master node; PID: {master_process.pid}")

    # launch 2 peers
    process_list = []
    for rank in range(2):
        process_list.append(launch_py_process(peer_script_path, [], {'PCCL_LOG_LEVEL': 'INFO', 'RANK': str(rank)}))
        print(f"Launched peer {rank}; PID: {process_list[-1].pid}")

    # wait for all processes to finish
    for process in process_list:
        exit_code = process.wait()
        assert exit_code == 0, "Peer process exited with non-zero exit code"

    # kill master process
    master_process.kill()
    master_process.wait()
