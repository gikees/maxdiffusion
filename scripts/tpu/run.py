import argparse
import logging
import os
import shlex
import signal
import subprocess
import threading
import time
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

GIT_USER = os.getenv("GIT_USER")
GIT_KEY = os.getenv("GIT_KEY")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "nyu-visionx")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "persist")
_WANDB_API = None


class TestsFailedError(RuntimeError):
    """Raised when remote test execution fails during full TPU setup."""


def parse_args():
    parser = argparse.ArgumentParser(description="Run maxdiffusion on TPU")

    # Resource configuration
    parser.add_argument(
        "--resource-name", type=str, required=True, help="Name of the TPU resource"
    )
    parser.add_argument(
        "--disk-name",
        type=str,
        default="solaris-dev-pd-1",
        help="Name of the disk to attach",
    )
    parser.add_argument(
        "--mount-spec-name",
        type=str,
        default="sdb",
        help="Disk specification name to mount",
    )
    parser.add_argument(
        "--gcp-zone",
        type=str,
        default="us-central1-a",
        help="GCP zone for the TPU resource",
    )
    parser.add_argument(
        "--storage_bucket",
        type=str,
        default="solaris-central1",
        help="Cloud Storage bucket to mount via gcsfuse",
    )
    parser.add_argument(
        "--gcp-project",
        type=str,
        default="nyu-vision-lab",
        help="GCP project for the TPU resource",
    )

    # Git configuration
    parser.add_argument(
        "--git-key-path",
        type=str,
        default=None,
        help="Path to the SSH private key for cloning the repo (loaded into ssh-agent via agent forwarding). Not needed when GIT_USER/GIT_KEY env vars are set.",
    )
    parser.add_argument(
        "--git-repo-url",
        type=str,
        default="git@github.com:gikees/maxdiffusion.git",
        help="Git repository URL",
    )
    parser.add_argument(
        "--git-branch", type=str, default="main", help="Git branch to pull"
    )
    parser.add_argument(
        "--git-commit", type=str, default=None, help="Git commit to checkout"
    )

    # Run configuration
    parser.add_argument(
        "--run-command", type=str, help="Command to run on the TPU", required=True
    )
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="W&B run id (required only when --retry is set).",
    )
    parser.add_argument(
        "--run-poll-interval",
        type=int,
        default=900,
        help="Seconds between monitoring checks (W&B and TPU).",
    )
    parser.add_argument(
        "--run-process-regex",
        type=str,
        default=r"([[:space:]]|^)([^ ]*/)?[^ ]*(main|train|inference)[^ ]*\.py([[:space:]]|$)",
        help="Regex used to detect active training process on TPU workers.",
    )
    parser.add_argument(
        "--check-time",
        type=int,
        default=120,
        help="Time between polling checks (seconds)",
    )
    parser.add_argument(
        "--conda-env-name",
        type=str,
        default="maxdiffusion",
        help="Conda env name to create/activate on TPU VM.",
    )
    parser.add_argument(
        "--no-setup",
        action="store_true",
        default=False,
        help="Skip setup and only run the command",
    )
    parser.add_argument(
        "--retry",
        action="store_true",
        default=False,
        help="Enable wandb monitoring and retrying of remote commands",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        default=False,
        help="Skip running tests during setup",
    )

    args = parser.parse_args()

    # Derive repo directory from URL
    args.git_repo_dir = args.git_repo_url.split("/")[-1].replace(".git", "")
    # Construct prefix
    args.prefix = (
        f"gcloud alpha compute tpus tpu-vm ssh {args.resource_name} "
        f"--zone={args.gcp_zone} --project={args.gcp_project} "
        f"--ssh-flag='-A' "
    )

    return args


def _has_working_ssh_agent() -> bool:
    """Return True if ssh-agent is reachable in this process environment."""
    probe = subprocess.run(
        ["ssh-add", "-l"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    # ssh-add -l returns:
    #   0 -> agent reachable with identities
    #   1 -> agent reachable but no identities
    #   2 -> cannot connect to agent
    return probe.returncode in (0, 1)


def _start_ssh_agent_for_process() -> None:
    """Start ssh-agent and export its env vars into this Python process."""
    proc = subprocess.run(
        ["ssh-agent", "-s"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to start ssh-agent: {proc.stderr.strip()}")

    for line in proc.stdout.splitlines():
        if line.startswith("SSH_AUTH_SOCK="):
            os.environ["SSH_AUTH_SOCK"] = line.split(";", 1)[0].split("=", 1)[1]
        elif line.startswith("SSH_AGENT_PID="):
            os.environ["SSH_AGENT_PID"] = line.split(";", 1)[0].split("=", 1)[1]

    if not _has_working_ssh_agent():
        raise RuntimeError(
            "ssh-agent started but is still unreachable; check local SSH setup."
        )


def ensure_ssh_key_loaded(git_key_path: str) -> None:
    """Ensure ssh-agent is available and load the configured git SSH key."""
    expanded_key_path = os.path.expanduser(git_key_path)
    if not os.path.exists(expanded_key_path):
        raise FileNotFoundError(f"SSH key not found: {expanded_key_path}")

    if not _has_working_ssh_agent():
        logging.info("No reachable ssh-agent found; starting a new agent.")
        _start_ssh_agent_for_process()

    add_proc = subprocess.run(
        ["ssh-add", expanded_key_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if add_proc.returncode != 0:
        raise RuntimeError(
            f"Failed to load SSH key with ssh-add ({expanded_key_path}): "
            f"{add_proc.stderr.strip()}"
        )


def run_gcloud_command_with_timeout(
    command: str, timeout_seconds: Optional[int] = 1200
) -> int:
    """
    Execute a shell command with a timeout, terminating the whole process group if exceeded.
    """
    proc = subprocess.Popen(command, shell=True, start_new_session=True)
    try:
        return_code = proc.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        terminate_process(proc)
        raise TimeoutError(f"Command timed out after {timeout_seconds}s: {command}")
    if return_code != 0:
        raise RuntimeError(f"Shell command failed with code {return_code}: {command}")
    return return_code


def run_gcloud_command_output_with_timeout(
    command: str, timeout_seconds: int = 300
) -> str:
    """
    Execute a shell command, returning stdout as text, with a timeout.
    On timeout, terminates the whole process group and raises TimeoutError.
    On non-zero exit, raises RuntimeError with stderr context.
    """
    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        terminate_process(proc)
        raise TimeoutError(f"Command timed out after {timeout_seconds}s: {command}")
    stdout_text = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
    stderr_text = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
    if proc.returncode != 0:
        raise RuntimeError(
            f"Shell command failed with code {proc.returncode}: {command}\n"
            f"stderr: {stderr_text[-2000:]}"
        )
    return stdout_text


def format_gcloud_command(args, command):
    return f"{args.prefix} --command={shlex.quote(command)} --worker=all"


def attach_disk(args):
    return f"gcloud alpha compute tpus tpu-vm attach-disk \
        {args.resource_name} \
        --disk {args.disk_name} \
        --zone {args.gcp_zone} \
        --mode read-only"


def mount_disk(args, disk_spec, folder_name):
    command = f"""
    sudo mkdir -p /mnt/disks/{folder_name}; sudo mount -o ro,noload /dev/{disk_spec} /mnt/disks/{folder_name};
    """
    return format_gcloud_command(args, command)


def ensure_gcsfuse_installed(args):
    command = """
    if ! command -v gcsfuse >/dev/null; then
        echo 'Installing gcsfuse...'
        attempt=0
        while ! command -v gcsfuse >/dev/null; do
            if [ $attempt -gt 0 ]; then
                echo 'Previous gcsfuse attempt failed - waiting 10 s before retry...'
                sleep 10
            fi
            attempt=$((attempt + 1))
            export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`;
            echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list;
            curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc;
            sudo apt-get update;
            echo "calling apt-get install -y gcsfuse...";

            install_output=$(sudo apt-get install -y gcsfuse 2>&1)
            install_exit_code=$?

            if [ $install_exit_code -ne 0 ]; then
                echo "apt-get install failed with exit code: $install_exit_code"
                echo "Error output: $install_output"

                if echo "$install_output" | grep -q "Could not get lock.*is held by process"; then
                    echo "Detected dpkg lock error, attempting to resolve..."
                    pid=$(echo "$install_output" | grep -o "process [0-9]*" | grep -o "[0-9]*")

                    if [ -n "$pid" ]; then
                        echo "Found blocking process ID: $pid"
                        echo "Killing process $pid..."
                        sudo kill -9 $pid
                        echo "Waiting 5 seconds for process cleanup..."
                        sleep 5
                        echo "Restarting installation cycle from beginning..."
                        continue
                    else
                        echo "Could not extract process ID from error message"
                        echo "Original error: $install_output"
                    fi
                else
                    echo "Non-dpkg lock error encountered: $install_output"
                fi
            fi
        done
    else
        echo 'gcsfuse is already installed.'
    fi
    """

    return format_gcloud_command(args, command)


def mount_cloud_bucket(args):
    command = f"""
    sudo mkdir -p /mnt/disks/gcs;
    sudo gcsfuse -o allow_other,default_permissions --file-mode=666 --dir-mode=777 --implicit-dirs {args.storage_bucket} /mnt/disks/gcs;
    """
    return format_gcloud_command(args, command)


def _to_https_url(url: str, user: str, key: str) -> str:
    """Convert an SSH or HTTPS git URL to an authenticated HTTPS URL."""
    if url.startswith("git@"):
        # git@github.com:owner/repo.git -> https://user:key@github.com/owner/repo.git
        url = url.replace(":", "/", 1).replace("git@", "https://", 1)
    return url.replace("https://", f"https://{user}:{key}@", 1)


def pull_repo(args):
    if GIT_USER and GIT_KEY:
        repo_url = _to_https_url(args.git_repo_url, GIT_USER, GIT_KEY)
        cmd = f"rm -rf {args.git_repo_dir}; git clone {repo_url} -b {args.git_branch}; cd {args.git_repo_dir}"
    else:
        cmd = (
            f"ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null; "
            f"rm -rf {args.git_repo_dir}; git clone {args.git_repo_url} -b {args.git_branch}; cd {args.git_repo_dir}"
        )
    if args.git_commit:
        cmd += f"; git checkout {args.git_commit}"
    return format_gcloud_command(args, cmd)


def install_dependencies(args):
    # maxdiffusion ships its own installer (uv + jax[tpu]); run it inside the conda env,
    # then make the package editable so the cloned source is what's imported.
    command = (
        f"cd {args.git_repo_dir}; "
        f"source $HOME/miniconda3/bin/activate {args.conda_env_name}; "
        "bash setup.sh MODE=stable DEVICE=tpu; "
        "python3 -m uv pip install --no-deps -e ."
    )
    return format_gcloud_command(args, command)


def run_tests(args):
    # Ensure the script is runnable even if git doesn't preserve exec bits.
    command = (
        f"cd {args.git_repo_dir}; "
        "chmod +x ./src/tests/run_tests.sh || true; "
        f"source $HOME/miniconda3/bin/activate {args.conda_env_name}; "
        "./src/tests/run_tests.sh"
    )
    return format_gcloud_command(args, command)


def install_conda(args):
    command = (
        'if [ ! -d "$HOME/miniconda3" ]; then '
        "    echo 'Installing conda...'; "
        "    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; "
        "    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3; "
        "    rm Miniconda3-latest-Linux-x86_64.sh; "
        "else "
        "    echo 'Conda is already installed.'; "
        "fi;"
    )
    return format_gcloud_command(args, command)


def setup_environment(args):
    env = args.conda_env_name
    command = f"cd {args.git_repo_dir}; \
        source $HOME/miniconda3/bin/activate; \
        if ! conda env list | grep -q '^{env}[[:space:]]'; then \
        echo 'Creating conda environment {env} (python 3.12)...'; \
        CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes conda create -y -n {env} python=3.12; \
        else \
        echo 'Conda environment {env} already exists.'; \
        fi; \
        conda activate {env}; \
        source $HOME/miniconda3/bin/activate {env}; \
        sudo mkdir -p /tmp/tpu_logs/ && sudo chmod -R 777 /tmp/tpu_logs/;"
    return format_gcloud_command(args, command)


def kill_python_process(args):
    command = (
        "sudo pkill -9 python; "
        "sudo lsof -w /dev/accel0 2>/dev/null | grep .py | "
        "awk '{print \"sudo kill -9 \" $2}' | sh; "
        "sudo rm -f /tmp/libtpu_lockfile"
    )
    return format_gcloud_command(args, command)


def clear_space(args):
    command = (
        "sudo bash -c '"
        "for d in /home/*; do "
        '[ -d "$d" ] || continue; '
        'case "$d" in '
        '"/home/$SUDO_USER"|"/home/tpu-runtime"|"/home/ubuntu"|"/home/root") continue ;; '
        "esac; "
        'if [ "$(du -s "$d" | cut -f1)" -ge 1048576 ]; then '
        'echo "Removing $d"; '
        'rm -rf -- "$d"; '
        "fi; "
        "done'"
    )
    return format_gcloud_command(args, command)


def run_command(args):
    wandb_export = (
        f"export WANDB_API_KEY={shlex.quote(WANDB_API_KEY)}; " if WANDB_API_KEY else ""
    )
    command = f"{wandb_export} \
        cd {args.git_repo_dir} ; touch config/local.yaml ; \
        source $HOME/miniconda3/bin/activate {args.conda_env_name}; \
        {args.run_command} "
    return format_gcloud_command(args, command)


def perform_setup(args, full_setup: bool) -> None:
    if full_setup:
        run_gcloud_command_with_timeout(clear_space(args))

    run_gcloud_command_with_timeout(kill_python_process(args))
    run_gcloud_command_with_timeout(pull_repo(args))

    if not full_setup:
        return

    logging.info("Running full setup before launch...")
    # maxdiffusion reads data from GCS via gcsfuse (no persistent data disk to attach).
    run_gcloud_command_with_timeout(ensure_gcsfuse_installed(args))
    run_gcloud_command_with_timeout(mount_cloud_bucket(args))
    logging.info("Installing conda...")
    run_gcloud_command_with_timeout(install_conda(args))
    run_gcloud_command_with_timeout(setup_environment(args))
    run_gcloud_command_with_timeout(install_dependencies(args))
    if args.skip_tests:
        logging.info("Skipping tests (--skip-tests flag set)")
    else:
        logging.info("Running tests on TPU...")
        try:
            run_gcloud_command_with_timeout(run_tests(args))
        except RuntimeError as exc:
            raise TestsFailedError(f"Remote test run failed: {exc}") from exc


def get_wandb_api():
    global _WANDB_API
    if _WANDB_API:
        return _WANDB_API
    if not WANDB_API_KEY:
        raise ValueError("Missing env variable WANDB_API_KEY")
    os.environ.setdefault("WANDB_API_KEY", WANDB_API_KEY)
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb Python package is required locally to poll run status. "
            "Install it with `pip install wandb`."
        ) from exc
    _WANDB_API = wandb.Api()
    return _WANDB_API


def get_wandb_run_state(entity: str, project: str, run_id: str) -> Optional[str]:
    api = get_wandb_api()
    path = f"{entity}/{project}/{run_id}"
    try:
        run = api.run(path)
        return run.state
    except Exception as exc:
        message = str(exc)
        if "404" in message or "permission" in message.lower():
            return None
        logging.warning(
            "Unexpected error when fetching W&B state for '%s': %s", path, message
        )
        return None


def any_tpu_worker_idle(args) -> bool:
    """Return True if any TPU worker reports that nothing is running."""
    remote_cmd = f"""
cmds=$(ps -eo command --no-headers | grep -E {shlex.quote(args.run_process_regex)});
if [[ -n "$cmds" ]]; then echo "$cmds"; else echo "Nothing is running"; fi
"""
    try:
        output = run_gcloud_command_output_with_timeout(
            format_gcloud_command(args, remote_cmd), timeout_seconds=300
        )
    except Exception as exc:
        logging.warning("Failed to query remote process state: %s", exc)
        return True

    return "Nothing is running" in output


def wait_for_active_machine(args) -> bool:
    status = machine_active(args)
    if status in ("ACTIVE", "READY"):
        return True

    while status not in ("ACTIVE", "READY"):
        logging.info("Waiting for machine to become active...")
        time.sleep(args.check_time)
        status = machine_active(args)

    return True


def machine_active(args):
    command = (
        f"gcloud alpha compute tpus tpu-vm list "
        f"--zone={args.gcp_zone} --project={args.gcp_project}"
    )
    try:
        resource_list = run_gcloud_command_output_with_timeout(
            command, timeout_seconds=120
        )
    except Exception as e:
        logging.error("Error occurred while checking machine status: %s", e)
        return "NOT CREATED"
    resource_list = resource_list.split("\n")
    names = [vm.split()[0] for vm in resource_list[1:] if len(vm) > 0]
    states = [vm.split()[7] for vm in resource_list[1:] if len(vm) > 0]
    if args.resource_name in names:
        index = names.index(args.resource_name)
        return states[index]
    return "NOT CREATED"


def unmount_all_mnt_disks(args):
    remote_cmd = """
    for d in /mnt/disks/*; do
        if [ -d "$d" ] && mountpoint -q "$d"; then
            echo "Unmounting $d..."
            sudo umount "$d" || true
        fi
    done
    """
    return format_gcloud_command(args, remote_cmd)


def list_attached_disks(args):
    describe_cmd = (
        f"gcloud alpha compute tpus tpu-vm describe {args.resource_name} "
        f"--zone={args.gcp_zone} --project={args.gcp_project} "
        f'--format="table(dataDisks[].sourceDisk.basename():label=DISK)"'
    )
    raw: str = run_gcloud_command_output_with_timeout(describe_cmd, timeout_seconds=120)
    raw = raw.strip()
    raw = raw.replace("DISK", "")
    raw = raw.strip()
    raw = raw.strip("[]")
    all_disks = [d.strip().strip("'") for d in raw.split(",") if d.strip()]
    return [d for d in all_disks if d != "persistent-disk-0"]


def detach_disk_cmd(args, disk_name):
    return (
        "gcloud alpha compute tpus tpu-vm detach-disk "
        f"{args.resource_name} --disk {disk_name} "
        f"--zone={args.gcp_zone} --project={args.gcp_project}"
    )


def kill_tpu_machine(args):
    command = (
        f"gcloud alpha compute tpus tpu-vm delete {args.resource_name} "
        f"--zone={args.gcp_zone} --project={args.gcp_project} --quiet"
    )
    return command


def terminate_process(proc: subprocess.Popen, grace_seconds: int = 30) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def run_monitor_thread(
    args,
    wandb_path: str,
    process: subprocess.Popen,
    stop_event: threading.Event,
    kill_tpu_event: threading.Event,
):
    """Monitor W&B and remote TPU state in a separate thread."""
    is_first_check = True
    try:
        while not stop_event.is_set():
            time.sleep(args.run_poll_interval)

            if process.poll() is not None:
                break

            logging.info("Checking TPU worker processes...")
            if args.wandb_run_id:
                state = get_wandb_run_state(WANDB_ENTITY, WANDB_PROJECT, args.wandb_run_id)
                logging.info("W&B run state: %s", state)
            tpu_idle = any_tpu_worker_idle(args)
            if tpu_idle:
                logging.info(
                    "At least one TPU worker reports: 'Nothing is running'. Marking as crashed."
                )
                logging.info("Detected crash condition. Terminating current run_command.")
                terminate_process(process)
                if not is_first_check:
                    kill_tpu_event.set()
                break

            logging.info("Run '%s' is still active.", wandb_path)
            is_first_check = False
    except Exception as e:
        logging.error("Error in W&B monitor thread: %s", e)


def kill_tpu_machine_and_wait(args):
    logging.info("Checking TPU machine status before killing...")
    current_status = machine_active(args)
    if current_status in ("ACTIVE", "READY"):
        logging.info(
            "TPU machine status is %s, killing TPU machine...",
            current_status,
        )
        kill_command = kill_tpu_machine(args)
        try:
            run_gcloud_command_with_timeout(kill_command)
            logging.info("TPU machine killed successfully.")
            logging.info("Waiting for TPU recreation to retry run...")
            time.sleep(60)
        except Exception as e:
            logging.error("Failed to kill TPU machine: %s", e)
    else:
        logging.info("TPU machine status is %s, not killing.", current_status)


def run_remote_command_with_monitor(args, wandb_path: str):
    command = run_command(args)
    process = subprocess.Popen(command, shell=True, start_new_session=True)

    stop_monitor = threading.Event()
    kill_tpu_event = threading.Event()
    monitor_thread = threading.Thread(
        target=run_monitor_thread,
        args=(
            args,
            wandb_path,
            process,
            stop_monitor,
            kill_tpu_event,
        ),
        daemon=True,
    )
    monitor_thread.start()

    try:
        return_code = process.wait()
        logging.info("Process finished naturally with return code %s", return_code)
    except KeyboardInterrupt:
        logging.info("Received interrupt, terminating process...")
        terminate_process(process)
        return_code = process.returncode if process.returncode is not None else -1
    finally:
        stop_monitor.set()
        if monitor_thread.is_alive():
            monitor_thread.join(timeout=1.0)

    return return_code, kill_tpu_event.is_set()


if __name__ == "__main__":
    args = parse_args()
    if args.git_key_path:
        ensure_ssh_key_loaded(args.git_key_path)
    elif not (GIT_USER and GIT_KEY):
        raise ValueError("Either --git-key-path or GIT_USER+GIT_KEY env vars must be provided")
    if args.retry:
        wandb_path = (
            f"{WANDB_ENTITY}/{WANDB_PROJECT}/{args.wandb_run_id}"
            if args.wandb_run_id
            else None
        )
        if wandb_path:
            logging.info("Monitoring W&B run at %s", wandb_path)

        full_setup_for_next_run = not args.no_setup

        try:
            while True:
                wait_for_active_machine(args)
                logging.info("Machine is active, performing setup...")

                try:
                    perform_setup(args, full_setup_for_next_run)
                except Exception as e:
                    logging.error("Error occurred while performing setup: %s", e)
                    kill_tpu_machine_and_wait(args)
                    continue
                full_setup_for_next_run = True

                return_code, kill_tpu = run_remote_command_with_monitor(
                    args, wandb_path
                )
                if return_code == 0:
                    logging.info(
                        "run_command completed successfully (return code %s).",
                        return_code,
                    )
                    break

                logging.info(
                    "run_command finished with error (return code %s), kill_tpu: %s",
                    return_code,
                    kill_tpu,
                )
                if kill_tpu:
                    kill_tpu_machine_and_wait(args)
                else:
                    logging.info("kill_tpu flag not set, continuing...")
        except KeyboardInterrupt:
            logging.info("Interrupted by user, exiting...")
    else:
        logging.info("Running without wandb monitoring and retrying")

        wait_for_active_machine(args)
        logging.info("Machine is active, performing setup...")

        perform_setup(args, not args.no_setup)

        command = run_command(args)
        run_gcloud_command_with_timeout(command, timeout_seconds=None)
