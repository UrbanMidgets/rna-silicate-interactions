#!/usr/bin/env python3
import subprocess
import time
from pathlib import Path

# ==== CONFIG ====
QUEUE_FILE = Path("orca_jobs.queue")   # job list
MAX_ORCA_JOBS = 2                      # max ORCA jobs running in parallel
POLL_INTERVAL = 300                    # seconds between checks
ORCA_MATCH = "orca "                   # pattern to match main orca processes
# =================

def get_running_orca_jobs():
    """
    Count currently running ORCA jobs by looking for 'orca ' in the process list.
    This counts main ORCA processes, not every MPI rank.
    """
    try:
        # '-f' matches against full command line
        out = subprocess.check_output(["pgrep", "-af", ORCA_MATCH], text=True)
        lines = [line for line in out.splitlines() if "orca_queue.py" not in line]
        return len(lines)
    except subprocess.CalledProcessError:
        # pgrep returns non-zero if nothing matched
        return 0

def load_queue():
    """Return a list of pending job commands from the queue file."""
    if not QUEUE_FILE.exists():
        return []

    jobs = []
    with QUEUE_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            jobs.append(line)
    return jobs

def save_queue(jobs):
    """Rewrite the queue file with remaining jobs."""
    with QUEUE_FILE.open("w") as f:
        f.write("# ORCA job queue\n")
        for job in jobs:
            f.write(job + "\n")

def main():
    print(f"[queue] Using queue file: {QUEUE_FILE.resolve()}")
    print(f"[queue] Max ORCA jobs in parallel: {MAX_ORCA_JOBS}")
    print(f"[queue] Poll interval: {POLL_INTERVAL} s")

    while True:
        jobs = load_queue()
        if not jobs:
            print("[queue] No more jobs in queue. Exiting.")
            break

        running = get_running_orca_jobs()
        print(f"[queue] ORCA jobs running: {running}, jobs left in queue: {len(jobs)}")

        if running < MAX_ORCA_JOBS:
            # Start next job
            job_cmd = jobs.pop(0)
            print(f"[queue] Starting job: {job_cmd}")
            # Start detached; shell=True so we can use 'cd && ...'
            subprocess.Popen(job_cmd, shell=True)
            save_queue(jobs)
        else:
            print("[queue] At capacity, waiting...")

        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
