# This python file is based on rlpyt/rlpyt/utils/launching/exp_launcher.py at https://github.com/astooke/rlpyt

import os
import os.path as osp
import datetime

from rlpyt.utils.launching.affinity import get_n_run_slots
from rlpyt.utils.logging.context import get_log_dir
from rlpyt.utils.launching.exp_launcher import log_exps_tree, log_num_launched, launch_experiment


def run_experiments_serial(script, affinity_code, run_slot, experiment_title, runs_per_setting,
        variants, log_dirs, common_args=None, runs_args=None,
        set_egl_device=False):
    """Call in a script to run a set of experiments locally on a machine.  Uses
    the ``launch_experiment()`` function for each individual run, which is a
    call to the ``script`` file.  The number of experiments to run at the same
    time is determined from the ``affinity_code``, which expresses the hardware
    resources of the machine and how much resource each run gets (e.g. 4 GPU
    machine, 2 GPUs per run).  Experiments are queued and run in sequence, with
    the intention to avoid hardware overlap.  Inputs ``variants`` and ``log_dirs``
    should be lists of the same length, containing each experiment configuration
    and where to save its log files (which have the same name, so can't exist
    in the same folder).

    Hint:
        To monitor progress, view the `num_launched.txt` file and `experiments_tree.txt`
        file in the experiment root directory, and also check the length of each
        `progress.csv` file, e.g. ``wc -l experiment-directory/.../run_*/progress.csv``.
    """
    n_run_slots = get_n_run_slots(affinity_code)
    print('rlpyt.projects.qcpo.experiments.exp_launcher.py | n_run_slots: ', n_run_slots)
    exp_dir = get_log_dir(experiment_title)
    print('rlpyt.projects.qcpo.experiments.exp_launcher.py | exp_dir: ', exp_dir)

    common_args = () if common_args is None else common_args
    assert len(variants) == len(log_dirs)
    if runs_args is None:
        runs_args = [()] * len(variants)
    assert len(runs_args) == len(variants)
    log_exps_tree(exp_dir, log_dirs, runs_per_setting)
    num_launched, total = 0, runs_per_setting * len(variants)
    for run_ID in range(runs_per_setting):
        for variant, log_dir, run_args in zip(variants, log_dirs, runs_args):
            print('rlpyt.projects.qcpo.experiments.exp_launcher.py | exp_dir: ', exp_dir)
            print('rlpyt.projects.qcpo.experiments.exp_launcher.py | log_dir: ', log_dir)
            yyyymmdd_hhmmss = datetime.datetime.today().strftime("%Y%m%d_%H%M%S")
            log_dir = osp.join(exp_dir, log_dir)
            os.makedirs(log_dir, exist_ok=True)
            p = launch_experiment(
                script=script,
                run_slot=run_slot,
                affinity_code=affinity_code,
                log_dir=log_dir,
                variant=variant,
                run_ID=f"{run_ID}_{yyyymmdd_hhmmss}",
                args=common_args + run_args,
                set_egl_device=set_egl_device,
            )
            p.wait()
            num_launched += 1
            log_num_launched(exp_dir, num_launched, total)
