import datetime
import psutil
import os.path as osp
from rlpyt.utils.launching.variant import make_variants, VariantLevel
from rlpyt.projects.qcpo.experiments.exp_launcher import run_experiments_serial

script = osp.abspath(osp.join(osp.dirname(__file__), 'train_qcpo.py'))
print(script)

default_config_key = "LSTM"
experiment_title = f"qcpo"
cost_limit = 15
target_prob = 0.2
n_steps = 5e6
log_interval_steps = 5e4

n_cpu = psutil.cpu_count() // 2  # assume hyperthreads will be counted
affinity_code = f'{n_cpu}cpu_0gpu_1cpr_1cpw_0hto'
run_slot = 0
n_run = 1 # number of serial run
assert run_slot < n_cpu, "run_slot should be less than n_cpu"

print('rlpyt.projects.qcpo.experiments.launch_qcpo.py | affinity_code : ', affinity_code)
print('rlpyt.projects.qcpo.experiments.launch_qcpo.py | run_slot : ', run_slot)

variant_levels = list()

env_ids = [
    'SimpleButtonEnv-v0',
    # 'DynamicEnv-v0',
    # 'GremlinEnv-v0',
    # 'DynamicButtonEnv-v0',
]

values = []
dir_names = []
for env_id in env_ids:
    values.append((env_id, cost_limit, target_prob, n_steps, log_interval_steps))
    dir_names.append(osp.join(f'{env_id}', f'{cost_limit}clim_{target_prob}outprob'))
keys = [("env", "id"), ("algo", "cost_limit"), ("algo", "target_outage_prob"), ("runner", "n_steps"), ("runner", "log_interval_steps")]
variant_levels.append(VariantLevel(keys, values, dir_names))

pid_Kis = [0.1]
values = list(zip(pid_Kis))
dir_names = ["{}Ki".format(*v) for v in values]
keys = [("algo", "pid_Ki")]
variant_levels.append(VariantLevel(keys, values, dir_names))


variants, log_dirs = make_variants(*variant_levels)
print('log_dirs : ', log_dirs)

run_experiments_serial(
    script=script,
    affinity_code=affinity_code,
    run_slot=run_slot,
    experiment_title=experiment_title,
    runs_per_setting=n_run,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_config_key,),
)
