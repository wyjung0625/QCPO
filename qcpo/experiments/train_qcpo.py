
import sys
import pprint

from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.projects.qcpo.experiments.minibatch_rl import MinibatchRlConstrained
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.launching.variant import load_variant, update_config
from rlpyt.utils.launching.affinity import affinity_from_code

from rlpyt.projects.qcpo.qcpo import QCPO
from rlpyt.projects.qcpo.qcpo_model import QcpoModel
from rlpyt.projects.qcpo.qcpo_agent import QcpoLstmAgent

from rlpyt.projects.qcpo.safety_gym_envs.safety_gym_env import safety_gym_make, SafetyGymTrajInfo

from rlpyt.projects.qcpo.experiments.config_qcpo import configs
from rlpyt.projects.qcpo.safety_gym_envs.config_safety_gym_env import (
    config0,
    config1,
    config2,
    config3,
)
from rlpyt.projects.qcpo.safety_gym_envs.safety_gym_env import register_more_envs


def build(slot_affinity_code="0slt_0gpu_1cpu_1cpr",
          log_dir="test",
          run_ID="0",
          config_key="LSTM",
          variant_dir=None,
          ):
    affinity = affinity_from_code(slot_affinity_code)
    config = configs[config_key]
    variant = load_variant(log_dir)
    config = update_config(config, variant)

    total_batch_size = config["sampler"]["batch_T"] * config["sampler"]["batch_B"]
    print("train_qcpo.py/build_and_train : total_batch_size : ", total_batch_size)
    print("train_qcpo.py/build_and_train : master_torch_threads : ", affinity["master_torch_threads"])

    new_batch_config = {"sampler": {
        "batch_T": int(total_batch_size // affinity["master_torch_threads"]),
        "batch_B": int(affinity["master_torch_threads"]),
    }}
    config = update_config(config, new_batch_config)

    print("rlpyt.projects.qcpo.experiments.train_qcpo.py | config")
    pprint.pprint(config)

    print("rlpyt.projects.qcpo.experiments.train_qcpo.py | affinity")
    pprint.pprint(affinity)

    register_more_envs([
        {'id': 'SimpleButtonEnv-v0', 'config': config0},
        {'id': 'DynamicEnv-v0', 'config': config1},
        {'id': 'GremlinEnv-v0', 'config': config2},
        {'id': 'DynamicButtonEnv-v0', 'config': config3},
    ])

    sampler = CpuSampler(
        EnvCls=safety_gym_make,
        env_kwargs=config["env"],
        TrajInfoCls=SafetyGymTrajInfo,
        **config["sampler"]
    )

    algo = QCPO(**config["algo"])
    agent = QcpoLstmAgent(ModelCls=QcpoModel, model_kwargs=config["model"], **config["agent"])
    runner = MinibatchRlConstrained(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        cost_limit=config["algo"]["cost_limit"],
        **config["runner"],
    )
    name = "qcpo_" + config["env"]["id"]
    return runner, log_dir, run_ID, name, config, agent


def build_and_train(
        slot_affinity_code="0slt_0gpu_1cpu_1cpr",
        log_dir="test",
        run_ID="0",
        config_key="LSTM",
        ):
    runner, log_dir, run_ID, name, config, _ = build(slot_affinity_code, log_dir, run_ID, config_key)
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    runner, log_dir, run_ID, name, config, _ = build(*sys.argv[1:])
    with logger_context(log_dir, run_ID, name, config):
        runner.train()




