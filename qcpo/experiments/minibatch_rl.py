
import os
import time
import torch
from collections import deque
import csv
import numpy as np

from rlpyt.utils.logging import logger
from rlpyt.utils.prog_bar import ProgBarCounter
from rlpyt.runners.minibatch_rl import MinibatchRlBase

class MinibatchRlConstrained(MinibatchRlBase):
    """
    Runs RL on minibatches; tracks performance online using learning
    trajectories.
    """

    def __init__(self, cost_limit=25.0, log_traj_window=100, **kwargs):
        """
        Args: 
            log_traj_window (int): How many trajectories to hold in deque for computing performance statistics.
        """
        super().__init__(**kwargs)
        self.cost_limit = cost_limit
        self.log_traj_window = int(log_traj_window)

    def train(self):
        """
        Performs startup, then loops by alternating between
        ``sampler.obtain_samples()`` and ``algo.optimize_agent()``, logging
        diagnostics at the specified interval.
        """
        n_itr = self.startup()
        for itr in range(n_itr):
            logger.set_iteration(itr)
            with logger.prefix(f"itr #{itr} "):
                self.agent.sample_mode(itr)  # Might not be this agent sampling.
                samples, traj_infos = self.sampler.obtain_samples(itr) # type of samples is tensor

                self.agent.train_mode(itr)
                opt_info = self.algo.optimize_agent(itr, samples)

                self.store_diagnostics(itr, traj_infos, opt_info)
                self.log_traj_infos(traj_infos)
                if (itr + 1) % self.log_interval_itrs == 0:
                    self.log_diagnostics(itr)
        self.shutdown()

    def initialize_logging(self):
        self._traj_infos = deque(maxlen=self.log_traj_window)
        self._new_completed_trajs = 0
        self._new_outage_trajs = 0
        self._new_zero_cost_trajs = 0
        self._new_half_cost_trajs = 0
        self._current_traj_ind = 0

        logger.log(f"Optimizing over {self.log_interval_itrs} iterations.")
        super().initialize_logging()
        self.pbar = ProgBarCounter(self.log_interval_itrs)

    def store_diagnostics(self, itr, traj_infos, opt_info):
        self._new_completed_trajs += len(traj_infos)
        traj_cost = np.array([info["Cost"] for info in traj_infos])
        self._new_outage_trajs += np.sum(traj_cost > self.cost_limit)
        self._new_zero_cost_trajs += np.sum(traj_cost == 0.)
        self._new_half_cost_trajs += np.sum(np.logical_and(traj_cost < self.cost_limit / 2, traj_cost > 0))
        self._traj_infos.extend(traj_infos)
        super().store_diagnostics(itr, traj_infos, opt_info)

    def log_diagnostics(self, itr, prefix='Diagnostics/'):
        l = len(self._traj_infos)
        traj_cost = np.array([info["Cost"] for info in self._traj_infos])
        traj_return = np.array([info["Return"] for info in self._traj_infos])
        logger.record_tabular('ReturnOutage', np.mean(traj_return[traj_cost > self.cost_limit]))
        logger.record_tabular('ReturnZeroCost', np.mean(traj_return[traj_cost == 0]))
        logger.record_tabular('ReturnHalfCost', np.mean(traj_return[np.logical_and(traj_cost < self.cost_limit / 2, traj_cost > 0)]))
        logger.record_tabular('CostOutage', np.mean(traj_cost[traj_cost > self.cost_limit]))
        logger.record_tabular('CostZeroCost', np.mean(traj_cost[traj_cost == 0]))
        logger.record_tabular('CostHalfCost', np.mean(traj_cost[np.logical_and(traj_cost < self.cost_limit / 2, traj_cost > 0)]))
        logger.record_tabular('ProbOutage', np.sum(traj_cost > self.cost_limit) / l)
        logger.record_tabular('ProbZeroCost', np.sum(traj_cost == 0) / l)
        logger.record_tabular('ProbHalfCost', np.sum(np.logical_and(traj_cost < self.cost_limit / 2, traj_cost > 0)) / l)

        with logger.tabular_prefix(prefix):
            logger.record_tabular('StepsInTrajWindow', sum(info["Length"] for info in self._traj_infos))
            logger.record_tabular('NewCompletedTrajs', self._new_completed_trajs)
            logger.record_tabular('NewOutageTrajs', self._new_outage_trajs)
            logger.record_tabular('NewZeroCostTrajs', self._new_zero_cost_trajs)
            logger.record_tabular('NewHalfCostTrajs', self._new_half_cost_trajs)
            logger.record_tabular('NewProbOutage', self._new_outage_trajs / self._new_completed_trajs)
            logger.record_tabular('NewProbZeroCost', self._new_zero_cost_trajs / self._new_completed_trajs)
            logger.record_tabular('NewProbHalfCost', self._new_half_cost_trajs / self._new_completed_trajs)
        super().log_diagnostics(itr, prefix=prefix)

        self._new_completed_trajs = 0
        self._new_outage_trajs = 0
        self._new_zero_cost_trajs = 0
        self._new_half_cost_trajs = 0

    def log_traj_infos(self, traj_infos, discount=0.99):

        def maybe_single_value(v, ind):
            try:
                v1 = v[ind]
            except:
                v1 = np.NaN
            if isinstance(v1, list):
                v1 = np.array(v1)
            if isinstance(v1, np.ndarray):
                return np.array2string(v1)
            else:
                return v1

        traj_summary_filename = os.path.join(
            logger.get_snapshot_dir(),
            'traj_summary.csv',
        )
        is_first_row = not os.path.exists(traj_summary_filename)
        summary_fd = open(traj_summary_filename, 'a', newline='')
        summary_writer = csv.writer(summary_fd)

        for i, traj_info in enumerate(traj_infos):
            joint_keys = []
            max_len = 0
            for k, v in traj_info.items():
                if isinstance(v, list) and not isinstance(v[0], np.ndarray):
                    joint_keys.append(k)
                    max_len = max(max_len, len(v))

            if len(joint_keys) == 0:
                continue

            # Compute Summary
            summary_dict = dict()
            summary_dict['TrajSumReward'] = np.sum(traj_info['TrajRewards'])
            summary_dict['TrajSumCost'] = np.sum(traj_info['TrajCosts'])
            summary_dict['TrajSumRValue'] = np.sum(traj_info['TrajRValues'])
            summary_dict['TrajSumCValue'] = np.sum(traj_info['TrajCValues'])
            summary_dict['IsOutageTraj'] = float(summary_dict['TrajSumCost'] > 25.0)

            if 'TrajCWAlphas' in traj_info.keys():
                summary_dict['TrajMeanCWAlphas'] = np.mean(traj_info['TrajCWAlphas'])
            if 'TrajCWBetas' in traj_info.keys():
                summary_dict['TrajMeanCWBetas'] = np.mean(traj_info['TrajCWBetas'])

            # Write Summary
            if is_first_row:
                summary_writer.writerow([*summary_dict.keys()])
                is_first_row = False
            summary_writer.writerow([v for v in summary_dict.values()])

        self._current_traj_ind += len(traj_infos)
        summary_fd.close()

    ####### Additional func
    def show_samples(self, samples, prefix=''):
        if not isinstance(samples, torch.Tensor):
            for k, v in samples.items():
                print(f'{prefix}{k} : {type(v)}')
                self.show_samples(v, prefix=prefix+'\t')
        else:
            print(f'{prefix}Tensor.shape : {tuple(samples.shape)}')
            return

    def reshape_samples(self, samples, shape, prefix=''):
        if not isinstance(samples, torch.Tensor):
            for k, v in samples.items():
                print(f'{prefix}{k} : {type(v)}, {isinstance(v, torch.Tensor)}')
                if isinstance(v, torch.Tensor):
                    # print('hello')
                    if len(v.shape) > 2:
                        print(v.shape[2:])
                        new_shape = shape + tuple(v.shape[2:])
                    else:
                        new_shape = shape
                    print('ok')
                    samples[k] = v.view(*new_shape)
                self.reshape_samples(v, shape, prefix=prefix+'\t')
        else:
            print(f'{prefix}Tensor.shape : {samples.shape}')
            return
