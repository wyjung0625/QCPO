
######################################################################
# Algorithm file.
######################################################################

import numpy as np
import torch
# import torch.nn.functional as F
from collections import namedtuple, deque

from rlpyt.algos.pg.base import PolicyGradientAlgo, OptInfo
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.misc import iterate_mb_idxs
from rlpyt.algos.utils import (discount_return,
    generalized_advantage_estimation, valid_from_done)
from rlpyt.projects.qcpo.dist_rl_utils import (normalize, quantile_huber_loss, weibull_tail_loss, quantile_target_estimation,
                                                gae_quantile_simple, compute_prob_ratio)
from rlpyt.distributions.gaussian import DistInfoStd
from rlpyt.projects.qcpo.qcpo_model import RnnState
from rlpyt.utils.logging import logger


LossInputs = namedarraytuple("LossInputs",
    ["agent_inputs", "action", "r_return", "r_advantage", "valid", "old_dist_info",
    "c_return", "c_advantage", "c_dist_target"])
LossInfo = namedtuple("LossInfo", ("piRLoss", "piCLoss", "RvalueLoss", "CvalueLoss", "CquantileLoss", "CweibullLoss"))
OptInfoCost = namedtuple("OptInfoCost", OptInfo._fields + LossInfo._fields + ("costPenalty",
    "costLimit", "valueError", "cvalueError", "valueAbsError", "cvalueAbsError",
    "pid_i"))


class QCPO(PolicyGradientAlgo):
    """
    Quantile Constrained Policy Optimization

    """
    opt_info_fields = OptInfoCost._fields

    def __init__(
            self,
            discount=0.99,
            learning_rate=0.001,
            value_loss_coeff=1.,
            entropy_loss_coeff=0.,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            clip_grad_norm=1.,
            initial_optim_state_dict=None,
            gae_lambda=0.97,
            minibatches=1,
            epochs=8,
            ratio_clip=0.1,
            linear_lr_schedule=False,
            normalize_advantage=False,
            cost_discount=None,  # if None, defaults to discount.
            cost_gae_lambda=None,
            cost_value_loss_coeff=None,
            ep_cost_ema_alpha=0,  # 0 for hard update, 1 for no update.
            ep_outage_ema_alpha=0,  # 0 for hard update, 1 for no update.
            ep_cost_eqa_alpha=0,  # 0 for hard update, 1 for no update.
            objective_penalized=True,  # False for reward-only learning
            learn_c_value=True,  # Also False for reward-only learning
            penalty_init=1.,
            cost_limit=25,
            cost_scale=10.,  # divides; applied to raw cost and cost_limit
            target_outage_prob=0.3,
            weibull_tail_prob=0.3,
            n_quantile=25,
            normalize_cost_advantage=False,
            pid_Ki=0.1,
            sum_norm=True,  # L = (J_r - lam * J_c) / (1 + lam); lam <= 0
            diff_norm=False,  # L = (1 - lam) * J_r - lam * J_c; 0 <= lam <= 1
            penalty_max=100,  # only used if sum_norm=diff_norm=False
            step_cost_limit_steps=None,  # Change the cost limit partway through
            step_cost_limit_value=None,  # New value.
            reward_scale=1,  # multiplicative (unlike cost_scale)
            lagrange_quadratic_penalty=False,
            quadratic_penalty_coeff=1,
            new_T=128,
            new_B=104,
            ):
        assert learn_c_value or not objective_penalized
        assert (step_cost_limit_steps is None) == (step_cost_limit_value is None)
        assert not (sum_norm and diff_norm)

        cost_discount = discount if cost_discount is None else cost_discount
        cost_gae_lambda = (gae_lambda if cost_gae_lambda is None else
            cost_gae_lambda)
        cost_value_loss_coeff = (value_loss_coeff if cost_value_loss_coeff is
            None else cost_value_loss_coeff)
        if optim_kwargs is None:
            optim_kwargs = dict()
        save__init__args(locals())
        self.cost_limit /= self.cost_scale
        if step_cost_limit_value is not None:
            self.step_cost_limit_value /= self.cost_scale

        self.target_outage_prob = target_outage_prob
        self.target_outage_ind = np.floor(n_quantile * (1 - target_outage_prob) - 0.5).astype(int)

        self.weibull_tail_prob = weibull_tail_prob
        self.weibull_tail_ind = np.floor(n_quantile * (1 - weibull_tail_prob) - 0.5).astype(int)

        logger.log('self.target_outage_prob', self.target_outage_prob)
        logger.log('self.target_outage_ind', self.target_outage_ind)
        logger.log('self.weibull_tail_prob', self.weibull_tail_prob)
        logger.log('self.weibull_tail_ind', self.weibull_tail_ind)
        self._n_quantile = n_quantile
        self.current_ep_costs = deque(maxlen=100)

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self._batch_size = self.batch_spec.size // self.minibatches  # For logging.
        if self.linear_lr_schedule:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=lambda itr: (self.n_itr - itr) / self.n_itr)  # Step once per itr.
            self._ratio_clip = self.ratio_clip  # Save base value.
        if self.step_cost_limit_steps is None:
            self.step_cost_limit_itr = None
        else:
            self.step_cost_limit_itr = int(self.step_cost_limit_steps //
                (self.batch_spec.size * self.world_size))
            # print("\n\n step cost itr: ", self.step_cost_limit_itr, "\n\n")
        self._ep_cost_ema = self.cost_limit  # No derivative at start.
        self._ep_outage_ema = self.target_outage_prob
        self._ep_cost_eqa = self.cost_limit
        self._ddp = self.agent._ddp
        assert self._ddp == False, print('ddp should be False')
        self.pid_i = self.cost_penalty = self.penalty_init

    def transform_sample(self, sample):
        assert isinstance(sample, torch.Tensor), print(f'sample is not a tensor')
        old_shape = sample.shape
        assert old_shape[0] * old_shape[1] == self.new_T * self.new_B, print(f'old_shape[0] * old_shape[1] ({old_shape[0]} * {old_shape[1]}) does not match to new_T * new_B ({self.new_T} * {self.new_B})')
        if old_shape[0] == self.new_T and old_shape[1] == self.new_B:
            return sample

        new_shape = (self.new_B, self.new_T) + tuple(old_shape[2:]) if len(old_shape) > 2 else (self.new_B, self.new_T)
        return sample.transpose(0, 1).reshape(*new_shape).transpose(0, 1).contiguous()

    def optimize_agent(self, itr, samples):
        recurrent = self.agent.recurrent
        agent_inputs = AgentInputs(  # Move inputs to device once, index there.
            observation=self.transform_sample(samples.env.observation),
            prev_action=self.transform_sample(samples.agent.prev_action),
            prev_reward=self.transform_sample(samples.env.prev_reward),
        )
        action = self.transform_sample(samples.agent.action)
        dist_info = samples.agent.agent_info.dist_info
        old_dist_info = DistInfoStd(
            mean=self.transform_sample(dist_info.mean),
            log_std=self.transform_sample(dist_info.log_std)
        )

        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)

        (r_return, r_advantage,
         c_return, c_adv_q_addcost, c_dist_target,
         valid, ep_cost_avg, ep_outage_avg, ep_cost_quantile) = self.process_returns(itr, samples)

        loss_inputs = LossInputs(  # So can slice all.
            agent_inputs=agent_inputs,
            action=action,
            r_return=r_return,
            r_advantage=r_advantage,
            valid=valid,
            old_dist_info=old_dist_info,
            c_return=c_return,  # Can be None.
            c_advantage=c_adv_q_addcost[:, :, self.target_outage_ind - self.weibull_tail_ind],
            c_dist_target=c_dist_target,
        )
        opt_info = OptInfoCost(*([] for _ in range(len(OptInfoCost._fields))))

        if (self.step_cost_limit_itr is not None and
                self.step_cost_limit_itr == itr):
            self.cost_limit = self.step_cost_limit_value
        opt_info.costLimit.append(self.cost_limit)

        delta = float(ep_cost_quantile - self.cost_limit)
        self.pid_i = max(0., self.pid_i + delta * self.pid_Ki)
        if self.diff_norm:
            self.pid_i = max(0., min(1., self.pid_i))
        self.cost_penalty = max(0., self.pid_i)

        if self.diff_norm:
            self.cost_penalty = min(1., self.cost_penalty)
        if not (self.diff_norm or self.sum_norm):
            self.cost_penalty = min(self.cost_penalty, self.penalty_max)

        opt_info.pid_i.append(self.pid_i)
        opt_info.costPenalty.append(self.cost_penalty)

        if hasattr(self.agent, "update_obs_rms"):
            self.agent.update_obs_rms(agent_inputs.observation)
            if itr == 0:
                return opt_info  # Sacrifice the first batch to get obs stats.

        if recurrent:
            # Leave in [B,N,H] for slicing to minibatches.
            prev_rnn_state = RnnState(h=self.transform_sample(samples.agent.agent_info.prev_rnn_state.h),
                                      c=self.transform_sample(samples.agent.agent_info.prev_rnn_state.c))
            init_rnn_state = prev_rnn_state[0]  # T=0.

        # If recurrent, use whole trajectories, only shuffle B; else shuffle all.
        batch_size = self.new_B if self.agent.recurrent else self.new_T * self.new_B
        mb_size = batch_size // self.minibatches

        for epo in range(self.epochs):
            for idxs in iterate_mb_idxs(batch_size, mb_size, shuffle=True):
                T_idxs = slice(None) if recurrent else idxs % self.new_T
                B_idxs = idxs if recurrent else idxs // self.new_T
                self.optimizer.zero_grad()
                rnn_state = init_rnn_state[B_idxs] if recurrent else None
                # NOTE: if not recurrent, will lose leading T dim, should be OK.
                loss, entropy, perplexity, value_errors, abs_value_errors, pi_losses, value_losses, quantile_loss, weibull_loss \
                    = self.loss(*loss_inputs[T_idxs, B_idxs], rnn_state)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.clip_grad_norm)
                self.optimizer.step()

                opt_info.loss.append(loss.item())
                opt_info.gradNorm.append(grad_norm)
                opt_info.entropy.append(entropy.item())
                opt_info.perplexity.append(perplexity.item())
                opt_info.valueError.extend(value_errors[0][::10].numpy())
                opt_info.cvalueError.extend(value_errors[1][::10].numpy())
                opt_info.valueAbsError.extend(abs_value_errors[0][::10].numpy())
                opt_info.cvalueAbsError.extend(abs_value_errors[1][::10].numpy())
                opt_info.piRLoss.append(pi_losses[0].numpy())
                opt_info.piCLoss.append(pi_losses[1].numpy())
                opt_info.RvalueLoss.append(value_losses[0].numpy())
                opt_info.CvalueLoss.append(value_losses[1].numpy())
                opt_info.CquantileLoss.append(quantile_loss.numpy())
                opt_info.CweibullLoss.append(weibull_loss.numpy())

                self.update_counter += 1
        if self.linear_lr_schedule:
            self.lr_scheduler.step()
            self.ratio_clip = self._ratio_clip * (self.n_itr - itr) / self.n_itr

        return opt_info

    def loss(self, agent_inputs, action, r_return, r_advantage, valid, old_dist_info,
            c_return, c_advantage, c_dist_target, init_rnn_state=None):
        if init_rnn_state is not None:
            # [B,N,H] --> [N,B,H] (for cudnn).
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            dist_info, value, _rnn_state = self.agent(*agent_inputs, init_rnn_state)
        else:
            dist_info, value = self.agent(*agent_inputs)
        dist = self.agent.distribution

        ratio = dist.likelihood_ratio(action, old_dist_info=old_dist_info,
            new_dist_info=dist_info)
        surr_1 = ratio * r_advantage
        clipped_ratio = torch.clamp(ratio, 1. - self.ratio_clip,
            1. + self.ratio_clip)
        surr_2 = clipped_ratio * r_advantage
        surrogate = torch.min(surr_1, surr_2)
        pi_loss = - valid_mean(surrogate, valid)
        pi_r_loss = pi_loss

        if self.reward_scale == 1.:
            value_error = value.r_value - r_return
        else:
            value_error = value.r_value - (r_return / self.reward_scale)  # Undo the scaling
        value_se = 0.5 * value_error ** 2
        r_value_loss = self.value_loss_coeff * valid_mean(value_se, valid)
        # Hmm, but with reward scaling, now the value gradient will be relatively smaller
        # than the pi gradient, unless we also change the value_loss_coeff??  Eh, leave it.

        entropy = dist.mean_entropy(dist_info, valid)
        entropy_loss = - self.entropy_loss_coeff * entropy

        if self.objective_penalized:
            # This, or just add c_advantage into advantage?
            c_surr_1 = ratio * c_advantage
            c_surr_2 = clipped_ratio * c_advantage
            c_surrogate = torch.max(c_surr_1, c_surr_2)
            pi_c_loss = valid_mean(c_surrogate, valid)

            if self.diff_norm:  # (1 - lam) * R + lam * C
                pi_loss *= (1 - self.cost_penalty)
                pi_loss += self.cost_penalty * pi_c_loss
            elif self.sum_norm:  # 1 / (1 + lam) * (R + lam * C)
                pi_loss += self.cost_penalty * pi_c_loss
                pi_loss /= (1 + self.cost_penalty)
            else:
                pi_loss += self.cost_penalty * pi_c_loss

            if self.lagrange_quadratic_penalty:
                quad_loss = (self.quadratic_penalty_coeff
                    * valid_mean(c_surrogate, valid)
                    * torch.max(torch.tensor(0.), self._ep_cost_ema - self.cost_limit))
                pi_loss += quad_loss
        else:
            pi_c_loss = None

        loss = pi_loss + entropy_loss + r_value_loss

        if self.learn_c_value:  # Then separate cost value estimate.
            assert value.c_dist is not None
            assert c_return is not None
            c_value_error = torch.mean(value.c_dist, dim=-1) - c_return
            c_value_se = 0.5 * c_value_error ** 2
            c_value_loss = valid_mean(c_value_se, valid)
            c_quantile_loss = quantile_huber_loss(value.c_dist, c_dist_target, self.agent.model.tau, valid)
            c_weibull_loss = weibull_tail_loss(value.c_dist, value.c_w_alpha, value.c_w_beta, self.agent.model.c_tau, self.weibull_tail_ind, valid)

            loss += self.cost_value_loss_coeff * c_value_loss + c_quantile_loss + c_weibull_loss
        else:
            c_quantile_loss = None
            c_value_loss = None
            c_weibull_loss = None

        value_errors = (value_error.detach(), c_value_error.detach())
        if valid is not None:
            valid_mask = valid > 0
            value_errors = tuple(v[valid_mask] for v in value_errors)
        else:
            value_errors = tuple(v.view(-1) for v in value_errors)
        abs_value_errors = tuple(abs(v) for v in value_errors)
        perplexity = dist.mean_perplexity(dist_info, valid)

        pi_losses = (pi_r_loss.detach(), pi_c_loss.detach())
        value_losses = (r_value_loss.detach(), c_value_loss.detach())
        quantile_loss = c_quantile_loss.detach()
        weibull_loss = c_weibull_loss.detach()

        return loss, entropy, perplexity, value_errors, abs_value_errors, pi_losses, value_losses, quantile_loss, weibull_loss

    def process_returns(self, itr, samples):
        reward, cost = self.transform_sample(samples.env.reward), self.transform_sample(samples.env.env_info.cost)
        cost /= self.cost_scale
        done = self.transform_sample(samples.env.done)
        r_value, c_dist, c_w_alpha, c_w_beta = samples.agent.agent_info.value  # A named 2-tuple.
        r_bv, c_bdist, c_bw_alpha, c_bw_beta = samples.agent.bootstrap_value  # A named 2-tuple.
        r_value = self.transform_sample(r_value)
        c_dist = self.transform_sample(c_dist)
        c_w_alpha = self.transform_sample(c_w_alpha)
        c_w_beta = self.transform_sample(c_w_beta)

        if not r_bv.shape[1] == self.new_B:
            r_bv_t, c_bdist_t, c_bw_alpha_t, c_bw_beta_t = [], [], [], []
            for i in range(r_bv.shape[1]):
                m = int(self.new_B // r_bv.shape[1])
                r_bv_t.extend([r_value[0, i * m + 1:(i + 1) * m].clone().unsqueeze(0), r_bv[:, i].clone().unsqueeze(0)])
                c_bdist_t.extend([c_dist[0, i * m + 1:(i + 1) * m].clone().unsqueeze(0), c_bdist[:, i].clone().unsqueeze(0)])
                c_bw_alpha_t.extend([c_w_alpha[0, i * m + 1:(i + 1) * m].clone().unsqueeze(0), c_bw_alpha[:, i].clone().unsqueeze(0)])
                c_bw_beta_t.extend([c_w_beta[0, i * m + 1:(i + 1) * m].clone().unsqueeze(0), c_bw_beta[:, i].clone().unsqueeze(0)])

            r_bv = torch.cat(r_bv_t, 1)
            c_bdist = torch.cat(c_bdist_t, 1)
            c_bw_alpha = torch.cat(c_bw_alpha_t, 1)
            c_bw_beta = torch.cat(c_bw_beta_t, 1)

        if self.reward_scale != 1:
            reward *= self.reward_scale
            r_value *= self.reward_scale  # Keep the value learning the same.
            r_bv *= self.reward_scale

        c_value, c_bv = torch.mean(c_dist, dim=-1), torch.mean(c_bdist, dim=-1)
        done = done.type(reward.dtype)  # rlpyt does this in discount_returns?

        # ====================================================
        # Compute advantages and returns
        # ====================================================
        if c_value is not None:  # Learning c_value, even if reward penalized.
            if self.cost_gae_lambda == 1:  # GAE reduces to empirical discount.
                c_return = discount_return(cost, done, c_bv, self.cost_discount)
                c_adv_q = c_return - c_dist
                c_adv_q_addcost = None
            else:
                _, c_return = generalized_advantage_estimation(cost, c_value, done, c_bv, self.cost_discount, self.cost_gae_lambda)

                tail_ind = self.weibull_tail_ind
                with torch.no_grad():
                    c_tail = torch.sort(c_dist)[0][:, :, self.weibull_tail_ind:]
                prob_ratio_rev = compute_prob_ratio(cost, c_tail, c_w_alpha, c_w_beta, done, c_bw_alpha, c_bw_beta, self.discount, log_clip_range=0.5)
                c_adv_q, _ = gae_quantile_simple(cost, c_dist, done, c_bdist, self.discount, self.cost_gae_lambda)
                c_adv_q_addcost = (1. + torch.log(prob_ratio_rev)) * c_adv_q[:, :, tail_ind:]

        else:
            c_return = c_adv_q = c_adv_q_addcost = None

        if self.gae_lambda == 1:  # GAE reduces to empirical discounted.
            r_return = discount_return(reward, done, r_bv, self.discount)
            r_advantage = r_return - r_value
        else:
            r_advantage, r_return = generalized_advantage_estimation(reward, r_value, done, r_bv, self.discount, self.gae_lambda)

        # ====================================================
        # Compute target distributions
        # ====================================================
        c_dist_target = quantile_target_estimation(cost, c_dist, done, c_bdist, self.cost_discount)

        # ====================================================
        # Compute valid
        # ====================================================
        if not self.mid_batch_reset or self.agent.recurrent:
            valid = valid_from_done(done)  # Recurrent: no reset during training.
            ep_cost_mask = torch.logical_and(valid, done)
        else:
            valid = None  # OR: torch.ones_like(done)
            ep_cost_mask = done  # Everywhere a done, is episode final cost.

        # ====================================================
        # Further info processing
        # ====================================================
        cum_cost = self.transform_sample(samples.env.env_info.cum_cost)
        ep_costs = cum_cost[ep_cost_mask.type(torch.bool)]
        ep_outages = torch.gt(ep_costs / self.cost_scale, self.cost_limit).float()
        self.current_ep_costs.extend(ep_costs)

        # if self._ddp:
        #     world_size = torch.distributed.get_world_size()  # already have self.world_size
        if ep_costs.numel() > 0:  # Might not have any completed trajectories.
            ep_cost_avg = ep_costs.mean()
            ep_cost_avg /= self.cost_scale
            self._ep_cost_ema *= self.ep_cost_ema_alpha
            self._ep_cost_ema += (1 - self.ep_cost_ema_alpha) * ep_cost_avg

            quantile_ind = np.floor(len(self.current_ep_costs) * (1 - self.target_outage_prob)).astype(int)
            ep_cost_quantile = np.sort(self.current_ep_costs)[quantile_ind]
            ep_cost_quantile /= self.cost_scale
            self._ep_cost_eqa *= self.ep_cost_eqa_alpha
            self._ep_cost_eqa += (1 - self.ep_cost_eqa_alpha) * ep_cost_quantile

        if ep_outages.numel() > 0:  # Might not have any completed trajectories.
            ep_outage_avg = ep_outages.mean()
            self._ep_outage_ema *= self.ep_outage_ema_alpha
            self._ep_outage_ema += (1 - self.ep_outage_ema_alpha) * ep_outage_avg

        # ====================================================
        # Normalize advantages
        # ====================================================
        if self.normalize_advantage:
            r_advantage[:] = normalize(value=r_advantage, valid=valid)

        if self.normalize_cost_advantage:
            c_adv_q_addcost[:] = normalize(value=c_adv_q_addcost, valid=valid)

        return (r_return, r_advantage,
                c_return, c_adv_q_addcost, c_dist_target,
                valid, self._ep_cost_ema, self._ep_outage_ema, self._ep_cost_eqa)





