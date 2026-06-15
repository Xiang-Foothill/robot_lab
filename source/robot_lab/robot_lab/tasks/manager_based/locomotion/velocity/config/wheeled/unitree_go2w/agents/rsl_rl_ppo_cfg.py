# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class UnitreeGo2WRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 1000
    experiment_name = "unitree_go2w_rough"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_type="scalar",
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class UnitreeGo2WFlatPPORunnerCfg(UnitreeGo2WRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "unitree_go2w_flat"


@configclass
class UnitreeGo2WSmoothSteerPPORunnerCfg(UnitreeGo2WFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 4000
        self.experiment_name = "unitree_go2w_smooth_steer"
        # lower LR + tighter KL for fine-tuning from flat checkpoint
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.desired_kl = 0.005


@configclass
class UnitreeGo2WTiltPPORunnerCfg(UnitreeGo2WSmoothSteerPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        # separate experiment dir so the tilt policy never overwrites the no-tilt control group
        self.experiment_name = "unitree_go2w_tilt"


@configclass
class UnitreeGo2WMomentSteerPPORunnerCfg(UnitreeGo2WRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        # train from scratch (obs space differs from the velocity/smooth-steer tasks)
        self.max_iterations = 8000
        self.save_interval = 500
        self.experiment_name = "unitree_go2w_moment_steer"
        self.algorithm.learning_rate = 1.0e-3
        self.algorithm.desired_kl = 0.01
        # Tier 4: empirical observation normalization, baked into the exported jit policy
        self.actor.obs_normalization = True
        self.critic.obs_normalization = True
