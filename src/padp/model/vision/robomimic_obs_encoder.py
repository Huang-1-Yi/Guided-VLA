from typing import Dict

import torch.nn as nn
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils

from padp.common.pytorch_util import replace_submodules
from padp.common.robomimic_config_util import get_robomimic_config
from padp.model.common.module_attr_mixin import ModuleAttrMixin
import padp.model.vision.crop_randomizer as dmvc

try:
    import robomimic.models.base_nets as rmbn
    if not hasattr(rmbn, "CropRandomizer"):
        raise ImportError("CropRandomizer is not in robomimic.models.base_nets")
except ImportError:
    import robomimic.models.obs_core as rmbn


class RobomimicObsEncoder(ModuleAttrMixin):
    """Hydra-instantiable wrapper around robomimic's observation encoder."""

    def __init__(
        self,
        shape_meta: dict,
        crop_shape=(76, 76),
        obs_encoder_group_norm: bool = False,
        eval_fixed_crop: bool = False,
        algo_name: str = "bc_rnn",
        hdf5_type: str = "image",
        task_name: str = "square",
        dataset_type: str = "ph",
        device: str = "cpu",
    ):
        super().__init__()
        self.shape_meta = shape_meta

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        obs_config = {
            "low_dim": [],
            "rgb": [],
            "depth": [],
            "scan": [],
        }
        obs_key_shapes = {}
        for key, attr in shape_meta["obs"].items():
            shape = attr["shape"]
            obs_key_shapes[key] = list(shape)
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                obs_config["rgb"].append(key)
            elif obs_type == "low_dim":
                obs_config["low_dim"].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")

        config = get_robomimic_config(
            algo_name=algo_name,
            hdf5_type=hdf5_type,
            task_name=task_name,
            dataset_type=dataset_type,
        )

        with config.unlocked():
            config.observation.modalities.obs = obs_config
            if crop_shape is None:
                for _, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality["obs_randomizer_class"] = None
            else:
                ch, cw = crop_shape
                for _, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        ObsUtils.initialize_obs_utils_with_config(config)

        policy: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device=device,
        )
        encoder = policy.nets["policy"].nets["encoder"].nets["obs"]

        if obs_encoder_group_norm:
            encoder = replace_submodules(
                root_module=encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16,
                    num_channels=x.num_features,
                ),
            )

        if eval_fixed_crop:
            encoder = replace_submodules(
                root_module=encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc,
                ),
            )

        self.encoder = encoder

    def forward(self, obs_dict: Dict):
        return self.encoder(obs_dict)

    def output_shape(self):
        return self.encoder.output_shape()

    def no_weight_decay(self):
        if hasattr(self.encoder, "no_weight_decay"):
            return self.encoder.no_weight_decay()
        return set()
