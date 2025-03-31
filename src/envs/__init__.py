from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
from .custom_env import CustomEnv
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

def env_fn(env, **kwargs) -> CustomEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["custom_env"] = partial(env_fn, env=CustomEnv)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
