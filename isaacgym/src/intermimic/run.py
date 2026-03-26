# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import glob
import subprocess
import shutil
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from .utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from .utils.parse_task import parse_task
from .utils.path_utils import resolve_repo_path

from rl_games.algos_torch import torch_ext
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner

import numpy as np
import copy
import torch

from .learning import intermimic_agent
from .learning import intermimic_players
from .learning import intermimic_models
from .learning import intermimic_network_builder

args = None
cfg = None
cfg_train = None


def _normalize_exp_override(exp_dir: str) -> str:
    """
    Match intermimic task render logic:
    - accept "exp/foo", "foo", "/abs/.../exp/foo"
    - keep only "foo"
    """
    exp_dir = exp_dir.strip().strip("/")
    if "/exp/" in exp_dir:
        exp_dir = exp_dir.split("/exp/", 1)[1]
    if exp_dir.startswith("exp/"):
        exp_dir = exp_dir[len("exp/") :]
    return exp_dir


def _encode_visualization_video_from_frames(exp_dir: str, fps: float, env_id: int = 0) -> str:
    """Encode rgb_env{env_id}_frame*.png into exp/{exp_dir}/visualization.mp4."""
    exp_override = _normalize_exp_override(exp_dir)
    exp_root = resolve_repo_path("exp", must_exist=False)
    images_dir = exp_root / exp_override / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    pattern = os.path.join(str(images_dir), f"rgb_env{env_id}_frame*.png")
    if not glob.glob(pattern):
        raise FileNotFoundError(f"No frames found for ffmpeg: {pattern}")

    video_out = exp_root / exp_override / "visualization.mp4"
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(video_out),
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    return str(video_out)

def create_rlgpu_env(**kwargs):
    """
    Works for:
      - Single GPU (python main.py ...)
      - Multi-GPU via torchrun (one process per GPU)

    Expects rl-games / your launcher to handle torch.distributed init.
    """
    import os
    import torch

    # ----- detect rank/local_rank from torchrun -----
    # Prefer torch.distributed if already initialized; otherwise use env vars.
    rank = 0
    local_rank = 0

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            rank = torch.distributed.get_rank()
        except Exception:
            rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        # torchrun always sets these; fall back to 0 for single-GPU runs
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # ----- per-rank seeding & device binding -----
    # mirror the old behavior: seed += rank
    cfg_train['params']['seed'] = cfg_train['params']['seed'] + rank

    # bind this process to its GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        args.device = 'cuda'
        args.device_id = local_rank
        args.rl_device = f'cuda:{local_rank}'
        cfg['rank'] = rank
        cfg['rl_device'] = f'cuda:{local_rank}'
    else:
        # Isaac Gym expects CUDA; this is just a safe fallback
        args.device = 'cpu'
        args.device_id = -1
        args.rl_device = 'cpu'
        cfg['rank'] = rank
        cfg['rl_device'] = 'cpu'

    # ----- build sim & env -----
    sim_params = parse_sim_params(args, cfg, cfg_train)
    task, env = parse_task(args, cfg, cfg_train, sim_params)

    print('rank:', rank, ' local_rank:', local_rank)
    print('num_envs: {:d}'.format(env.num_envs))
    print('num_actions: {:d}'.format(env.num_actions))
    print('num_obs: {:d}'.format(env.num_obs))
    print('num_states: {:d}'.format(env.num_states))

    # optional frame stacking
    frames = kwargs.pop('frames', 1)
    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)
    return env



class RLGPUAlgoObserver(AlgoObserver):
    def __init__(self, use_successes=True):
        self.use_successes = use_successes
        return

    def after_init(self, algo):
        self.algo = algo
        self.consecutive_successes = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.writer = self.algo.writer
        return

    def process_infos(self, infos, done_indices):
        if isinstance(infos, dict):
            if (self.use_successes == False) and 'consecutive_successes' in infos:
                cons_successes = infos['consecutive_successes'].clone()
                self.consecutive_successes.update(cons_successes.to(self.algo.ppo_device))
            if self.use_successes and 'successes' in infos:
                successes = infos['successes'].clone()
                self.consecutive_successes.update(successes[done_indices].to(self.algo.ppo_device))
        return

    def after_clear_stats(self):
        self.mean_scores.clear()
        return

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.consecutive_successes.current_size > 0:
            mean_con_successes = self.consecutive_successes.get_mean()
            self.writer.add_scalar('successes/consecutive_successes/mean', mean_con_successes, frame)
            self.writer.add_scalar('successes/consecutive_successes/iter', mean_con_successes, epoch_num)
            self.writer.add_scalar('successes/consecutive_successes/time', mean_con_successes, total_time)
        return


class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)
        # pdb.set_trace()
        self.use_global_obs = (self.env.num_states > 0)

        self.full_state = {}
        self.full_state["obs"] = self.reset()
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
        return

    def step(self, action):
        next_obs, reward, is_done, info = self.env.step(action)

        # todo: improve, return only dictinary
        self.full_state["obs"] = next_obs
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state, reward, is_done, info
        else:
            return self.full_state["obs"], reward, is_done, info

    def reset(self, env_ids=None):
        self.full_state["obs"] = self.env.reset(env_ids)
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state
        else:
            return self.full_state["obs"]

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space
        info['amp_observation_space'] = self.env.amp_observation_space

        if self.use_global_obs:
            info['state_space'] = self.env.state_space
            print(info['action_space'], info['observation_space'], info['state_space'])
        else:
            print(info['action_space'], info['observation_space'])

        return info


vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {
    'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
    'vecenv_type': 'RLGPU'})

def build_alg_runner(algo_observer):
    runner = Runner(algo_observer)

    runner.algo_factory.register_builder('intermimic', lambda **kwargs : intermimic_agent.InterMimicAgent(**kwargs))
    runner.player_factory.register_builder('intermimic', lambda **kwargs : intermimic_players.InterMimicPlayerContinuous(**kwargs))
    runner.model_builder.model_factory.register_builder('intermimic', lambda network, **kwargs : intermimic_models.ModelInterMimicContinuous(network))  
    runner.model_builder.network_factory.register_builder('intermimic', lambda **kwargs : intermimic_network_builder.InterMimicBuilder())

    return runner

def main():
    global args
    global cfg
    global cfg_train

    set_np_formatting()
    args = get_args()
    cfg, cfg_train, logdir = load_cfg(args)
    do_visualize = getattr(args, "visualize", False)

    # Visualization requires an actual viewer in this codebase.
    # Even if --headless is set, we force headless=false only when visualization is requested.
    if do_visualize:
        cfg["headless"] = False

    # Task render code checks EXP_DIR env var; expose CLI --exp_dir to it.
    if getattr(args, "exp_dir", "").strip():
        os.environ["EXP_DIR"] = args.exp_dir.strip()

    cfg_train['params']['seed'] = set_seed(cfg_train['params'].get("seed", -1), cfg_train['params'].get("torch_deterministic", False))

    cfg_train['params']['config']['multi_gpu'] = args.multi_gpu

    if args.horizon_length != -1:
        cfg_train['params']['config']['horizon_length'] = args.horizon_length

    if args.minibatch_size != -1:
        cfg_train['params']['config']['minibatch_size'] = args.minibatch_size
        
    if args.motion_file:
        cfg['env']['motion_file'] = args.motion_file

    if args.play_dataset:
        cfg['env']['playdataset'] = True
    if getattr(args, "play_dataset_physics", False):
        cfg['env']['playdatasetPhysics'] = True
    if getattr(args, "object_mode", ""):
        cfg['env']['objectMode'] = args.object_mode
    if getattr(args, "root_track_mode", ""):
        cfg['env']['rootTrackMode'] = args.root_track_mode
    if getattr(args, "disable_dataset_contact_overlay", False):
        cfg['env']['datasetContactOverlay'] = False

    if args.projtype:
        cfg['env']['projtype'] = args.projtype

    if args.cg1 != -1.:
        cfg['env']['rewardWeights']['cg1'] = args.cg1

    if args.cg2 != -1.:
        cfg['env']['rewardWeights']['cg2'] = args.cg2

    if args.ig != -1.:
        cfg['env']['rewardWeights']['ig'] = args.ig

    if args.op != -1.:
        cfg['env']['rewardWeights']['op'] = args.op

    if do_visualize:
        cfg['env']['saveImages'] = True

    if getattr(args, "num_episode", 0) > 0:
        cfg['env']['numEpisode'] = int(args.num_episode)
    
    if args.init_vel:
        cfg['env']['initVel'] = True

    if args.frames_scale!= 0.:
        cfg['env']['dataFramesScale'] = args.frames_scale

    if args.ball_size!= 0.:
        cfg['env']['ballSize'] = args.ball_size
    
    # Create default directories for weights and statistics
    cfg_train['params']['config']['train_dir'] = args.output_path
    
    vargs = vars(args)

    algo_observer = RLGPUAlgoObserver()

    runner = build_alg_runner(algo_observer)
    runner.load(cfg_train)
    runner.reset()
    try:
        if getattr(args, "test", False) and do_visualize and getattr(args, "exp_dir", "").strip():
            exp_override = _normalize_exp_override(args.exp_dir.strip())
            exp_root = resolve_repo_path("exp", must_exist=False)
            images_dir = exp_root / exp_override / "images"
            if images_dir.is_dir():
                try:
                    shutil.rmtree(images_dir)
                except Exception:
                    pass

        runner.run(vargs)
    finally:
        # Encode after the test run completes.
        if getattr(args, "test", False) and do_visualize and getattr(args, "exp_dir", "").strip():
            try:
                env_cfg = cfg.get("env", {}) if isinstance(cfg, dict) else {}
                fps = float(env_cfg.get("dataFPS", 30.0))
                out_video = _encode_visualization_video_from_frames(args.exp_dir.strip(), fps=fps, env_id=0)
                print(f"[run.py] visualization video saved to: {out_video}", flush=True)
            except Exception as e:
                # print(f"[run.py] Video encoding failed/skipped: {e}", flush=True)
                pass

    return

if __name__ == '__main__':
    main()
