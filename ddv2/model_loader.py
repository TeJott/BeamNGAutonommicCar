import os
import sys
import numpy as np
import torch

# Ensure ddv2 package is importable and stubs are loaded
_ddv2_dir = os.path.dirname(os.path.abspath(__file__))
if _ddv2_dir not in sys.path:
    sys.path.insert(0, _ddv2_dir)
_ddv2_repo = os.path.join(_ddv2_dir, 'DiffusionDriveV2')
if _ddv2_repo not in sys.path:
    sys.path.insert(0, _ddv2_repo)

import ddv2_imports  # noqa: E402, F811


class DDV2Model:
    """Wrapper for the DiffusionDriveV2 end-to-end driving model."""

    def __init__(self, checkpoint_path=None, device=None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(_ddv2_dir, 'checkpoints', 'diffusiondrivev2_sel.ckpt')
        self.checkpoint_path = checkpoint_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.loaded = False
        self._captured_trajectories = None

    def load(self):
        """Load the DDV2 model from checkpoint."""
        try:
            from navsim.agents.diffusiondrivev2.diffusiondrivev2_model_sel import V2TransfuserModel
            from navsim.agents.diffusiondrivev2.diffusiondrivev2_sel_config import TransfuserConfig
        except ImportError as e:
            print(f"[DDV2] Failed to import model: {e}")
            return False

        if not os.path.exists(self.checkpoint_path):
            print(f"[DDV2] Checkpoint not found: {self.checkpoint_path}")
            return False

        try:
            cwd = os.getcwd()
            if not os.path.exists('kmeans_navsim_traj_20.npy'):
                os.chdir(_ddv2_repo)

            config = TransfuserConfig()
            self.model = V2TransfuserModel(config)

            os.chdir(cwd)

            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            state = checkpoint.get('state_dict', checkpoint)

            prefix = 'agent._transfuser_model.'
            new_state = {}
            for k, v in state.items():
                if k.startswith(prefix):
                    new_state[k[len(prefix):]] = v
                else:
                    new_state[k] = v

            missing, unexpected = self.model.load_state_dict(new_state, strict=False)
            if missing:
                print(f"[DDV2] Warning: {len(missing)} missing keys")
            if unexpected:
                print(f"[DDV2] Note: {len(unexpected)} unexpected keys (vocab/pdm, OK for inference)")

            # Monkey-patch PDM scoring to capture trajectories for inference
            self._patch_for_inference()

            self.model.to(self.device)
            self.model.eval()
            self.loaded = True

            total = sum(p.numel() for p in self.model.parameters()) / 1e6
            print(f"[DDV2] Model loaded. {total:.1f}M params on {self.device}")
            return True

        except Exception as e:
            import traceback
            print(f"[DDV2] Failed to load: {e}")
            traceback.print_exc()
            return False

    def _patch_for_inference(self):
        """Monkey-patch the trajectory head to skip PDM scoring and return trajectories."""
        traj_head = self.model._trajectory_head
        model_self = self

        original_get_pdm = traj_head.get_pdm_score_para

        def patched_get_pdm(trajectory, metric_cache_path):
            # Save trajectories for extraction
            model_self._captured_trajectories = trajectory.detach().cpu().numpy()
            B = trajectory.shape[0]
            scores = np.ones((B, trajectory.shape[1]), dtype=np.float32)
            return torch.from_numpy(scores).to(trajectory.device), None, None, None

        traj_head.get_pdm_score_para = patched_get_pdm

    def predict(self, features):
        """
        Run inference on pre-built feature tensors.

        Args:
            features: dict with 'camera_feature', 'lidar_feature', 'status_feature'
                      Each (1, C, H, W) numpy arrays

        Returns:
            trajectories: numpy array (1, K, T, 3) — K proposals, T timesteps, (x, y, yaw)
            scores: numpy array (1, K) — confidence score per trajectory
        """
        if not self.loaded:
            return np.zeros((1, 4, 8, 3), dtype=np.float32), np.zeros((1, 4), dtype=np.float32)

        self._captured_trajectories = None

        camera = torch.from_numpy(features['camera_feature']).to(self.device)
        lidar = torch.from_numpy(features['lidar_feature']).to(self.device)
        status = torch.from_numpy(features['status_feature']).to(self.device)

        feat_dict = {
            'camera_feature': camera,
            'lidar_feature': lidar,
            'status_feature': status,
        }

        with torch.no_grad():
            try:
                self.model(feat_dict)
            except Exception:
                pass  # PDM scoring may fail; trajectories captured via patch

        if self._captured_trajectories is not None:
            traj_np = self._captured_trajectories  # (B, K, T, 3)
        else:
            traj_np = np.zeros((1, 4, 8, 3), dtype=np.float32)

        # Remove batch dim: (K, T, 3)
        trajectories = traj_np[0]
        num_modes = trajectories.shape[0]

        # All modes have equal score (dummy) since PDM scoring is disabled
        scores = np.ones(num_modes, dtype=np.float32)
        return trajectories, scores

    def is_available(self):
        return self.loaded
