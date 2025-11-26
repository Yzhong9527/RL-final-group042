import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models import EEGNetHybridNorm

class BCI2DCursorEnv(gym.Env):


    def __init__(self, eeg_segments, labels, sl_model_path,
                 feature_npz_path=None, method='hybrid', success_radius=2.0):
        super().__init__()

        self.eeg_segments = eeg_segments  # (N_segments, 6, 22, 1000)
        self.labels = labels              # (N_segments, 6)
        self.num_segments = len(eeg_segments)
        self.segment_len = eeg_segments.shape[1]
        self.success_radius = float(success_radius)

        self.method = method
        self.grid_size = 20
        self.max_steps = 6

        self.sl_decoder = self._load_sl_decoder(sl_model_path)
        if method == 'hybrid' and feature_npz_path is not None:
            self.feature_bank = np.load(feature_npz_path)['features']
            print(f" Hybrid : {self.feature_bank.shape}")
        else:
            self.feature_bank = None

        self.action_space = spaces.Discrete(4)
        if method == 'sl-only':
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        elif method == 'sl-rl':
            self.observation_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        else:

            self.observation_space = spaces.Box(low=-1, high=1, shape=(168,), dtype=np.float32)

        self.current_segment_idx = None
        self.current_segment = None
        self.segment_labels = None
        self.trial_ptr = 0
        self.cursor_pos = None
        self.target_pos = None
        self._last_distance = None
        self.steps = 0

    def _load_sl_decoder(self, path):
        
        model = EEGNetHybridNorm(num_classes=4, num_channels=22, sample_length=1000)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()
        print(f" SL : {path}")
        return model

    def _get_sl_probabilities(self, eeg_signal):
        with torch.no_grad():
            eeg_tensor = torch.FloatTensor(eeg_signal).unsqueeze(0).unsqueeze(0)
            probs = self.sl_decoder(eeg_tensor)
            probs = F.softmax(probs, dim=1).cpu().numpy().flatten()
            return self._apply_noise(probs, sigma=0.05).astype(np.float32)

    def _get_eeg_features(self):
        if self.feature_bank is not None:
            global_idx = self.current_segment_idx * self.segment_len + self.trial_ptr

            if global_idx >= len(self.feature_bank):
                global_idx = len(self.feature_bank) - 1

            feats = self.feature_bank[global_idx]
            if feats.shape[0] > 160:
                feats = feats[:160]
            return feats.astype(np.float32)
        else:
            return np.zeros(160, dtype=np.float32)

    def _build_state(self, eeg_signal):
        sl_probs = self._get_sl_probabilities(eeg_signal)
        if self.method == 'sl-only':
            return sl_probs
        elif self.method == 'sl-rl':
            return np.concatenate([sl_probs, self.cursor_pos, self.target_pos]).astype(np.float32)
        else:
            eeg_features = self._get_eeg_features()
            return np.concatenate([eeg_features, sl_probs, self.cursor_pos, self.target_pos]).astype(np.float32)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_segment_idx = np.random.randint(0, self.num_segments)
        self.current_segment = self.eeg_segments[self.current_segment_idx]
        self.segment_labels = self.labels[self.current_segment_idx]
        self.trial_ptr = 0
        self.steps = 0

        self.cursor_pos = np.array([self.grid_size // 2, self.grid_size // 2], dtype=np.float32)
        target_offsets = {1: [-5, 0], 2: [5, 0], 3: [0, -5], 4: [0, 5]}
        label = int(self.segment_labels[-1])  
        offset = np.array(target_offsets.get(label, [0, 0]))
        self.target_pos = np.clip(self.cursor_pos + offset, 0, self.grid_size - 1)
        self._last_distance = np.linalg.norm(self.cursor_pos - self.target_pos)

        eeg_signal = self.current_segment[self.trial_ptr]
        state = self._build_state(eeg_signal)
        return state, {}

    def step(self, action):

        self.steps += 1


        move_map = {
            0: np.array([0, 1]),    # up
            1: np.array([0, -1]),   # down
            2: np.array([-1, 0]),   # left
            3: np.array([1, 0])     # right
        }
        move = move_map[action]
        self.cursor_pos = np.clip(self.cursor_pos + move, 0, self.grid_size - 1)

        reward, done = self._calculate_reward()


        self.trial_ptr += 1
        if self.trial_ptr < self.segment_len:
            eeg_signal = self.current_segment[self.trial_ptr]
            next_state = self._build_state(eeg_signal)
        else:
            
            done = True
            next_state = np.zeros_like(self._build_state(self.current_segment[0]))

        if self.steps >= self.segment_len:
            done = True

        return next_state, reward, done, False, {}

    def _calculate_reward(self):
        new_dist = np.linalg.norm(self.cursor_pos - self.target_pos)
        old_dist = self._last_distance
        reward = -1
        reward += (old_dist - new_dist) * 10

        done = False
        if new_dist < 0.1:
            reward += 100
            done = True

        self._last_distance = new_dist
        return reward, done
    def expert_sl_action(self):
        """
        Convert SL decoder (class 1–4) → RL action (0–3)
        """
        eeg_signal = self.current_segment[self.trial_ptr]
        eeg_tensor = torch.FloatTensor(eeg_signal).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            sl_probs = self.sl_decoder(eeg_tensor)
            sl_probs = F.softmax(sl_probs, dim=1).cpu().numpy().flatten()

        sl_class = int(np.argmax(sl_probs)) + 1  # SL 类别是 {1,2,3,4}

        # SL label → RL action 对应表
        mapping = {
            3: 0,  # 上
            4: 1,  # 下
            1: 2,  # 左
            2: 3   # 右
        }

        return mapping[sl_class]



    def _apply_noise(self, probs, sigma=0.0):
        if sigma == 0:
            return probs / np.sum(probs)
        noise = np.random.normal(0, sigma, size=probs.shape)
        noisy = np.clip(probs + noise, 0, 1)
        return noisy / np.sum(noisy)

    def render(self):
        plt.figure(figsize=(5, 5))
        plt.scatter(*self.target_pos, c='r', label='Target')
        plt.scatter(*self.cursor_pos, c='b', label='Cursor')
        plt.xlim(0, self.grid_size)
        plt.ylim(0, self.grid_size)
        plt.legend()
        plt.title(f"Step: {self.steps}")
        plt.show()