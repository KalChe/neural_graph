import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
from typing import Tuple, Optional, List, Dict
import random
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
class RLConfig:
    STATE_DIM = 5
    NUM_ACTIONS = 3
    ACTION_SUPPRESS = 0
    ACTION_WARN = 1
    ACTION_EMERGENCY = 2
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    LEARNING_RATE = 0.0001
    GAMMA = 0.99
    TAU = 0.005
    BATCH_SIZE = 64
    BUFFER_SIZE = 10000
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.995
    REWARD_TRUE_POSITIVE = 10.0
    REWARD_TRUE_NEGATIVE = 1.0
    REWARD_FALSE_POSITIVE = -3.0
    REWARD_FALSE_NEGATIVE = -50.0
    REWARD_WARN = 0.5
    HISTORY_LENGTH = 10
class ReplayBuffer:
    def __init__(self, capacity: int=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append(Experience(state, action, reward, next_state, done))
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    def __len__(self) -> int:
        return len(self.buffer)
class DQNetwork(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int=64, num_layers: int=2):
        super().__init__()
        layers = []
        in_dim = state_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(0.1)])
            in_dim = hidden_dim
        self.feature_net = nn.Sequential(*layers)
        self.value_stream = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, num_actions))
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_net(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values
class StateTracker:
    def __init__(self, history_length: int=10):
        self.history_length = history_length
        self.reset()
    def reset(self):
        self.prob_history = deque(maxlen=self.history_length)
        self.forecast_history = deque(maxlen=self.history_length)
        self.time_since_alert = 0
        self.last_action = -1
    def update(self, detection_prob: float, forecast_prob: float, action: Optional[int]=None) -> np.ndarray:
        self.prob_history.append(detection_prob)
        self.forecast_history.append(forecast_prob)
        if action is not None:
            self.last_action = action
            if action != RLConfig.ACTION_SUPPRESS:
                self.time_since_alert = 0
            else:
                self.time_since_alert += 1
        return self.get_state()
    def get_state(self) -> np.ndarray:
        if len(self.prob_history) == 0:
            return np.zeros(RLConfig.STATE_DIM)
        probs = np.array(self.prob_history)
        current_prob = probs[-1]
        variance = probs.var() if len(probs) > 1 else 0.0
        if len(probs) > 1:
            rate_of_change = probs[-1] - probs[-2]
        else:
            rate_of_change = 0.0
        forecast_prob = self.forecast_history[-1] if self.forecast_history else 0.0
        time_normalized = min(self.time_since_alert / 10.0, 1.0)
        state = np.array([current_prob, variance, rate_of_change, forecast_prob, time_normalized], dtype=np.float32)
        return state
class SeizureAlertEnvironment:
    def __init__(self, config: RLConfig=RLConfig(), simulate_feedback: bool=True):
        self.config = config
        self.simulate_feedback = simulate_feedback
        self.state_tracker = StateTracker(config.HISTORY_LENGTH)
        self.current_state = None
        self.done = False
        self.step_count = 0
        self.max_steps = 1000
        self.stats = {'true_positives': 0, 'false_positives': 0, 'true_negatives': 0, 'false_negatives': 0, 'total_reward': 0}
    def reset(self) -> np.ndarray:
        self.state_tracker.reset()
        self.current_state = np.zeros(self.config.STATE_DIM)
        self.done = False
        self.step_count = 0
        for key in self.stats:
            self.stats[key] = 0
        return self.current_state
    def step(self, action: int, detection_prob: float, forecast_prob: float, ground_truth: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.step_count += 1
        reward = self._compute_reward(action, ground_truth, detection_prob)
        self.stats['total_reward'] += reward
        next_state = self.state_tracker.update(detection_prob, forecast_prob, action)
        self.current_state = next_state
        self.done = self.step_count >= self.max_steps
        info = {'action': action, 'ground_truth': ground_truth, 'detection_prob': detection_prob, 'stats': self.stats.copy()}
        return (next_state, reward, self.done, info)
    def _compute_reward(self, action: int, ground_truth: int, detection_prob: float) -> float:
        is_seizure = ground_truth == 1
        is_alert = action != self.config.ACTION_SUPPRESS
        if is_seizure:
            if action == self.config.ACTION_EMERGENCY:
                self.stats['true_positives'] += 1
                return self.config.REWARD_TRUE_POSITIVE
            elif action == self.config.ACTION_WARN:
                self.stats['true_positives'] += 1
                return self.config.REWARD_WARN
            else:
                self.stats['false_negatives'] += 1
                return self.config.REWARD_FALSE_NEGATIVE
        elif is_alert:
            self.stats['false_positives'] += 1
            return self.config.REWARD_FALSE_POSITIVE * (1 + detection_prob)
        else:
            self.stats['true_negatives'] += 1
            return self.config.REWARD_TRUE_NEGATIVE
    def get_statistics(self) -> Dict:
        tp = self.stats['true_positives']
        fp = self.stats['false_positives']
        tn = self.stats['true_negatives']
        fn = self.stats['false_negatives']
        total = tp + fp + tn + fn
        accuracy = (tp + tn) / max(total, 1)
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        precision = tp / max(tp + fp, 1)
        return {**self.stats, 'accuracy': accuracy, 'sensitivity': sensitivity, 'specificity': specificity, 'precision': precision}
class DQNAgent:
    def __init__(self, config: RLConfig=RLConfig(), device: str='cpu'):
        self.config = config
        self.device = torch.device(device)
        self.q_network = DQNetwork(config.STATE_DIM, config.NUM_ACTIONS, config.HIDDEN_DIM, config.NUM_LAYERS).to(self.device)
        self.target_network = DQNetwork(config.STATE_DIM, config.NUM_ACTIONS, config.HIDDEN_DIM, config.NUM_LAYERS).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(config.BUFFER_SIZE)
        self.epsilon = config.EPSILON_START
        self.training_step = 0
        self.episode = 0
    def select_action(self, state: np.ndarray, training: bool=True) -> int:
        if training and random.random() < self.epsilon:
            return random.randint(0, self.config.NUM_ACTIONS - 1)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=-1).item()
    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.replay_buffer.push(state, action, reward, next_state, done)
    def update(self) -> Optional[float]:
        if len(self.replay_buffer) < self.config.BATCH_SIZE:
            return None
        batch = self.replay_buffer.sample(self.config.BATCH_SIZE)
        states = torch.FloatTensor(np.array([e.state for e in batch])).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).to(self.device)
        dones = torch.FloatTensor([e.done for e in batch]).to(self.device)
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=-1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions)
            target_q = rewards.unsqueeze(1) + self.config.GAMMA * next_q * (1 - dones.unsqueeze(1))
        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        self._soft_update()
        self.training_step += 1
        return loss.item()
    def _soft_update(self):
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.config.TAU * param.data + (1 - self.config.TAU) * target_param.data)
    def decay_epsilon(self):
        self.epsilon = max(self.config.EPSILON_END, self.epsilon * self.config.EPSILON_DECAY)
    def end_episode(self):
        self.episode += 1
        self.decay_epsilon()
    def get_action_distribution(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy()[0]
    def save(self, path: str):
        checkpoint = {'q_network': self.q_network.state_dict(), 'target_network': self.target_network.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epsilon': self.epsilon, 'training_step': self.training_step, 'episode': self.episode}
        torch.save(checkpoint, path)
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        self.episode = checkpoint['episode']
class AgentTrainer:
    def __init__(self, agent: DQNAgent, env: SeizureAlertEnvironment):
        self.agent = agent
        self.env = env
        self.history = {'episode_rewards': [], 'losses': [], 'epsilons': [], 'sensitivities': [], 'specificities': []}
    def train_episode(self, model_predictions: List[Tuple[float, float, int]]) -> Dict:
        state = self.env.reset()
        total_loss = 0
        num_updates = 0
        for detection_prob, forecast_prob, label in model_predictions:
            action = self.agent.select_action(state, training=True)
            next_state, reward, done, info = self.env.step(action, detection_prob, forecast_prob, label)
            self.agent.store_transition(state, action, reward, next_state, done)
            loss = self.agent.update()
            if loss is not None:
                total_loss += loss
                num_updates += 1
            state = next_state
            if done:
                break
        self.agent.end_episode()
        stats = self.env.get_statistics()
        self.history['episode_rewards'].append(stats['total_reward'])
        self.history['epsilons'].append(self.agent.epsilon)
        self.history['sensitivities'].append(stats['sensitivity'])
        self.history['specificities'].append(stats['specificity'])
        if num_updates > 0:
            self.history['losses'].append(total_loss / num_updates)
        return stats
    def evaluate(self, model_predictions: List[Tuple[float, float, int]]) -> Dict:
        state = self.env.reset()
        for detection_prob, forecast_prob, label in model_predictions:
            action = self.agent.select_action(state, training=False)
            next_state, reward, done, info = self.env.step(action, detection_prob, forecast_prob, label)
            state = next_state
            if done:
                break
        return self.env.get_statistics()
if __name__ == '__main__':
    config = RLConfig()
    agent = DQNAgent(config)
    env = SeizureAlertEnvironment(config, simulate_feedback=True)
    trainer = AgentTrainer(agent, env)
    np.random.seed(42)
    n_samples = 200
    labels = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    detection_probs = []
    for label in labels:
        if label == 1:
            prob = np.random.beta(5, 2)
        else:
            prob = np.random.beta(2, 5)
        detection_probs.append(prob)
    forecast_probs = [p * 0.8 + np.random.random() * 0.2 for p in detection_probs]
    predictions = list(zip(detection_probs, forecast_probs, labels))
    for episode in range(10):
        np.random.shuffle(predictions)
        stats = trainer.train_episode(predictions)
        if (episode + 1) % 2 == 0:
            print(f"Episode {episode + 1}: Reward={stats['total_reward']:.2f}, Sens={stats['sensitivity']:.3f}, Spec={stats['specificity']:.3f}")