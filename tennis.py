import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from torchvision import transforms
import multiprocessing as mp
import cv2
import os


gym.register_envs(ale_py)


def preprocess_frame(frame):
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
        ]
    )
    frame_tensor = transform(frame).squeeze(0)
    return frame_tensor.numpy()


def stack_frames(stacked_frames, new_frame, is_new_episode):
    if is_new_episode:
        stacked_frames = [new_frame for _ in range(4)]
    else:
        stacked_frames.append(new_frame)
        stacked_frames.pop(0)
    return np.stack(stacked_frames, axis=0), stacked_frames


class PPOActorCritic(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(PPOActorCritic, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(nn.Linear(conv_out_size, 512), nn.ReLU())
        self.policy = nn.Linear(512, n_actions)
        self.value = nn.Linear(512, 1)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape).to(next(self.parameters()).device))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        fc_out = self.fc(conv_out)
        return self.policy(fc_out), self.value(fc_out)

    def act(self, x):
        logits, value = self.forward(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value


class PPO:
    def __init__(
        self, actor_critic, lr=1e-4, gamma=0.99, clip_epsilon=0.2, c1=0.5, c2=0.01
    ):
        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.c1 = c1
        self.c2 = c2

    def compute_advantages(self, rewards, values, dones):
        returns = []
        advantages = []
        gae = 0
        next_value = 0
        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step]
                + self.gamma * next_value * (1 - dones[step])
                - values[step]
            )
            gae = delta + self.gamma * gae * (1 - dones[step])
            returns.insert(0, gae + values[step])
            advantages.insert(0, gae)
            next_value = values[step]
        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(
            returns, dtype=torch.float32
        )

    def update(self, obs, actions, log_probs, returns, advantages):
        for _ in range(4):
            logits, values = self.actor_critic(obs)
            probs = Categorical(logits=logits)
            new_log_probs = probs.log_prob(actions)
            entropy = probs.entropy()

            ratio = (new_log_probs - log_probs).exp()
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                * advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = self.c1 * (returns - values).pow(2).mean()

            entropy_loss = -self.c2 * entropy.mean()

            loss = policy_loss + value_loss + entropy_loss

            # Log losses
            print(
                f"Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, Entropy Loss: {entropy_loss.item():.4f}, Total Loss: {loss.item():.4f}"
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
            self.optimizer.step()

            # Check for NaN values in gradients and parameters
            for name, param in self.actor_critic.named_parameters():
                if torch.isnan(param).any():
                    print(f"NaN detected in parameter: {name}")
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN detected in gradient: {name}")


def render_game(
    actor_critic, stacked_frames, is_new_episode, episode, output_dir="rendered_games"
):
    os.makedirs(output_dir, exist_ok=True)
    env = gym.make("ALE/Tennis-v5", render_mode="rgb_array")
    obs = preprocess_frame(env.reset()[0])
    obs_stack, stacked_frames = stack_frames(stacked_frames, obs, is_new_episode)
    done = False
    frame_count = 0

    video_path = os.path.join(output_dir, f"episode_{episode}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30
    frame_size = (160, 210)
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

    while not done:
        frame = env.render()
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        obs_tensor = (
            torch.from_numpy(np.expand_dims(obs_stack, axis=0))
            .float()
            .to(next(actor_critic.parameters()).device)
        )
        action, _, _, _ = actor_critic.act(obs_tensor)
        next_obs, _, done, _, _ = env.step(action.item())
        next_obs_processed = preprocess_frame(next_obs)
        obs_stack, stacked_frames = stack_frames(
            stacked_frames, next_obs_processed, False
        )
        frame_count += 1

    video_writer.release()
    env.close()


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    env = gym.make("ALE/Tennis-v5")
    n_actions = env.action_space.n
    input_shape = (4, 84, 84)

    actor_critic = PPOActorCritic(input_shape, n_actions).to(device)
    ppo = PPO(actor_critic)

    obs = preprocess_frame(env.reset()[0])
    stacked_frames = [obs for _ in range(4)]
    obs_stack = np.stack(stacked_frames, axis=0)

    for batch in range(10000):
        total_rewards = []
        game_reward = 0
        if batch == 0 or (batch + 1) % 100 == 0:
            render_process = mp.Process(
                target=render_game,
                args=(
                    actor_critic,
                    stacked_frames,
                    True,
                    batch + 1,
                    "rendered_games",
                ),
                daemon=True,
            )
            render_process.start()

        rewards, log_probs, values, actions, dones = [], [], [], [], []

        for step in range(2048):
            obs_tensor = (
                torch.from_numpy(np.expand_dims(obs_stack, axis=0)).float().to(device)
            )

            action, log_prob, entropy, value = actor_critic.act(obs_tensor)
            next_obs, reward, done, _, _ = env.step(action.item())

            if np.isnan(reward) or np.isinf(reward):
                print("Invalid reward detected! Setting reward to 0.")
                reward = 0

            next_obs_processed = preprocess_frame(next_obs)
            obs_stack, stacked_frames = stack_frames(
                stacked_frames, next_obs_processed, done
            )

            rewards.append(reward)
            game_reward += reward
            log_probs.append(log_prob)
            values.append(value)
            actions.append(action)
            dones.append(done)

            if done:
                total_rewards.append(game_reward)
                game_reward = 0
                obs = preprocess_frame(env.reset()[0])
                obs_stack, stacked_frames = stack_frames([], obs, is_new_episode=True)

        advantages, returns = ppo.compute_advantages(rewards, values, dones)

        # Log rewards and advantages for debugging
        print(f"Rewards: {rewards}")
        print(f"Rewards Len: {len(rewards)}")
        print(f"Advantages: {advantages.tolist()}")
        print(f"Returns: {returns.tolist()}")

        ppo.update(
            torch.from_numpy(np.expand_dims(obs_stack, axis=0)).float().to(device),
            torch.tensor(actions, dtype=torch.int64).to(device),
            torch.tensor(log_probs, dtype=torch.float32).to(device),
            returns.clone().detach().to(device),
            advantages.clone().detach().to(device),
        )

        if total_rewards:
            print(
                f"Batch {batch + 1} completed - Average Reward per Game: {np.mean(total_rewards):.2f}"
            )
        else:
            print(f"Batch {batch + 1} completed - No games finished.")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    train()
