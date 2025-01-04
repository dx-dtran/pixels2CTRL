import logging
import numpy as np
import pickle
import gymnasium as gym
import ale_py
import time
import os
from datetime import datetime

# Register ALE environments
gym.register_envs(ale_py)

# Set up logging
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_name = f"save_pong_ppo_{timestamp}"
os.makedirs(folder_name, exist_ok=True)
log_filename = os.path.join(folder_name, f"training_log_{timestamp}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename, mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger()

# Hyperparameters
H = 200  # number of hidden layer neurons
batch_size = 10  # episodes per batch
mini_batch_size = 5  # mini-batch size for PPO updates
epochs = 4  # number of epochs per PPO update
learning_rate = 1e-3
gamma = 0.99  # discount factor
lam = 0.95  # GAE lambda
clip_epsilon = 0.2  # PPO clipping parameter
resume = False  # resume from previous checkpoint?
render = False

# Model initialization
D = 80 * 80  # input dimensionality: 80x80 grid
if resume:
    actor, critic = pickle.load(open("save_ppo.p", "rb"))
else:
    actor = {
        "W1": np.random.randn(H, D) / np.sqrt(D),
        "W2": np.random.randn(H) / np.sqrt(H),
    }
    critic = {
        "W1": np.random.randn(H, D) / np.sqrt(D),
        "W2": np.random.randn(H) / np.sqrt(H),
    }

# Adam optimizer parameters for actor and critic
adam_cache = {
    "actor": {
        "m": {k: np.zeros_like(v) for k, v in actor.items()},
        "v": {k: np.zeros_like(v) for k, v in actor.items()},
    },
    "critic": {
        "m": {k: np.zeros_like(v) for k, v in critic.items()},
        "v": {k: np.zeros_like(v) for k, v in critic.items()},
    },
}
beta1, beta2 = 0.9, 0.999
epsilon = 1e-8


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def prepro(I):
    """Preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector"""
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(float).ravel()


def discount_rewards(r):
    """Compute discounted rewards"""
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0  # reset the sum, since this was a game boundary
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(model, x):
    h = np.dot(model["W1"], x)
    h[h < 0] = 0  # ReLU nonlinearity
    logp = np.dot(model["W2"], h)
    p = sigmoid(logp)
    return p, h  # return probability and hidden state


def value_forward(model, x):
    h = np.dot(model["W1"], x)
    h[h < 0] = 0  # ReLU nonlinearity
    v = np.dot(model["W2"], h)
    return v


def compute_advantages(rewards, values, gamma, lam):
    advantages = np.zeros_like(rewards)
    last_adv = 0
    for t in reversed(range(len(rewards) - 1)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        advantages[t] = last_adv = delta + gamma * lam * last_adv
    return advantages


def compute_gradients(loss, model):
    """Compute gradients for a simple 2-layer model"""
    grads = {}
    dW2 = loss * model["W2"]
    dW1 = loss * model["W1"]
    grads["W1"] = dW1
    grads["W2"] = dW2
    return grads


def ppo_update(actor, critic, observations, actions, advantages, returns, old_probs):
    """Perform PPO update"""
    for _ in range(epochs):
        for start in range(0, len(observations), mini_batch_size):
            end = start + mini_batch_size
            mb_obs = observations[start:end]
            mb_actions = actions[start:end]
            mb_advantages = advantages[start:end]
            mb_returns = returns[start:end]
            mb_old_probs = old_probs[start:end]

            # Forward pass
            probs, _ = policy_forward(actor, mb_obs.T)
            values = value_forward(critic, mb_obs.T)

            # Compute PPO loss
            ratios = probs / mb_old_probs
            clipped_ratios = np.clip(ratios, 1 - clip_epsilon, 1 + clip_epsilon)
            actor_loss = -np.mean(
                np.minimum(ratios * mb_advantages, clipped_ratios * mb_advantages)
            )

            critic_loss = np.mean((values - mb_returns) ** 2)

            total_loss = actor_loss + 0.5 * critic_loss

            # Compute gradients
            d_actor = compute_gradients(actor_loss, actor)
            d_critic = compute_gradients(critic_loss, critic)

            # Update parameters using Adam
            for k in actor:
                m, v = adam_cache["actor"]["m"][k], adam_cache["actor"]["v"][k]
                m = beta1 * m + (1 - beta1) * d_actor[k]
                v = beta2 * v + (1 - beta2) * (d_actor[k] ** 2)
                actor[k] += learning_rate * m / (np.sqrt(v) + epsilon)

            for k in critic:
                m, v = adam_cache["critic"]["m"][k], adam_cache["critic"]["v"][k]
                m = beta1 * m + (1 - beta1) * d_critic[k]
                v = beta2 * v + (1 - beta2) * (d_critic[k] ** 2)
                critic[k] += learning_rate * m / (np.sqrt(v) + epsilon)


# Initialize environment
env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
observation, _ = env.reset()
prev_x = None

observations, actions, rewards, values, old_probs = [], [], [], [], []
episode_number = 0
reward_sum = 0
running_reward = None
start_time = time.time()  # track total duration
batch_start_time = time.time()  # track batch duration

while True:
    if render:
        env.render()

    # Preprocess observation
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    # Forward pass through policy and value networks
    aprob, _ = policy_forward(actor, x)
    v = value_forward(critic, x)
    action = 2 if np.random.uniform() < aprob else 3

    # Record data
    observations.append(x)
    actions.append(action)
    old_probs.append(aprob if action == 2 else 1 - aprob)
    values.append(v)

    # Step environment
    observation, reward, done, _, _ = env.step(action)
    rewards.append(reward)
    reward_sum += reward

    if done:
        episode_number += 1

        # Compute discounted returns and advantages
        returns = discount_rewards(np.array(rewards))
        advantages = compute_advantages(rewards, values, gamma, lam)

        # Update policy and value networks
        ppo_update(
            actor,
            critic,
            np.array(observations),
            np.array(actions),
            advantages,
            returns,
            np.array(old_probs),
        )

        # Log progress
        running_reward = (
            reward_sum
            if running_reward is None
            else running_reward * 0.99 + reward_sum * 0.01
        )
        episode_duration = time.time() - start_time
        logger.info(
            f"Episode {episode_number} completed. Reward total: {reward_sum}. Running mean: {running_reward:.2f}. Duration: {episode_duration:.2f} seconds"
        )

        if episode_number % batch_size == 0:
            batch_duration = time.time() - batch_start_time
            logger.info(
                f"Batch update completed. Duration: {batch_duration:.2f} seconds"
            )
            batch_start_time = time.time()  # reset batch start time

        if episode_number % 500 == 0:
            pickle.dump(
                (actor, critic),
                open(os.path.join(folder_name, f"save_ppo_{episode_number}.p"), "wb"),
            )

        # Reset episode data
        observations, actions, rewards, values, old_probs = [], [], [], [], []
        reward_sum = 0
        observation, _ = env.reset()
        prev_x = None
        start_time = time.time()
