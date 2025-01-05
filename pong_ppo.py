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
H = 200  # Number of hidden layer neurons
batch_size = 10  # Episodes per batch
mini_batch_size = 1000  # Mini-batch size for PPO updates
epochs = 4  # Number of epochs per PPO update
learning_rate = 1e-3
gamma = 0.99  # Discount factor
lam = 0.95  # GAE lambda
clip_epsilon = 0.2  # PPO clipping parameter
resume = False  # Resume from previous checkpoint?
render = False  # Render the environment?

# Model initialization
D = 80 * 80  # Input dimensionality: 80x80 grid
if resume:
    actor, critic = pickle.load(open("save_ppo.p", "rb"))
else:
    actor = {
        "W1": np.random.randn(H, D) / np.sqrt(D),
        "b1": np.zeros((H, 1)),
        "W2": np.random.randn(1, H) / np.sqrt(H),
        "b2": np.zeros((1, 1)),
    }
    critic = {
        "W1": np.random.randn(H, D) / np.sqrt(D),
        "b1": np.zeros((H, 1)),
        "W2": np.random.randn(1, H) / np.sqrt(H),
        "b2": np.zeros((1, 1)),
    }

# Adam optimizer parameters for actor and critic
adam_cache = {
    "actor": {
        "mW1": np.zeros_like(actor["W1"]),
        "vW1": np.zeros_like(actor["W1"]),
        "mb1": np.zeros_like(actor["b1"]),
        "vb1": np.zeros_like(actor["b1"]),
        "mW2": np.zeros_like(actor["W2"]),
        "vW2": np.zeros_like(actor["W2"]),
        "mb2": np.zeros_like(actor["b2"]),
        "vb2": np.zeros_like(actor["b2"]),
    },
    "critic": {
        "mW1": np.zeros_like(critic["W1"]),
        "vW1": np.zeros_like(critic["W1"]),
        "mb1": np.zeros_like(critic["b1"]),
        "vb1": np.zeros_like(critic["b1"]),
        "mW2": np.zeros_like(critic["W2"]),
        "vW2": np.zeros_like(critic["W2"]),
        "mb2": np.zeros_like(critic["b2"]),
        "vb2": np.zeros_like(critic["b2"]),
    },
}
beta1, beta2 = 0.9, 0.999
epsilon = 1e-8


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def prepro(I):
    """Preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector"""
    I = I[35:195]  # Crop
    I = I[::2, ::2, 0]  # Downsample by factor of 2
    I[I == 144] = 0  # Erase background (background type 1)
    I[I == 109] = 0  # Erase background (background type 2)
    I[I != 0] = 1  # Everything else (paddles, ball) just set to 1
    return I.astype(float).ravel()


def discount_rewards(r):
    """Compute discounted rewards"""
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(r.size)):
        if r[t] != 0:
            running_add = 0  # Reset the sum, since this was a game boundary
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(model, x):
    """
    Forward pass for the policy network.
    Args:
        model: Actor network parameters
        x: Input features (D x N)
    Returns:
        p: Probability of taking action 2 (1 x N)
        h: Hidden layer activations (H x N)
    """
    z1 = np.dot(model["W1"], x) + model["b1"]  # (H x N)
    h = np.maximum(z1, 0)  # ReLU activation
    logp = np.dot(model["W2"], h) + model["b2"]  # (1 x N)
    p = sigmoid(logp)  # (1 x N)
    return p, h


def value_forward(model, x):
    """
    Forward pass for the value network.
    Args:
        model: Critic network parameters
        x: Input features (D x N)
    Returns:
        v: Value estimates (1 x N)
        h: Hidden layer activations (H x N)
    """
    z1 = np.dot(model["W1"], x) + model["b1"]  # (H x N)
    h = np.maximum(z1, 0)  # ReLU activation
    v = np.dot(model["W2"], h) + model["b2"]  # (1 x N)
    return v, h


def compute_advantages(rewards, values, gamma, lam):
    """Compute Generalized Advantage Estimation (GAE)"""
    advantages = np.zeros_like(rewards)
    last_adv = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            delta = rewards[t] - values[t]
        else:
            delta = rewards[t] + gamma * values[t + 1] - values[t]
        advantages[t] = last_adv = delta + gamma * lam * last_adv
    return advantages


def ppo_actor_grad_piecemeal(p, old_probs, advantages, actions, eps=0.2):
    """
    Compute dL/d(logit_i) for each sample i based on PPO's clipped objective.
    Args:
        p: shape (N,) => actor's probability of action=2
        old_probs: shape (N,) => old pi(a_i)
        advantages: shape (N,)
        actions: shape (N,), 2 or 3
        eps: clipping parameter
    Returns:
        dL_dlogits: shape (N,)
    """
    N = len(p)

    # 1) ratio_i
    # define pi_new[i] = p[i] if a=2 else (1-p[i])
    a2_mask = actions == 2  # boolean mask
    pi_new = np.where(a2_mask, p, 1 - p)  # shape (N,)
    ratio = pi_new / old_probs  # shape (N,)

    # 2) surr1, surr2
    surr1 = ratio * advantages
    clipped_ratio = np.clip(ratio, 1.0 - eps, 1.0 + eps)
    surr2 = clipped_ratio * advantages

    # 3) Determine which surrogate is active
    use_surr1 = surr1 < surr2  # boolean mask, shape (N,)

    # Initialize dL/dr_i
    dL_dratio = np.zeros(N, dtype=np.float32)

    # Case A: surr1 < surr2 (unclipped)
    dL_dratio[use_surr1] = -advantages[use_surr1]

    # Case B: surr2 <= surr1 (clipped)
    # Further split into whether ratio is within [1 - eps, 1 + eps]
    not_use_surr1 = ~use_surr1
    # Determine if ratio is clipped
    ratio_clipped = (ratio < (1.0 - eps)) | (ratio > (1.0 + eps))
    # If not clipped, derivative is -A_i
    dL_dratio[not_use_surr1 & ~ratio_clipped] = -advantages[
        not_use_surr1 & ~ratio_clipped
    ]
    # If clipped, derivative is 0 (already initialized)

    # 4) Average over N
    dL_dratio /= N  # shape (N,)

    # 5) Chain rule: ratio = pi_new / old_prob
    # => derivative wrt pi_new = 1 / old_prob
    # if a=2 => pi_new = p => d pi_new/d p = +1
    # if a=3 => pi_new = 1-p => d pi_new/d p = -1
    dL_dpi_new = dL_dratio / old_probs  # shape (N,)
    dL_dp = np.where(a2_mask, dL_dpi_new, -dL_dpi_new)  # shape (N,)

    # 6) p = sigmoid(logit), => dp/d(logit) = p*(1-p)
    dL_dlogits = dL_dp * p * (1.0 - p)  # shape (N,)

    return dL_dlogits


def update_actor(
    actor, adam_cache_actor, mb_obs, mb_actions, mb_advantages, mb_old_probs
):
    """Perform manual backpropagation and update for the actor network"""
    # Forward pass for actor
    p, h_actor = policy_forward(actor, mb_obs)  # p: (1 x N), h_actor: (H x N)
    p = p.squeeze()  # shape (N,)

    # Compute gradients using piecewise logic
    dL_dlogits = ppo_actor_grad_piecemeal(
        p=p,
        old_probs=mb_old_probs,
        advantages=mb_advantages,
        actions=mb_actions,
        eps=clip_epsilon,
    )  # shape (N,)

    # Compute gradients for W2 and b2
    # dL/dW2 = sum(dL/dlogits * h_actor) / N
    dL_dW2_actor = np.dot(dL_dlogits, h_actor.T) / mb_obs.shape[1]  # (1 x H)

    # dL/db2 = sum(dL/dlogits) / N
    dL_db2_actor = (
        np.sum(dL_dlogits, axis=0, keepdims=True).T / mb_obs.shape[1]
    )  # (1 x 1)

    # Compute gradient w.r. to h_actor
    dL_dh_actor = np.dot(actor["W2"].T, dL_dlogits.reshape(1, -1))  # (H x N)

    # Gradient through ReLU
    dL_dh_actor[h_actor <= 0] = 0  # (H x N)

    # Compute gradients for W1 and b1
    # dL/dW1 = sum(dL/dh * mb_obs.T) / N
    dL_dW1_actor = np.dot(dL_dh_actor, mb_obs.T) / mb_obs.shape[1]  # (H x D)

    # dL/db1 = sum(dL/dh) / N
    dL_db1_actor = (
        np.sum(dL_dh_actor, axis=1, keepdims=True) / mb_obs.shape[1]
    )  # (H x 1)

    # ---------- Adam Update for Actor ----------
    for param, dparam, cache_m, cache_v in zip(
        ["W1", "b1", "W2", "b2"],
        [dL_dW1_actor, dL_db1_actor, dL_dW2_actor, dL_db2_actor],
        ["mW1", "mb1", "mW2", "mb2"],
        ["vW1", "vb1", "vW2", "vb2"],
    ):
        # Update biased first moment estimate
        adam_cache_actor[cache_m] = (
            beta1 * adam_cache_actor[cache_m] + (1 - beta1) * dparam
        )
        # Update biased second raw moment estimate
        adam_cache_actor[cache_v] = beta2 * adam_cache_actor[cache_v] + (1 - beta2) * (
            dparam**2
        )
        # Compute bias-corrected first moment estimate
        m_hat = adam_cache_actor[cache_m] / (1 - beta1**epochs)
        # Compute bias-corrected second raw moment estimate
        v_hat = adam_cache_actor[cache_v] / (1 - beta2**epochs)
        # Update parameters
        actor[param] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)


def compute_critic_gradients(critic, mb_obs, mb_returns):
    """
    Compute gradients for the critic network.
    Args:
        critic: Critic network parameters
        mb_obs: Mini-batch observations (D x N)
        mb_returns: Mini-batch discounted returns (N,)
    Returns:
        gradients: Dictionary containing gradients for W1, b1, W2, b2
    """
    # Forward pass for critic
    v, h_critic = value_forward(critic, mb_obs)  # v: (1 x N), h_critic: (H x N)
    v = v.squeeze()  # shape (N,)

    # Compute critic loss (Mean Squared Error)
    error = v - mb_returns  # shape (N,)
    critic_loss = np.mean(error**2)  # Scalar

    # Gradient of critic loss w.r. to v
    dL_dv = (2 * error) / mb_obs.shape[1]  # shape (N,)

    # Compute gradients for W2 and b2
    # dL/dW2 = sum(dL/dv * h_critic) / N
    dL_dW2_critic = np.dot(dL_dv, h_critic.T) / mb_obs.shape[1]  # (1 x H)

    # dL/db2 = sum(dL/dv) / N
    dL_db2_critic = np.sum(dL_dv, axis=0, keepdims=True).T / mb_obs.shape[1]  # (1 x 1)

    # Compute gradient w.r. to h_critic
    dL_dh_critic = np.dot(critic["W2"].T, dL_dv.reshape(1, -1))  # (H x N)

    # Gradient through ReLU
    dL_dh_critic[h_critic <= 0] = 0  # (H x N)

    # Compute gradients for W1 and b1
    # dL/dW1 = sum(dL/dh * mb_obs.T) / N
    dL_dW1_critic = np.dot(dL_dh_critic, mb_obs.T) / mb_obs.shape[1]  # (H x D)

    # dL/db1 = sum(dL/dh) / N
    dL_db1_critic = (
        np.sum(dL_dh_critic, axis=1, keepdims=True) / mb_obs.shape[1]
    )  # (H x 1)

    gradients = {
        "W1": dL_dW1_critic,
        "b1": dL_db1_critic,
        "W2": dL_dW2_critic,
        "b2": dL_db2_critic,
    }

    return gradients


def update_critic(critic, adam_cache_critic, mb_obs, mb_returns):
    """Perform manual backpropagation and update for the critic network"""
    # Compute gradients for critic
    gradients = compute_critic_gradients(critic, mb_obs, mb_returns)

    # ---------- Adam Update for Critic ----------
    for param, dparam, cache_m, cache_v in zip(
        ["W1", "b1", "W2", "b2"],
        [gradients["W1"], gradients["b1"], gradients["W2"], gradients["b2"]],
        ["mW1", "mb1", "mW2", "mb2"],
        ["vW1", "vb1", "vW2", "vb2"],
    ):
        # Update biased first moment estimate
        adam_cache_critic[cache_m] = (
            beta1 * adam_cache_critic[cache_m] + (1 - beta1) * dparam
        )
        # Update biased second raw moment estimate
        adam_cache_critic[cache_v] = beta2 * adam_cache_critic[cache_v] + (
            1 - beta2
        ) * (dparam**2)
        # Compute bias-corrected first moment estimate
        m_hat = adam_cache_critic[cache_m] / (1 - beta1**epochs)
        # Compute bias-corrected second raw moment estimate
        v_hat = adam_cache_critic[cache_v] / (1 - beta2**epochs)
        # Update parameters
        critic[param] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)


def compute_entropy(p):
    """
    Compute entropy of the policy.
    Args:
        p: shape (N,)
    Returns:
        entropy: scalar
    """
    return -np.mean(p * np.log(p + 1e-10) + (1 - p) * np.log(1 - p + 1e-10))


def ppo_update(actor, critic, observations, actions, advantages, returns, old_probs):
    """
    Perform PPO update with separate backpropagation for actor and critic.
    Args:
        actor: Actor network parameters
        critic: Critic network parameters
        observations: shape (N x D)
        actions: shape (N,)
        advantages: shape (N,)
        returns: shape (N,)
        old_probs: shape (N,)
    """
    entropy_coefficient = 0.01  # Entropy regularization coefficient

    for epoch in range(epochs):
        # Shuffle the data
        idx = np.arange(len(observations))
        np.random.shuffle(idx)
        mb_obs = observations[idx].T  # (D x N)
        mb_actions = actions[idx]  # (N,)
        mb_advantages = advantages[idx]  # (N,)
        mb_returns = returns[idx]  # (N,)
        mb_old_probs = old_probs[idx]  # (N,)

        # Normalize advantages
        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
            mb_advantages.std() + 1e-8
        )

        # Update actor
        update_actor(
            actor, adam_cache["actor"], mb_obs, mb_actions, mb_advantages, mb_old_probs
        )

        # Update critic
        update_critic(critic, adam_cache["critic"], mb_obs, mb_returns)


def main():
    # Initialize environment
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    observation, _ = env.reset()
    prev_x = None

    # Initialize storage
    observations, actions, rewards, values, old_probs = [], [], [], [], []
    episode_number = 0
    reward_sum = 0
    running_reward = None
    start_time = time.time()  # Track total duration
    batch_start_time = time.time()  # Track batch duration

    while True:
        if render:
            env.render()

        # Preprocess observation
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        x = x.reshape(-1, 1)  # (D x 1)
        prev_x = cur_x

        # Forward pass through policy and value networks
        aprob, _ = policy_forward(actor, x)  # aprob: (1 x 1)
        v, _ = value_forward(critic, x)  # v: (1 x 1)
        p = aprob.squeeze()  # scalar
        action = 2 if np.random.uniform() < p else 3

        # Record data
        observations.append(x.flatten())
        actions.append(action)
        old_prob = p if action == 2 else 1 - p
        old_probs.append(old_prob)
        values.append(v.squeeze())

        # Step environment
        observation, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        reward_sum += reward

        if done or truncated:
            episode_number += 1

            # Convert lists to arrays
            observations_np = np.array(observations)  # (N x D)
            actions_np = np.array(actions)  # (N,)
            rewards_np = np.array(rewards)  # (N,)
            values_np = np.array(values + [0])  # (N+1,)
            old_probs_np = np.array(old_probs)  # (N,)

            # Compute discounted returns and advantages
            returns_np = discount_rewards(rewards_np)
            advantages_np = compute_advantages(rewards_np, values_np, gamma, lam)

            # Update policy and value networks
            ppo_update(
                actor,
                critic,
                observations_np,
                actions_np,
                advantages_np,
                returns_np,
                old_probs_np,
            )

            # Compute entropy for logging
            p_new, _ = policy_forward(actor, observations_np.T)
            entropy = compute_entropy(p_new.squeeze())

            # Log progress
            running_reward = (
                reward_sum
                if running_reward is None
                else running_reward * 0.99 + reward_sum * 0.01
            )
            episode_duration = time.time() - start_time
            logger.info(
                f"Episode {episode_number} completed. Reward total: {reward_sum}. Running mean: {running_reward:.2f}. Num actions: {len(rewards)}. Duration: {episode_duration:.2f} seconds. Entropy: {entropy:.4f}"
            )

            # Batch update logging
            if episode_number % batch_size == 0:
                batch_duration = time.time() - batch_start_time
                logger.info(
                    f"Batch update completed. Duration: {batch_duration:.2f} seconds"
                )
                batch_start_time = time.time()  # Reset batch start time

            # Checkpointing
            if episode_number % 500 == 0 or episode_number == 1:
                pickle.dump(
                    (actor, critic),
                    open(
                        os.path.join(folder_name, f"save_ppo_{episode_number}.p"), "wb"
                    ),
                )

            # Reset episode data
            observations, actions, rewards, values, old_probs = [], [], [], [], []
            reward_sum = 0
            observation, _ = env.reset()
            prev_x = None
            start_time = time.time()


if __name__ == "__main__":
    main()
