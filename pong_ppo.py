import logging
import numpy as np
import pickle
import gymnasium as gym
import ale_py
import time
import os
from datetime import datetime

# -------------------------
# 1) Environment & Logging
# -------------------------

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

# -------------------------
# 2) Hyperparameters
# -------------------------
H = 200  # Number of hidden layer neurons
mini_batch_size = 1000  # Mini-batch size for PPO updates
epochs = 4  # Number of epochs per PPO update
learning_rate = 1e-3
gamma = 0.99  # Discount factor
lam = 0.95  # GAE lambda
clip_epsilon = 0.2  # PPO clipping parameter
resume = False  # Resume from previous checkpoint?
render = False  # Render environment?

# Entropy coefficient if you want to encourage exploration
ENTROPY_COEFF = 0.01

# -------------------------
# 3) Model Initialization
# -------------------------
D = 80 * 80  # Input dimensionality: 80x80 grid (after preprocessing)
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

# Adam optimizer parameters (actor & critic)
beta1, beta2 = 0.9, 0.999
epsilon = 1e-8
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


# -------------------------
# 4) Utility Functions
# -------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def prepro(I):
    """
    Preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector
    """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) set to 1
    return I.astype(float).ravel()


def discount_rewards(r):
    """
    Compute discounted rewards for an episode
    (Used if you are not doing full GAE, or if you want a baseline.)
    """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(r.size)):
        if r[t] != 0:
            running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(model, x):
    """
    Forward pass for the policy network.
    x: (D x N) input
    Returns:
      p: (1 x N) probability of action=2
      h: (H x N) hidden layer after ReLU
    """
    z1 = np.dot(model["W1"], x) + model["b1"]  # (H x N)
    h = np.maximum(z1, 0)  # ReLU
    logp = np.dot(model["W2"], h) + model["b2"]  # (1 x N)
    p = sigmoid(logp)  # (1 x N)
    return p, h


def value_forward(model, x):
    """
    Forward pass for the value network.
    x: (D x N) input
    Returns:
      v: (1 x N) value predictions
      h: (H x N) hidden layer
    """
    z1 = np.dot(model["W1"], x) + model["b1"]  # (H x N)
    h = np.maximum(z1, 0)  # ReLU
    v = np.dot(model["W2"], h) + model["b2"]  # (1 x N)
    return v, h


def compute_advantages(rewards, values, gamma, lam):
    """
    Compute GAE (Generalized Advantage Estimation).
    rewards: shape (N,)
    values: shape (N+1,) => note last value is for the terminal state
    """
    advantages = np.zeros_like(rewards)
    last_adv = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            delta = rewards[t] - values[t]
        else:
            delta = rewards[t] + gamma * values[t + 1] - values[t]
        advantages[t] = last_adv = delta + gamma * lam * last_adv
    return advantages


# -------------------------
# 5) PPO Actor Gradient
# -------------------------
def ppo_actor_grad_piecemeal(p, old_probs, advantages, actions, eps=0.2, ent_coeff=0.0):
    """
    Compute dL/d(logit) for each sample i for PPO's clipped objective + optional entropy.
    p: shape (N,) => current policy's prob of action=2
    old_probs: shape (N,) => old pi(a_i)
    advantages: shape (N,)
    actions: shape (N,) in {2,3}
    eps: clipping parameter
    ent_coeff: how much to weight the (negative) entropy derivative
    Returns:
      dL_dlogits: shape (N,)
    """
    N = len(p)

    # 1) ratio_i
    # pi_new[i] = p[i] if a=2 else (1-p[i])
    a2_mask = actions == 2
    pi_new = np.where(a2_mask, p, 1 - p)  # shape (N,)
    ratio = pi_new / old_probs  # shape (N,)

    # 2) surr1, surr2
    surr1 = ratio * advantages
    clipped_ratio = np.clip(ratio, 1.0 - eps, 1.0 + eps)
    surr2 = clipped_ratio * advantages

    # 3) pick min => piecewise derivative wrt ratio
    use_surr1 = surr1 < surr2  # boolean mask

    # Initialize derivative wrt ratio
    dL_dratio = np.zeros(N, dtype=np.float32)

    # (A) surr1 active => derivative is -(A_i)
    dL_dratio[use_surr1] = -advantages[use_surr1]

    # (B) surr2 active => derivative depends on if ratio is clipped or not
    not_use_surr1 = ~use_surr1
    ratio_clipped = (ratio < (1.0 - eps)) | (ratio > (1.0 + eps))
    # if it's inside [1-eps, 1+eps], derivative is -(A_i)
    inside_clip = not_use_surr1 & (~ratio_clipped)
    dL_dratio[inside_clip] = -advantages[inside_clip]
    # if it's outside => derivative is 0 (already is 0)

    # Average over batch
    dL_dratio /= N

    # 4) chain rule ratio => pi_new => p
    # ratio = pi_new / old_prob
    # => d(ratio)/d(pi_new) = 1/old_prob
    # pi_new = p if a=2 else (1-p)
    # => d(pi_new)/dp = +1 if a=2, else -1
    dL_dpi_new = dL_dratio / old_probs
    sign = np.where(a2_mask, 1.0, -1.0)
    dL_dp = dL_dpi_new * sign

    # 5) if we want to add an entropy term => derivative wrt p
    #   Entropy = - [ p log p + (1-p) log(1-p) ]
    #   so d(Entropy)/dp = - [ log p + 1 ] + [ log(1-p) + 1 ] = - log p + log(1-p)
    # => the derivative wrt p of ( -ent_coeff * Entropy ) is:
    #    dL_dp += -ent_coeff * d/dp(Entropy)
    if ent_coeff > 0.0:
        ent_grad = -(np.log(p + 1e-10) - np.log(1.0 - p + 1e-10))
        # multiply by ent_coeff / N so it is also averaged
        dL_dp += (ent_coeff / N) * ent_grad

    # 6) p = sigmoid(logit) => dp/d(logit) = p*(1-p)
    dL_dlogits = dL_dp * (p * (1.0 - p))

    return dL_dlogits


def update_actor(
    actor, adam_cache_actor, mb_obs, mb_actions, mb_advantages, mb_old_probs
):
    """
    Perform manual backpropagation + update for the actor network
    using the piecewise PPO derivative + optional entropy.
    """
    # 1) Forward pass => (1 x N)
    p, h_actor = policy_forward(actor, mb_obs)  # p: shape(1,N)
    p = p.squeeze(axis=0)  # shape(N,)
    N = mb_obs.shape[1]

    # 2) Compute piecewise gradients wrt logit
    dL_dlogits = ppo_actor_grad_piecemeal(
        p=p,
        old_probs=mb_old_probs,
        advantages=mb_advantages,
        actions=mb_actions,
        eps=clip_epsilon,
        ent_coeff=ENTROPY_COEFF,  # if you want an entropy bonus
    )

    # 3) Backprop through W2, b2
    # logits_i = W2 dot h_i + b2 => dlogits shape = (N,)
    # => dL/dW2 = sum_i [ dL/dlogits_i * h_i^T ]
    dL_dW2_actor = np.dot(dL_dlogits.reshape(1, N), h_actor.T) / N  # (1 x H)
    dL_db2_actor = np.sum(dL_dlogits, keepdims=True).reshape(1, 1) / N  # (1 x 1)

    # 4) backprop into hidden layer
    dL_dh_actor = np.dot(actor["W2"].T, dL_dlogits.reshape(1, -1))  # (H x N)
    # ReLU
    dL_dh_actor[h_actor <= 0] = 0

    # 5) W1, b1
    dL_dW1_actor = np.dot(dL_dh_actor, mb_obs.T) / N  # (H x D)
    dL_db1_actor = np.sum(dL_dh_actor, axis=1, keepdims=True) / N  # (H x 1)

    # 6) Adam updates
    for param, dparam, cache_m, cache_v in zip(
        ["W1", "b1", "W2", "b2"],
        [dL_dW1_actor, dL_db1_actor, dL_dW2_actor, dL_db2_actor],
        ["mW1", "mb1", "mW2", "mb2"],
        ["vW1", "vb1", "vW2", "vb2"],
    ):
        adam_cache_actor[cache_m] = (
            beta1 * adam_cache_actor[cache_m] + (1 - beta1) * dparam
        )
        adam_cache_actor[cache_v] = beta2 * adam_cache_actor[cache_v] + (1 - beta2) * (
            dparam**2
        )

        # NOTE: ideally use a global step t here for bias correction, not 'epochs'
        m_hat = adam_cache_actor[cache_m] / (1 - beta1**epochs)
        v_hat = adam_cache_actor[cache_v] / (1 - beta2**epochs)

        actor[param] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)


# -------------------------
# 6) Critic Grad & Update
# -------------------------
def compute_critic_gradients(critic, mb_obs, mb_returns):
    """
    Compute gradients for the critic network (standard MSE).
    """
    v, h_critic = value_forward(critic, mb_obs)  # (1 x N), (H x N)
    v = v.squeeze(axis=0)  # shape (N,)
    N = mb_obs.shape[1]

    # MSE loss = mean((v - R)^2)
    error = v - mb_returns
    dL_dv = 2 * error / N  # derivative wrt v

    # W2, b2
    dL_dW2 = np.dot(dL_dv.reshape(1, N), h_critic.T) / 1.0  # shape(1,H)
    dL_dW2 /= N  # average
    dL_db2 = np.sum(dL_dv, keepdims=True).reshape(1, 1) / N

    # backprop into hidden
    dL_dh = np.dot(critic["W2"].T, dL_dv.reshape(1, -1))  # (H x N)
    dL_dh[h_critic <= 0] = 0

    dL_dW1 = np.dot(dL_dh, mb_obs.T) / N  # (H x D)
    dL_db1 = np.sum(dL_dh, axis=1, keepdims=True) / N  # (H x 1)

    return {"W1": dL_dW1, "b1": dL_db1, "W2": dL_dW2, "b2": dL_db2}


def update_critic(critic, adam_cache_critic, mb_obs, mb_returns):
    """
    Perform manual backprop + update for the critic.
    """
    grads = compute_critic_gradients(critic, mb_obs, mb_returns)

    for param, dparam, cache_m, cache_v in zip(
        ["W1", "b1", "W2", "b2"],
        [grads["W1"], grads["b1"], grads["W2"], grads["b2"]],
        ["mW1", "mb1", "mW2", "mb2"],
        ["vW1", "vb1", "vW2", "vb2"],
    ):
        adam_cache_critic[cache_m] = (
            beta1 * adam_cache_critic[cache_m] + (1 - beta1) * dparam
        )
        adam_cache_critic[cache_v] = beta2 * adam_cache_critic[cache_v] + (
            1 - beta2
        ) * (dparam**2)

        m_hat = adam_cache_critic[cache_m] / (1 - beta1**epochs)
        v_hat = adam_cache_critic[cache_v] / (1 - beta2**epochs)

        critic[param] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)


# -------------------------
# 7) PPO Update Loop
# -------------------------
def compute_entropy(p):
    """
    Compute mean Bernoulli entropy of probabilities p (shape N,).
    """
    return -np.mean(p * np.log(p + 1e-10) + (1 - p) * np.log(1 - p + 1e-10))


def ppo_update(actor, critic, obs, actions, advantages, returns_, old_probs):
    """
    Perform PPO updates for a few epochs.
    obs: shape (N,D)
    actions: shape (N,)
    advantages: shape (N,)
    returns_: shape (N,)
    old_probs: shape (N,)
    """
    N = len(obs)
    for ep in range(epochs):
        idx = np.arange(N)
        np.random.shuffle(idx)
        mb_obs = obs[idx].T  # (D x N)
        mb_actions = actions[idx]  # (N,)
        mb_adv = advantages[idx]  # (N,)
        mb_ret = returns_[idx]  # (N,)
        mb_oldp = old_probs[idx]  # (N,)

        # normalize advantages
        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

        # Actor
        update_actor(actor, adam_cache["actor"], mb_obs, mb_actions, mb_adv, mb_oldp)
        # Critic
        update_critic(critic, adam_cache["critic"], mb_obs, mb_ret)


# -------------------------
# 8) Main Training Loop
# -------------------------
def main():
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    observation, _ = env.reset()
    prev_x = None

    # Storage
    observations, actions, rewards, values, old_probs = [], [], [], [], []
    episode_number = 0
    reward_sum = 0
    running_reward = None
    start_time = time.time()

    while True:
        if render:
            env.render()

        # Preprocess
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        x = x.reshape(-1, 1)  # (D,1)
        prev_x = cur_x

        # Forward pass => actor & critic
        aprob, _ = policy_forward(actor, x)  # (1 x 1)
        p = aprob.squeeze()  # scalar
        v, _ = value_forward(critic, x)
        action = 2 if np.random.uniform() < p else 3

        # Record
        observations.append(x.flatten())
        actions.append(action)
        old_prob = p if action == 2 else (1 - p)
        old_probs.append(old_prob)
        values.append(v.squeeze())

        # Step env
        observation, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        reward_sum += reward

        if done or truncated:
            episode_number += 1

            # Convert to np arrays
            observations_np = np.array(observations)  # (N x D)
            actions_np = np.array(actions)
            rewards_np = np.array(rewards)
            values_np = np.array(values + [0])  # (N+1,) => last=0 for terminal
            old_probs_np = np.array(old_probs)

            # Compute returns & advantages
            returns_np = discount_rewards(rewards_np)
            adv_np = compute_advantages(rewards_np, values_np, gamma, lam)

            # PPO update
            ppo_update(
                actor,
                critic,
                observations_np,
                actions_np,
                adv_np,
                returns_np,
                old_probs_np,
            )

            # Logging
            p_new, _ = policy_forward(actor, observations_np.T)  # shape(1,N)
            entropy_val = compute_entropy(p_new.squeeze())

            running_reward = (
                reward_sum
                if running_reward is None
                else 0.99 * running_reward + 0.01 * reward_sum
            )
            ep_dur = time.time() - start_time
            logger.info(
                f"Episode {episode_number} done. Reward: {reward_sum}. "
                f"Running mean: {running_reward:.2f}. Steps: {len(rewards)}. "
                f"Duration: {ep_dur:.2f}s. Entropy: {entropy_val:.4f}"
            )

            if episode_number % 100 == 0 or episode_number == 1:
                pickle.dump(
                    (actor, critic),
                    open(
                        os.path.join(folder_name, f"save_ppo_{episode_number}.p"), "wb"
                    ),
                )

            # Reset
            observations, actions, rewards, values, old_probs = [], [], [], [], []
            reward_sum = 0
            observation, _ = env.reset()
            prev_x = None
            start_time = time.time()


if __name__ == "__main__":
    main()
