import numpy as np
import pickle
import gymnasium as gym
import ale_py
import os

gym.register_envs(ale_py)


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


def policy_forward(x, model):
    h = np.dot(model["W1"], x)
    h[h < 0] = 0  # ReLU nonlinearity
    logp = np.dot(model["W2"], h)
    p = sigmoid(logp)
    return p  # probability of taking action 2


# Load the saved model
model_path = "save_2025-01-04_10-47-08/save_4900.p"  # Update with your path and file
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Initialize the environment
env = gym.make("ALE/Pong-v5", render_mode="human")
observation, _ = env.reset()
prev_x = None

while True:
    # Preprocess the observation and compute the difference frame
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(80 * 80)
    prev_x = cur_x

    # Forward the policy network to get action probability
    aprob = policy_forward(x, model)
    action = 2 if np.random.uniform() < aprob else 3  # Sample an action

    # Step the environment
    observation, reward, done, _, _ = env.step(action)

    if done:
        print("Game over! Resetting environment.")
        observation, _ = env.reset()
        prev_x = None
