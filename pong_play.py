import numpy as np
import pickle
import gymnasium as gym
import ale_py
import os
import time
import cv2


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

# Create folder to save video
video_folder = f"pong_inference_video_{int(time.time())}"
os.makedirs(video_folder, exist_ok=True)

# Initialize the environment
env = gym.make("ALE/Pong-v5", render_mode="rgb_array")

# Create video writer
video_path = os.path.join(video_folder, "pong_episode.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = 30
frame_size = (160, 210)  # Dynamically get frame size
observation, _ = env.reset()
video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

observation, _ = env.reset()
prev_x = None
reward_sum = 0

while True:  # Run until the episode finishes
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(80 * 80)
    prev_x = cur_x

    aprob = policy_forward(x, model)
    action = 2 if np.random.uniform() < aprob else 3

    observation, reward, done, _, _ = env.step(action)
    reward_sum += reward

    # Write the current frame to the video
    frame = env.render()
    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if done:
        print(f"Episode finished. Total reward: {reward_sum}")
        break

video_writer.release()
env.close()
print(f"Video saved at: {video_path}")
