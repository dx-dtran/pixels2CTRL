import numpy as np
import pickle
import gymnasium as gym
import ale_py
import os
import imageio


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


def play_and_record(env, model, gif_path):
    """Plays one episode using the given model and records it as a GIF."""
    fps = 30
    frames = []

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

        # Append the current frame to the frame list
        frame = env.render()
        frames.append(frame)

        if done:
            print(f"Episode finished. Total reward: {reward_sum}")
            break

    # Save the frames as a GIF
    imageio.mimsave(gif_path, frames, fps=fps)


# Main logic
weights_folder = (
    "save_pong_ppo_2025-01-04_22-05-16"  # Update this with your folder path
)
# Use the same folder as weights for saving GIFs
gif_output_folder = weights_folder

env = gym.make("ALE/Pong-v5", render_mode="rgb_array")

for file_name in os.listdir(weights_folder):
    if file_name.endswith(".p"):
        model_path = os.path.join(weights_folder, file_name)
        with open(model_path, "rb") as f:
            actor, critic = pickle.load(f)  # Unpack the tuple
            model = actor  # Use only the actor part for the existing methods

        gif_path = os.path.join(
            gif_output_folder, f"{os.path.splitext(file_name)[0]}.gif"
        )
        print(f"Processing {file_name}... Saving GIF to {gif_path}")
        play_and_record(env, model, gif_path)

env.close()
print(f"All GIFs saved in folder: {gif_output_folder}")
