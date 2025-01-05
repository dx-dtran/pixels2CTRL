import os
import re
import matplotlib.pyplot as plt

# Path to the folder containing the .log file
log_folder_path = "save_pong_ppo_2025-01-04_22-05-16"

# Find the first .log file in the folder
log_file_path = None
for file_name in os.listdir(log_folder_path):
    if file_name.endswith(".log"):
        log_file_path = os.path.join(log_folder_path, file_name)
        break

if not log_file_path:
    print("No .log file found in the specified folder.")
    exit()

# Initialize lists for episodes and running means
episodes = []
running_means = []

# Read and parse the log file
with open(log_file_path, "r") as file:
    for line in file:
        # Updated regex to match positive and negative numbers, including decimals
        match = re.search(
            r"Episode (\d+) done.*Running mean: (-?[\d]+(?:\.\d+)?)", line
        )
        if match:
            episodes.append(int(match.group(1)))
            running_means.append(float(match.group(2)))

# Check if data was extracted
if not episodes or not running_means:
    print("No valid data found in the log file.")
    exit()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(episodes, running_means, marker="o")
plt.title("Reward Running Mean Over Time")
plt.xlabel("Episode Number")
plt.ylabel("Reward Running Mean")
plt.grid(True)
plt.show()
