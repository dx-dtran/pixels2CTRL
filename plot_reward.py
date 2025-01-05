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

# Initialize lists for episodes, running means, and durations
episodes = []
running_means = []
durations = []

# Read and parse the log file
with open(log_file_path, "r") as file:
    for line in file:
        # Updated regex to match episode number, running mean, and duration
        match = re.search(
            r"Episode (\d+) done.*Running mean: (-?[\d]+(?:\.\d+)?).*Duration: ([\d\.]+)s",
            line,
        )
        if match:
            episodes.append(int(match.group(1)))
            running_means.append(float(match.group(2)))
            durations.append(float(match.group(3)))

# Check if data was extracted
if not episodes or not running_means or not durations:
    print("No valid data found in the log file.")
    exit()

# Create a figure with two subplots
plt.figure(figsize=(12, 10))

# Subplot 1: Reward Running Mean Over Time
plt.subplot(2, 1, 1)
plt.plot(episodes, running_means, linestyle="-", linewidth=0.8)
plt.title("Reward Running Mean Over Time")
plt.xlabel("Episode Number")
plt.ylabel("Reward Running Mean")
plt.grid(True)

# Subplot 2: Episode Duration Over Time
plt.subplot(2, 1, 2)
plt.plot(
    episodes,
    durations,
    linestyle="-",
    linewidth=0.8,
    color="orange",
)
plt.title("Episode Duration Over Time")
plt.xlabel("Episode Number")
plt.ylabel("Duration (s)")
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
