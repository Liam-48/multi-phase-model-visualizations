import imageio
import os
import numpy as np
import re

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
output_dir = os.path.join(parent_dir, "results/frames_3d")

def frames_to_video(
    frame_folder,
    output_file,
    fps=350,
    min_repeat=1,
    max_repeat=20
):
    def extract_number(filename):
        match = re.search(r'frame_(\d+)', filename)
        return int(match.group(1)) if match else -1

    frame_files = sorted(
        [f for f in os.listdir(frame_folder) if f.endswith('.png')],
        key=extract_number
    )

    n_frames = len(frame_files)
    log_values = np.logspace(0.01, 1, n_frames, base=10)
    log_values = (log_values - log_values.min()) / (log_values.max() - log_values.min())
    repeats = (max_repeat - (max_repeat - min_repeat) * log_values).astype(int)

    with imageio.get_writer(output_file, fps=fps) as writer:
        for file, rep in zip(frame_files, repeats):
            image = imageio.imread(os.path.join(frame_folder, file))
            for _ in range(rep):
                writer.append_data(image)

frames_to_video(output_dir, os.path.join(parent_dir, "results/phase_separation_3D.mp4"))