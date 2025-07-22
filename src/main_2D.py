import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.animation import FuncAnimation, PillowWriter
from physics import CH_step, laplacian, source_term
from visualization import visualize_2D, create_figure, generate_rgb, update_imshow

plt.style.use('dark_background')

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
output_dir = os.path.join(parent_dir, "results/phase_conversion_constant.mp4")

np.random.seed(0)
N = 128
dirichlet_samples = np.random.dirichlet([1, 1, 1], size=(N, N))
c1, c2 = dirichlet_samples[..., 0], dirichlet_samples[..., 1]
c3 = 1.0 - c1 - c2
gamma, dt, steps = 3, 0.0025, 100
tf, k = 120, 0.1
save_every = 50

total_frames = steps // save_every

source_terms = {
    'A1': 0,
    'A2': lambda c3: 1e-2 * c3,
    'A3': lambda t: source_term(t, tf, k)
}

fig, ax, im, time_text = create_figure(N=8)

def update(frame):
    global c1, c2, c3
    for _ in range(save_every):
        A = source_terms['A2'](c3)
        c1, c2, c3 = CH_step(c1, c2, gamma, laplacian, dt, A)

    rgb_image = generate_rgb(c1, c2, c3)
    return update_imshow(im, time_text, rgb_image, frame * save_every * dt)

ani = FuncAnimation(fig, update, frames=tqdm(range(total_frames)), blit=True)
ani.save(output_dir, fps=40, dpi=300)
