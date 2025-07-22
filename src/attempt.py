import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from tqdm import tqdm
from matplotlib.animation import FuncAnimation, PillowWriter

def laplacian(u):
    return (
        np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
        np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) -
        4 * u
    )

def W(t, tf, k):
    return 1/(1+np.exp(-k*(t-tf)))

def compute_rgb(c1, c2, c3):
    color1 = np.array(to_rgb("#FF4545"))
    color2 = np.array(to_rgb("#55F863"))
    color3 = np.array(to_rgb("#4545FF"))
    rgb_image = (
        c1[..., None] * color1 +
        c2[..., None] * color2 +
        c3[..., None] * color3
    )
    return np.clip(rgb_image, 0, 1)

np.random.seed(0)
N = 128
dirichlet_samples = np.random.dirichlet([1, 1, 1], size=(N, N))
c1, c2 = dirichlet_samples[..., 0], dirichlet_samples[..., 1]
gamma, dt, steps = 3, 0.0025, 10000
save_every = 50

fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
img = ax.imshow(np.zeros((N, N, 3)), origin="lower")
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, color='white',
                    fontsize=12, verticalalignment='top', bbox=dict(facecolor='black', alpha=0.4))
ax.axis('off')

frames = []

def update(frame_idx):
    global c1, c2
    for _ in range(save_every):
        t = frame_idx * save_every * dt
        c3 = 1 - c1 - c2
        c1_laplace = laplacian(c1)
        c2_laplace = laplacian(c2)
        c3_laplace = laplacian(c3)
        mu1 = 2 * c1 * (1 - c1) * (1 - 2 * c1) - gamma * c1_laplace
        mu2 = 2 * c2 * (1 - c2) * (1 - 2 * c2) - gamma * c2_laplace
        mu3 = 2 * c3 * (1 - c3) * (1 - 2 * c3) - gamma * c3_laplace
        c1 += dt * (2 * laplacian(mu1) - laplacian(mu2) - laplacian(mu3))
        c2 += dt * (2 * laplacian(mu2) - laplacian(mu1) - laplacian(mu3) + 1e-2 * c3)
    c3 = 1 - c1 - c2
    rgb = compute_rgb(c1, c2, c3)
    img.set_data(rgb)
    time_text.set_text(f't = {frame_idx * save_every * dt:.2f}')
    return [img, time_text]

total_frames = steps // save_every

ani = FuncAnimation(fig, update, frames=tqdm(range(total_frames)), blit=True)
ani.save("attempt.mp4", fps=40, dpi=300)