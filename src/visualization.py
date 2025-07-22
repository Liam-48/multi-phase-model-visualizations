import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import pyvista as pv

COLOR1 = np.array(to_rgb("#FF4545"))
COLOR2 = np.array(to_rgb("#55F863"))
COLOR3 = np.array(to_rgb("#4545FF"))

def generate_rgb(c1, c2, c3):
    """Return RGB blended image from three phases."""
    rgb = (
        c1[..., None] * COLOR1 +
        c2[..., None] * COLOR2 +
        c3[..., None] * COLOR3
    )
    return np.clip(rgb, 0, 1)

def create_figure(N=8):
    """Create figure, imshow, and time text for animation."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(N, N), facecolor='white')
    dummy = np.zeros((1, 1, 3))
    im = ax.imshow(dummy, origin="lower")
    time_text = ax.text(
        0.05, 0.95, '', transform=ax.transAxes,
        color='white', fontsize=15,
        verticalalignment='top',
        bbox=dict(facecolor='black', alpha=0.4)
    )
    ax.axis('off')
    return fig, ax, im, time_text

def update_imshow(im, time_text, rgb_image, time):
    im.set_array(rgb_image)
    time_text.set_text(f't = {time:.2f}')
    return [im, time_text]

def visualize_2D(c1, c2, c3, time, show=True, save_dir=None, filename='plot.png'):
    color1 = np.array(to_rgb("#FF4545"))
    color2 = np.array(to_rgb("#55F863"))
    color3 = np.array(to_rgb("#4545FF"))

    rgb_image = (
        c1[..., None] * color1 +
        c2[..., None] * color2 +
        c3[..., None] * color3
    )
    rgb_image = np.clip(rgb_image, 0, 1)
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    img = ax.imshow(rgb_image, origin="lower")
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, color='white',
                    fontsize=15, verticalalignment='top', bbox=dict(facecolor='black', alpha=0.4))
    time_text.set_text(f't = {time:.2f}')
    plt.axis('off')

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close()

def visualize_3D(c1, c2, c3, t, show=True, save_dir=None, filename='plot.png'):
    grid = pv.ImageData()
    grid.dimensions = np.array(c1.shape) + 1
    for i, c in enumerate([c1, c2, c3], 1):
        grid[f'c{i}'] = c.flatten('F')

    grid_points = grid.cell_data_to_point_data()

    iso_c1 = grid_points.contour([0.6], scalars='c1')
    iso_c2 = grid_points.contour([0.6], scalars='c2')
    iso_c3 = grid_points.contour([0.6], scalars='c3')

    iso_c1["colors"] = np.array([[81, 71, 255]] * iso_c1.n_points)
    iso_c2["colors"] = np.array([[130, 255, 209]] * iso_c2.n_points)
    iso_c3["colors"] = np.array([[255, 79, 94]] * iso_c3.n_points)

    iso = iso_c1 + iso_c2 + iso_c3
    iso = iso.smooth(n_iter=20, relaxation_factor=0.1)

    p = pv.Plotter(off_screen=True)
    title = p.add_text(f't={t}', font='courier', color='white', font_size=20, position=[40,680])
    actor = p.add_mesh(iso, scalars='colors', rgb=True)
    p.camera.position = (155.63850576500076, 155.63850576500076, 180)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        p.screenshot(save_path)

    if show:
        p.show()
    else:
        p.close()

    del grid, iso_c1, iso_c2, iso_c3, iso