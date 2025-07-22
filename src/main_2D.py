import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from physics import CH_step, laplacian_2D, source_term
from visualization import create_figure, generate_rgb, update_imshow

#Frames output settings
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
output_dir = os.path.join(parent_dir, "results/phase_conversion_constant.mp4")

#Background setting
plt.style.use('dark_background')

#Parameters and initial conditions
np.random.seed(0)
dirichlet_samples = np.random.dirichlet([1, 1, 1], size=(128, 128)) #Random initial condition on 128x128 grid
c1, c2 = dirichlet_samples[..., 0], dirichlet_samples[..., 1]
c3 = 1.0 - c1 - c2
gamma, dt, steps = 3, 0.0025, 100 #CH parameters
tf, k = 120, 0.1 #Source term parameters
save_every = 50 

total_frames = steps // save_every

#Source term examples
source_terms = {
    'A1': 0,
    'A2': lambda c3: 1e-2 * c3,
    'A3': lambda t, c3: source_term(t, tf, k) * c3
}

#Matplotlob figure setup
fig, ax, im, time_text = create_figure(N=8)

def update(frame):
    global c1, c2, c3
    for _ in range(save_every):
        A = source_terms['A2'](c3) #Main CH loop
        c1, c2, c3 = CH_step(c1, c2, gamma, laplacian_2D, dt, A) 

    rgb_image = generate_rgb(c1, c2, c3)
    return update_imshow(im, time_text, rgb_image, frame * save_every * dt)

ani = FuncAnimation(fig, update, frames=tqdm(range(total_frames)), blit=True) #Run animation
ani.save(output_dir, fps=40, dpi=300) #Save animation
