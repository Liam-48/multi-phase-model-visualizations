import numpy as np, pyvista as pv
from tqdm import tqdm
import os
from physics import CH_step, laplacian_3D, source_term
from visualization import visualize_3D

#Frames output settings
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
output_dir = os.path.join(parent_dir, "results/frames_3d")

#Background setting
pv.set_plot_theme('dark')

#Parameters and initial conditions
np.random.seed(0)
samples = np.random.dirichlet((1.0, 1.0, 1.0), size = 64 ** 3) #Random initial condition on 64x64x64 grid
samples = samples.reshape(64,64,64, 3)
c1 = samples[...,0]
c2 = samples[...,1]
c3 = 1.0 - c1 - c2 
dt = 0.001 #Time step
gamma = 0.1 #Gradient energy coefficient

#Main CH loop, creates frames inside the output folder
for i in tqdm(range(10001)):
    t = i * dt
    c1, c2, c3 = CH_step(c1, c2, gamma, laplacian_3D, dt, 0)
    
    if i % 10 == 0:
        visualize_3D(c1, c2, c3, i, False, output_dir, f't={i:05d}')