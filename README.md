# Three-Phase Cahn–Hilliard Simulation and Visualization
This repository contains a Python implementation of the three-phase Cahn–Hilliard model with 2D and 3D visualizations, including a script to generate a video animation for both simulations. This code was used to analyze how a source term affects the normal Cahn–Hilliard equation in both domains.

## About the project
This project focuses on
- **Solving** the three-phase Cahn–Hilliard modified equation (added a Source term) in 2D and 3D, with given parameters.
- **Visualizing** the phase evolution over time, using:
    - **2D:** Real-time animation generated on the fly.
    - **3D:** Saves frame to a folder, then stitches them into a video.

## How to use it

**2D example**

Run: 
```bash
python src/main_2D.py
```
An `.mp4` animation file will be saved to the `results` folder.

**3D example**

Run: 
```bash
python src/main_3D.py
```
to generate frames inside the `results/frames_3d` folder. Feel free to modify `main_3D.py` to adjust frame rate, video duration, etc.

To stitch these frames into a video, run:
```bash
python src/video_script.py
```
This will create an `.mp4` file inside the `results` folder.

**Parameters**

If you want to experiment with other parameters (gradient energy coefficient, source term, grid size, etc.), you can change them inside the `main_2d.py` or `main_3d.py` file, specifically in the CH loop. See `src/physics.py` to check how the functions used are defined.

##
**Note:** This solver and visualization code were written during my internship at CICY (Centro de Investigación Científica de Yucatán).
