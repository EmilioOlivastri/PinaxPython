# Pinax Camera Model

Python implementation from the paper on the [pinax](https://www.sciencedirect.com/science/article/pii/S0029801817300434) camera model.

## Structure
```
scripts/lib/pinax.py
scripts/lib/pinax_cuda.py 
```
It contains all the logic for modeling refraction, remove distortion and compute the optimal distance from the interface

```
scripts/d0_estimation.py 
```
Estimates the optimal distance from the interface and given the current distance estimates how long is the caustic

```
scripts/correction.py
```
Given camera, flat port, and refraction indices parameters it removes the distortion from the input image.

# How to use
Example of using d0_estimation.py

```
python scripts/demo_correction.py --cfg config/camera_flatport_D0.yaml
```

Example of using correction.py

```
python scripts/correction.py --cfg config/camera_flatport_Correction.yaml
```