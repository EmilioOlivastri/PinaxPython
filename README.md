# Pinax Camera Model

Python implementation of the [code](https://github.com/tomluc/Pinax-camera-model) from the paper on the [pinax](https://www.sciencedirect.com/science/article/pii/S0029801817300434) camera model.

## Structure
```
scripts/lib/pinax.py
scripts/lib/pinax_cuda.py 
```
It contains all the logic for modeling refraction, remove distortion and compute the optimal distance from the interface, both using CPU and CUDA strategies.

```
scripts/d0_estimation.py 
```
Estimates the optimal distance at which you should put the camera from the interface in order to mininmize the length of the caustic. To compute this values some parameters it is need to estimate the following parameters:
- camera matrix K ; <- obtained with in air calibration
- distortion vector D; <- obtained with in air calibration
- glass and water refractive indices; <- known
- glass thickness; <- known
- normal from camera to glass interface; <- obtained with in water calibration
- distance from glass to camera projection plane; <- obtained with in water calibration

In case it is possible to mechanically adjust the camera, then put it at the obtained distance to minimize the refraction effects. It prints the caustic length both at the actual distance and at the optimal distance.
```
scripts/correction.py
```
Estimates the undistortion map using the pinax camera model, given the camera and flat port parameters. It requires the same parameters as above, plus some extra parameters:
- width of the image;
- height of the image
- scale; <- Computing the undistortion map can take a while at full resolution
- image path;
- save_path; <- path at which to save the undistortion map
- save_mapping; <- if true saves current undistortion map at save_path
- cuda; <- if uses cuda to compute the undistortion map (a lot faster)
- compute_map; <- if true computes the undistortion map with the current parameters
- map_path; <- undistortion map path

It also shows the difference between the original image and the undistorted image. In this way in case can be decided if you are satisfied with the undistortion or not.

# How to use
Example of using d0_estimation.py

```
python scripts/d0_estimation.py --cfg config/camera_flatport_D0.yaml
```

Example of using correction.py

```
python scripts/correction.py --cfg config/camera_flatport_Correction.yaml
```
