import numpy as np
import cv2

import time
from tqdm import tqdm

import lib.pinax as pu
import lib.pinax_cuda as pcu
import lib.general_utlis as gu

def main():

    args = gu.parse_args('Flat Port Distortion Correction')
    cfg = gu.read_cfg(args.cfg)

    scaling_factor = cfg['scale']
    original_width = cfg['width']
    original_height = cfg['height']

    width = int(original_width * scaling_factor)
    height = int(original_height * scaling_factor)

    if cfg['compute_map'] is True:
        # Example of camera matrix (Ori)
        matrixK = np.array(cfg['K_matrix'], dtype=np.float64)
        matrixK = matrixK.reshape(3,3)
        matrixK = matrixK * scaling_factor
        matrixK[2,2] = 1
        distCoeffs= np.array(cfg['D_matrix'], dtype=np.float64)


        # Flat port parameters
        d1 = cfg['glass_thickness']
        d0 = cfg['distance_to_glass']
        d0virtual = np.array([0.0, 0.0, cfg['virtual_distance']], dtype=float) #virtual d0 distance
        d2 = d0 + d1

        # Refraction indexes
        ng=cfg['glass_refractive_index']
        nw=cfg['water_refractive_index']
        normal=np.array(cfg['normal_vector'], dtype=float) #normal vector    

        img_pts = np.zeros((3,width*height), dtype=float)
        num_pts = width*height

        # Instead of for loop we can use numpy functions
        # Create a grid of indices
        u_v, v_v = np.meshgrid(np.arange(width), np.arange(height))

        # Flatten the grid and assign to img_pts
        img_pts[0, :] = u_v.flatten()
        img_pts[1, :] = v_v.flatten()
        img_pts[2, :] = 1.0

        rays = np.linalg.inv(matrixK) @ img_pts
        m = np.zeros((3,num_pts), dtype=float)

        dist = cfg['projection_distance'] 
        pt_vs = dist * rays + d0virtual[:, np.newaxis]

        if cfg['cuda']:
            start = time.time()
            m = pcu.solveForwardProjectionCase3Hybrid(d=d0, d2=d2, normal=-normal, n1=ng, n2=nw, pt_vs=pt_vs)
            end = time.time()
            print(f'Hybrid method took: {end - start:.2f} seconds')
        else:
            # Takes 03:35 minutes to run for 1280x960 image
            for idx in tqdm(range(num_pts)):
                pt_v = pt_vs[:,idx]
                m[:,idx] = pu.solveForwardProjectionCase3(d=d0, d2=d2, normal=-normal, n1=ng, n2=nw, pt_v=pt_v) 

        # Undistortion mapping
        mapx, mapy = pu.undistortionMapping(K=matrixK, D=distCoeffs, raytraced_pts=m, 
                                            width=width, height=height)

        if cfg['save_mapping']:
            map = np.stack((mapx, mapy), axis=-1)
            np.save(cfg['save_path'], map)
    else:
        map = np.load(cfg['map_path'])
        mapx = map[:,:,0]
        mapy = map[:,:,1]

    # Create the undistorted image
    img = cv2.imread(cfg['path'])
    rez_img = cv2.resize(img, (int(original_width * scaling_factor), int(original_height * scaling_factor)))
    
    # Undistortion using the correction map
    undistorted_img = cv2.remap(rez_img, mapx, mapy, cv2.INTER_LINEAR)

    cv2.imshow('Undistorted Image', undistorted_img)
    cv2.waitKey(0)
    cv2.imshow('Original Image', rez_img)
    cv2.waitKey(0)

    return 0


if __name__ == "__main__":
    main()