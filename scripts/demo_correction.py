import numpy as np
import pinax_utlis as pu
import cv2
import time
from tqdm import tqdm

def main():

    scaling_factor = 1.0
    original_width = 1280
    original_height = 960

    width = int(original_width * scaling_factor)
    height = int(original_height * scaling_factor)

    # Example of camera matrix (Ori)
    matrixK = np.array([[1403.882943803675700, 0, 616.238499749930610],
                        [0, 1407.482157830063400, 479.792602354303430],
                        [0, 0, 1]], dtype=float)
    
    matrixK = matrixK * scaling_factor
    matrixK[2,2] = 1

    distCoeffs= np.array([-0.273139587484678, 0.216093918966729, -0.000214253361360, -0.001770247078481, 0], dtype=float)


    # Flat port parameters
    d1=10; #glass thicknes
    d0=1.4282 #physical d0 distance
    d0virtual= np.array([0,0,0.5851], dtype=float) #virtual d0 distance
    d2=d0+d1

    # Refraction indexes
    ng=1.5 #glass refraction index
    nw=1.335; #water refraction index
    normal=np.array([0,0,1], dtype=float) #normal vector    

    img_pts = np.zeros((3,width*height), dtype=float)
    num_pts = width*height

    # Instead of for loop we can use numpy functions
    # Create a grid of indices
    u_v, v_v = np.meshgrid(np.arange(width), np.arange(height))

    # Flatten the grid and assign to img_pts
    img_pts[0, :] = u_v.flatten()
    img_pts[1, :] = v_v.flatten()
    img_pts[2, :] = 1

    rays = np.linalg.inv(matrixK) @ img_pts
    m = np.zeros((3,num_pts), dtype=float)

    dist = 5000 # 5 Meters
    pt_vs = dist * rays + d0virtual[:, np.newaxis]
    ''' Takes 03:35 minutes to run for 1280x960 image
    for idx in tqdm(range(num_pts)):
        pt_v = pt_vs[:,idx]
        m[:,idx] = pu.solveForwardProjectionCase3(d=d0, d2=d2, normal=-normal, n1=ng, n2=nw, pt_v=pt_v)
    ''' 

    # Doesn't work because there is no cuda implementation for finding the roots of the polynomial 
    #m = pu.solveForwardProjectionCase3Cuda(d=d0, d2=d2, normal=-normal, n1=ng, n2=nw, pt_vs=pt_vs)

    # It took 24.74 seconds to run for 1280x960 image
    start = time.time()
    m = pu.solveForwardProjectionCase3Hybrid(d=d0, d2=d2, normal=-normal, n1=ng, n2=nw, pt_vs=pt_vs)
    end = time.time()
    # Print only two decimal places
    print(f'Hybrid method took: {end - start:.2f} seconds')


    rvec = np.zeros((3,1), dtype=float)
    tvec = np.zeros((3,1), dtype=float)
    
    src = np.zeros((num_pts, 3), dtype=float)
    for idx in range(num_pts):
        src[idx, :] = m[:,idx]

    
    image_pts, _ = cv2.projectPoints(src, rvec, tvec, matrixK, distCoeffs)
    image_pts = image_pts.squeeze()

    mx = np.zeros(num_pts, dtype=np.float32)
    my = np.zeros(num_pts, dtype=np.float32)

    for idx in range(num_pts):
        mx[idx] = image_pts[idx][0]
        my[idx] = image_pts[idx][1]


    mapx = np.reshape(mx, (height, width))
    mapy = np.reshape(my, (height, width))

    # Create the undistorted image
    img = cv2.imread('testImg.jpg')
    print(f'Test image shape: {img.shape}')
    rez_img = cv2.resize(img, (int(original_width * scaling_factor), int(original_height * scaling_factor)))
    print(f'Resized image shape: {rez_img.shape}')
    undistorted_img = cv2.remap(rez_img, mapx, mapy, cv2.INTER_LINEAR)

    cv2.imshow('Undistorted Image', undistorted_img)
    cv2.waitKey(0)
    cv2.imshow('Original Image', rez_img)
    cv2.waitKey(0)

    return 0


if __name__ == "__main__":
    main()