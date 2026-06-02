import numpy as np
import cv2
 
def undistortionMapping(K, D, raytraced_pts, width, height):
    num_pts = raytraced_pts.shape[1]
    rvec = np.zeros((3,1), dtype=float)
    tvec = np.zeros((3,1), dtype=float)    
    src = np.zeros((num_pts, 3), dtype=float)
    for idx in range(num_pts):
        src[idx, :] = raytraced_pts[:,idx]

    image_pts, _ = cv2.projectPoints(src, rvec, tvec, K, D)
    image_pts = image_pts.squeeze()

    mx = np.zeros(num_pts, dtype=np.float32)
    my = np.zeros(num_pts, dtype=np.float32)

    for idx in range(num_pts):
        mx[idx] = image_pts[idx][0]
        my[idx] = image_pts[idx][1]

    mapx = np.reshape(mx, (height, width))
    mapy = np.reshape(my, (height, width))

    return mapx, mapy

# Function to calculate the refracted ray 
# vi: incident ray
# normal: normal to the surface
# n1: refractive index of the medium where the incident ray is coming from
# n2: refractive index of the medium where the refracted ray is going to
def refractedRay(vi: np.array, normal: np.array, n1: float,n2: float):

    # Be sure it is a unit vector
    normal = normal/np.linalg.norm(normal) 
    tir = 0

    kk = n1**2*(np.dot(vi,normal))**2 - (n1**2-n2**2)*(np.dot(vi,vi))
    if(kk < 0):
        tir = 1
        a = 0
        b = 0
        vr = np.zeros(2)
        return np.array([vr, tir], dtype=float)
    
    a = n1/n2
    b = -(n1 * np.dot(vi,normal) + np.sqrt(kk))/n2
    vr = a*vi + b*normal

    return [vr, tir]

def rayTrace(ray0, normal, nglass, nwater, d0, d1, init_guess):

    v0 = ray0 / np.linalg.norm(ray0)
    norm_tmp = normal.copy() / np.linalg.norm(normal)

    pt = init_guess + d0 * v0 / np.dot(v0, norm_tmp)
    norm_tmp = -norm_tmp
    c = -np.dot(v0, norm_tmp)
    rglass = 1 / nglass
    rwater = 1 / nwater

    f1 = (1 - c**2)
    v1 = rglass * v0 + (c * rglass - np.sqrt(1 - f1 * (rglass**2))) * norm_tmp
    v2 = rwater * v0 + (rwater * c - np.sqrt(1 - f1 * (rwater**2))) * norm_tmp

    norm_tmp = -norm_tmp
    p0 = pt + d1 * v1 / np.dot(v1, norm_tmp)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    return v0, pt, v1, p0, v2

def optim_d0(init_guess: np.float64, K: np.array, normal, nglass: np.float64, nwater: np.float64, thickness: np.float64):
    d0 = init_guess[0]
    img_pts = np.zeros((3, 300), dtype=np.float64)

    # Flatten the grid and assign to img_pts
    for idx in range(20):
        for idy in range(15):
            index = idy * 20 + idx
            img_pts[0, index] = 50 * (idx+1)
            img_pts[1, index] = 50 * (idy+1)
            img_pts[2, index] = 1

    rays = np.linalg.inv(K) @ img_pts

    dmin = 9999.0
    dmax = -9999.0
    init_t = np.array([0.0, 0.0, 0.0], dtype=np.float64).T

    for i in range(300):
        _, _, _, p0, v2 = rayTrace(rays[:, i], normal, nglass, nwater, d0, thickness, init_t)        
        pom = np.abs(p0[2] - v2[2] * (p0[0] / (2 * v2[0]) + p0[1] / (2 * v2[1])))
        dmin = min(dmin, pom)
        dmax = max(dmax, pom)

    # Debug message
    #print(f"Status: dmin={dmin}, dmax={dmax}, virtual_d0={(dmax+dmin)/2}, deltax = {dmax-dmin}")
    init_guess[1] = (dmin+dmax)/2.0
    init_guess[2] = dmax - dmin

    return dmax - dmin

def solveForwardProjectionCase3(d, d2, normal, n1, n2, pt_v):

    m = np.array([0, 0, 1], dtype=float)

    # Find plane of refraction
    por = np.cross(normal, pt_v)
    por = por/np.linalg.norm(por)

    z1 = -normal.copy()
    z1 = z1/np.linalg.norm(z1)

    z2 = np.cross(por, z1)
    z2 = z2/np.linalg.norm(z2)

    v_p = np.dot(pt_v, z1)
    u_p = np.dot(pt_v, z2)
    pt_p = np.array([u_p, v_p], dtype=float)

    # Computing factors for solving 12th degree polynomial
    s1 = (n1**2 - 1)**2 * (n1**2 - 1)**2
    s2 = (-4)*u_p*(n1**2 - 1)**2*(n2**2 - 1)**2
    s3 = 4*u_p**2*(n1**2 - 1)**2*(n2**2 - 1)**2 + 2*(n1**2 - 1)*(n2**2 - 1)*((n2**2 - 1)*(u_p**2*(n1**2 - 1) + d**2*n1**2) - (n2**2 - 1)*(d - d2)**2 - (n1**2 - 1)*(d2 - v_p)**2 + d**2*n2**2*(n1**2 - 1))
    s4 = - 2*(n1**2 - 1)*(n2**2 - 1)*(2*d**2*n1**2*u_p*(n2**2 - 1) + 2*d**2*n2**2*u_p*(n1**2 - 1)) - 4*u_p*(n1**2 - 1)*(n2**2 - 1)*((n2**2 - 1)*(u_p**2*(n1**2 - 1) + d**2*n1**2) - (n2**2 - 1)*(d - d2)**2 - (n1**2 - 1)*(d2 - v_p)**2 + d**2*n2**2*(n1**2 - 1))
    s5 = ((n2**2 - 1)*(u_p**2*(n1**2 - 1) + d**2*n1**2) - (n2**2 - 1)*(d - d2)**2 - (n1**2 - 1)*(d2 - v_p)**2 + d**2*n2**2*(n1**2 - 1))**2 + 2*(n1**2 - 1)*(n2**2 - 1)*(d**2*n2**2*(u_p**2*(n1**2 - 1) + d**2*n1**2) - d**2*n2**2*(d - d2)**2 - d**2*n1**2*(d2 - v_p)**2 + d**2*n1**2*u_p**2*(n2**2 - 1)) - 4*(n1**2 - 1)*(n2**2 - 1)*(d - d2)**2*(d2 - v_p)**2 + 4*u_p*(n1**2 - 1)*(n2**2 - 1)*(2*d**2*n1**2*u_p*(n2**2 - 1) + 2*d**2*n2**2*u_p*(n1**2 - 1))
    s6 = -2*(2*d**2*n1**2*u_p*(n2**2 - 1) + 2*d**2*n2**2*u_p*(n1**2 - 1))*((n2**2 - 1)*(u_p**2*(n1**2 - 1) + d**2*n1**2) - (n2**2 - 1)*(d - d2)**2 - (n1**2 - 1)*(d2 - v_p)**2 + d**2*n2**2*(n1**2 - 1)) - 4*u_p*(n1**2 - 1)*(n2**2 - 1)*(d**2*n2**2*(u_p**2*(n1**2 - 1) + d**2*n1**2) - d**2*n2**2*(d - d2)**2 - d**2*n1**2*(d2 - v_p)**2 + d**2*n1**2*u_p**2*(n2**2 - 1)) - 4*d**4*n1**2*n2**2*u_p*(n1**2 - 1)*(n2**2 - 1)
    s7 = 2*((n2**2 - 1)*(u_p**2*(n1**2 - 1) + d**2*n1**2) - (n2**2 - 1)*(d - d2)**2 - (n1**2 - 1)*(d2 - v_p)**2 + d**2*n2**2*(n1**2 - 1))*(d**2*n2**2*(u_p**2*(n1**2 - 1) + d**2*n1**2) - d**2*n2**2*(d - d2)**2 - d**2*n1**2*(d2 - v_p)**2 + d**2*n1**2*u_p**2*(n2**2 - 1)) + (2*d**2*n1**2*u_p*(n2**2 - 1) + 2*d**2*n2**2*u_p*(n1**2 - 1))**2 - 4*(d - d2)**2*(d**2*n1**2*(n2**2 - 1) + d**2*n2**2*(n1**2 - 1))*(d2 - v_p)**2 + 10*d**4*n1**2*n2**2*u_p**2*(n1**2 - 1)*(n2**2 - 1)
    s8 = -2*(2*d**2*n1**2*u_p*(n2**2 - 1) + 2*d**2*n2**2*u_p*(n1**2 - 1))*(d**2*n2**2*(u_p**2*(n1**2 - 1) + d**2*n1**2) - d**2*n2**2*(d - d2)**2 - d**2*n1**2*(d2 - v_p)**2 + d**2*n1**2*u_p**2*(n2**2 - 1)) - 4*d**4*n1**2*n2**2*u_p*((n2**2 - 1)*(u_p**2*(n1**2 - 1) + d**2*n1**2) - (n2**2 - 1)*(d - d2)**2 - (n1**2 - 1)*(d2 - v_p)**2 + d**2*n2**2*(n1**2 - 1)) - 4*d**4*n1**2*n2**2*u_p**3*(n1**2 - 1)*(n2**2 - 1)
    s9 = (d**2*n2**2*(u_p**2*(n1**2 - 1) + d**2*n1**2) - d**2*n2**2*(d - d2)**2 - d**2*n1**2*(d2 - v_p)**2 + d**2*n1**2*u_p**2*(n2**2 - 1))**2 + 2*d**4*n1**2*n2**2*u_p**2*((n2**2 - 1)*(u_p**2*(n1**2 - 1) + d**2*n1**2) - (n2**2 - 1)*(d - d2)**2 - (n1**2 - 1)*(d2 - v_p)**2 + d**2*n2**2*(n1**2 - 1)) - 4*d**4*n1**2*n2**2*(d - d2)**2*(d2 - v_p)**2 + 4*d**4*n1**2*n2**2*u_p*(2*d**2*n1**2*u_p*(n2**2 - 1) + 2*d**2*n2**2*u_p*(n1**2 - 1))
    s10 = - 2*d**4*n1**2*n2**2*u_p**2*(2*d**2*n1**2*u_p*(n2**2 - 1) + 2*d**2*n2**2*u_p*(n1**2 - 1)) - 4*d**4*n1**2*n2**2*u_p*(d**2*n2**2*(u_p**2*(n1**2 - 1) + d**2*n1**2) - d**2*n2**2*(d - d2)**2 - d**2*n1**2*(d2 - v_p)**2 + d**2*n1**2*u_p**2*(n2**2 - 1))
    s11 = 4*d**8*n1**4*n2**4*u_p**2 + 2*d**4*n1**2*n2**2*u_p**2*(d**2*n2**2*(u_p**2*(n1**2 - 1) + d**2*n1**2) - d**2*n2**2*(d - d2)**2 - d**2*n1**2*(d2 - v_p)**2 + d**2*n1**2*u_p**2*(n2**2 - 1))
    s12 = (-4)*d**8*n1**4*n2**4*u_p**3    
    s13 = d**8*n1**4*n2**4*u_p**4

    coeffs = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13]
    sols = np.roots(coeffs)
    idx = np.where(np.abs(np.imag(sols)) < 1e-6)[0] # Don't consider complex solutions
    
    if len(idx) == 0:
        print("No real solution found")
        return

    sol1 = sols[idx].real
    sol1 = np.sort(sol1)
    nn = len(sol1)
    new_normal = np.array([0, -1], dtype=float)

    for i in range(nn):        
        x = sol1[i]
        vi = np.array([x, d], dtype=float)

        v2, _ = refractedRay(vi=vi, normal=new_normal, n1=1.0, n2=n1)
        q2 = vi + (d-d2) * v2 / np.dot(v2, new_normal) 

        v3, _ = refractedRay(vi=v2, normal=new_normal, n1=n1, n2=n2)
        vrd = pt_p - q2
        
        error = np.abs(vrd[0] * v3[1] - vrd[1] * v3[0])  
        if error < 1e-4:
            m = x * z2 + d * z1
            return m

    return m