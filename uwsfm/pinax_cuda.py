from numba import cuda
import math
import numpy as np
#import torch

# Cuda basic operations
@cuda.jit(device=True)
def zeros2(z):
    z[0] = 0.0; z[1] = 0.0
    return 

@cuda.jit(device=True)
def zeros3(z):
    z[0] = 0.0; z[1] = 0.0; z[2] = 0.0
    return 

@cuda.jit(device=True)
def cross_product(a, b, result):
    result[0] = a[1] * b[2] - a[2] * b[1]
    result[1] = a[2] * b[0] - a[0] * b[2]
    result[2] = a[0] * b[1] - a[1] * b[0]

@cuda.jit(device=True)
def norm3(v):
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

@cuda.jit(device=True)
def norm2(v):
    return math.sqrt(v[0]**2 + v[1]**2)

@cuda.jit(device=True)
def dot3(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

@cuda.jit(device=True)
def dot2(a, b):
    return a[0] * b[0] + a[1] * b[1]



@cuda.jit(device=True)
def refractedRayCUDA(vi, normal, n1, n2, vr):

    # Be sure it is a unit vector
    norm = norm2(normal)
    normal[0] /= norm
    normal[1] /= norm

    n1 = np.float64(n1)
    n2 = np.float64(n2)

    kk = n1**2*(dot2(vi,normal))**2 - (n1**2-n2**2)*(dot2(vi,vi))
    if(kk < 0):
        zeros2(vr)
    else: 
        a = n1/n2
        b = -(n1 * dot2(vi,normal) + np.float64(math.sqrt(kk)))/n2
        for i in range(2):
            vr[i] = a * vi[i] + b * normal[i]
    return

@cuda.jit(device=True)
def poly_eval(coeff, x):
    # Evaluate f(x) = coeff[0]*x^12 + coeff[1]*x^11 + ... + coeff[12]
    f = coeff[0]
    for i in range(1, 13):
        f = f * x + coeff[i]
    return f

@cuda.jit(device=True)
def poly_eval_deriv(coeff, x):
    # Evaluate the derivative f'(x) for f of degree 12
    fprime = 12.0 * coeff[0]
    for i in range(1, 12):
        fprime = fprime * x + (12.0 - i) * coeff[i]
    return fprime

@cuda.jit(device=True)
def newton_solve(coeff, x0, max_iter, tol):
    x = x0
    for it in range(max_iter):
        f_val = poly_eval(coeff, x)
        fprime_val = poly_eval_deriv(coeff, x)
        if math.fabs(fprime_val) < 1e-6:
            break
        dx = f_val / fprime_val
        x_new = x - dx
        if math.fabs(dx) < tol:
            return x_new
        x = x_new
    return x

@cuda.jit(device=True)
def solve_polynomial(coeff, best_x_out):
    """
    Try several initial guesses in [x_min, x_max] and return the four candidate
    solutions with the smallest |f(x)| values.
    best_x_out is an output array of length 4.
    Returns 1 if at least one candidate is found.
    """
    NGUESSES = 100
    x_min = -5.0
    x_max = 5.0

    # Local arrays to hold the best four errors and corresponding candidate solutions.
    best_err = cuda.local.array(4, dtype=np.float64)
    best_x = cuda.local.array(4, dtype=np.float64)
    for j in range(4):
        best_err[j] = 1e30  # A very large number
        best_x[j] = 0.0

    # Loop over initial guesses
    for i in range(NGUESSES):
        x0 = x_min + i * (x_max - x_min) / (NGUESSES - 1)
        x_candidate = newton_solve(coeff, x0, 50, 1e-5)
        err = math.fabs(poly_eval(coeff, x_candidate))
        # Insert candidate into the sorted list of best errors
        for j in range(4):
            if err < best_err[j]:
                # Shift down the current candidates starting at position j
                for k in range(3, j, -1):
                    best_err[k] = best_err[k-1]
                    best_x[k] = best_x[k-1]
                best_err[j] = err
                best_x[j] = x_candidate
                break

    # Copy the four best candidate solutions into the output array
    for j in range(4):
        best_x_out[j] = best_x[j]

    return True

@cuda.jit
def solveForwardProjectionCase3Kernel(d, d2, normal, n1, n2, pt_vs, m):
    idx = cuda.grid(1)
    if idx < m.shape[1]:
        pt_v = pt_vs[:, idx]

        # Manually compute the cross product
        por = cuda.local.array((3,), dtype=np.float64)
        zeros3(por)
        cross_product(normal, pt_v, por)
        por_norm = norm3(por)
        por[0] /= por_norm
        por[1] /= por_norm
        por[2] /= por_norm

        z1 = cuda.local.array((3,), dtype=np.float64)
        for i in range(3):
            z1[i] = normal[i]

        z1_norm = -1 * norm3(z1)
        z1[0] /= z1_norm
        z1[1] /= z1_norm
        z1[2] /= z1_norm

        # Manually compute the cross product
        z2 = cuda.local.array((3,), dtype=np.float64)
        cross_product(por, z1, z2)
        z2_norm = norm3(z1)
        z2[0] /= z2_norm
        z2[1] /= z2_norm
        z2[2] /= z2_norm

        v_p = dot3(pt_v, z1)
        u_p = dot3(pt_v, z2)
        pt_p = cuda.local.array((2,), dtype=np.float64)
        pt_p[0] = u_p; pt_p[1] = v_p

        # Computing factors for solving 12th degree polynomial
        coeff = cuda.local.array(13, dtype=np.float64)
        s1 = (n1**2 - 1)**2 * (n1**2 - 1)**2
        s2 = (-4) * u_p * (n1**2 - 1)**2 * (n2**2 - 1)**2
        s3 = 4 * u_p**2 * (n1**2 - 1)**2 * (n2**2 - 1)**2 + 2 * (n1**2 - 1) * (n2**2 - 1) * ((n2**2 - 1) * (u_p**2 * (n1**2 - 1) + d**2 * n1**2) - (n2**2 - 1) * (d - d2)**2 - (n1**2 - 1) * (d2 - v_p)**2 + d**2 * n2**2 * (n1**2 - 1))
        s4 = -2 * (n1**2 - 1) * (n2**2 - 1) * (2 * d**2 * n1**2 * u_p * (n2**2 - 1) + 2 * d**2 * n2**2 * u_p * (n1**2 - 1)) - 4 * u_p * (n1**2 - 1) * (n2**2 - 1) * ((n2**2 - 1) * (u_p**2 * (n1**2 - 1) + d**2 * n1**2) - (n2**2 - 1) * (d - d2)**2 - (n1**2 - 1) * (d2 - v_p)**2 + d**2 * n2**2 * (n1**2 - 1))
        s5 = ((n2**2 - 1) * (u_p**2 * (n1**2 - 1) + d**2 * n1**2) - (n2**2 - 1) * (d - d2)**2 - (n1**2 - 1) * (d2 - v_p)**2 + d**2 * n2**2 * (n1**2 - 1))**2 + 2 * (n1**2 - 1) * (n2**2 - 1) * (d**2 * n2**2 * (u_p**2 * (n1**2 - 1) + d**2 * n1**2) - d**2 * n2**2 * (d - d2)**2 - d**2 * n1**2 * (d2 - v_p)**2 + d**2 * n1**2 * u_p**2 * (n2**2 - 1)) - 4 * (n1**2 - 1) * (n2**2 - 1) * (d - d2)**2 * (d2 - v_p)**2 + 4 * u_p * (n1**2 - 1) * (n2**2 - 1) * (2 * d**2 * n1**2 * u_p * (n2**2 - 1) + 2 * d**2 * n2**2 * u_p * (n1**2 - 1))
        s6 = -2 * (2 * d**2 * n1**2 * u_p * (n2**2 - 1) + 2 * d**2 * n2**2 * u_p * (n1**2 - 1)) * ((n2**2 - 1) * (u_p**2 * (n1**2 - 1) + d**2 * n1**2) - (n2**2 - 1) * (d - d2)**2 - (n1**2 - 1) * (d2 - v_p)**2 + d**2 * n2**2 * (n1**2 - 1)) - 4 * u_p * (n1**2 - 1) * (n2**2 - 1) * (d**2 * n2**2 * (u_p**2 * (n1**2 - 1) + d**2 * n1**2) - d**2 * n2**2 * (d - d2)**2 - d**2 * n1**2 * (d2 - v_p)**2 + d**2 * n1**2 * u_p**2 * (n2**2 - 1)) - 4 * d**4 * n1**2 * n2**2 * u_p * (n1**2 - 1) * (n2**2 - 1)
        s7 = 2 * ((n2**2 - 1) * (u_p**2 * (n1**2 - 1) + d**2 * n1**2) - (n2**2 - 1) * (d - d2)**2 - (n1**2 - 1) * (d2 - v_p)**2 + d**2 * n2**2 * (n1**2 - 1)) * (d**2 * n2**2 * (u_p**2 * (n1**2 - 1) + d**2 * n1**2) - d**2 * n2**2 * (d - d2)**2 - d**2 * n1**2 * (d2 - v_p)**2 + d**2 * n1**2 * u_p**2 * (n2**2 - 1)) + (2 * d**2 * n1**2 * u_p * (n2**2 - 1) + 2 * d**2 * n2**2 * u_p * (n1**2 - 1))**2 - 4 * (d - d2)**2 * (d**2 * n1**2 * (n2**2 - 1) + d**2 * n2**2 * (n1**2 - 1)) * (d2 - v_p)**2 + 10 * d**4 * n1**2 * n2**2 * u_p**2 * (n1**2 - 1) * (n2**2 - 1)
        s8 = -2 * (2 * d**2 * n1**2 * u_p * (n2**2 - 1) + 2 * d**2 * n2**2 * u_p * (n1**2 - 1)) * (d**2 * n2**2 * (u_p**2 * (n1**2 - 1) + d**2 * n1**2) - d**2 * n2**2 * (d - d2)**2 - d**2 * n1**2 * (d2 - v_p)**2 + d**2 * n1**2 * u_p**2 * (n2**2 - 1)) - 4 * d**4 * n1**2 * n2**2 * u_p * ((n2**2 - 1) * (u_p**2 * (n1**2 - 1) + d**2 * n1**2) - (n2**2 - 1) * (d - d2)**2 - (n1**2 - 1) * (d2 - v_p)**2 + d**2 * n2**2 * (n1**2 - 1)) - 4 * d**4 * n1**2 * n2**2 * u_p**3 * (n1**2 - 1) * (n2**2 - 1)
        s9 = (d**2 * n2**2 * (u_p**2 * (n1**2 - 1) + d**2 * n1**2) - d**2 * n2**2 * (d - d2)**2 - d**2 * n1**2 * (d2 - v_p)**2 + d**2 * n1**2 * u_p**2 * (n2**2 - 1))**2 + 2 * d**4 * n1**2 * n2**2 * u_p**2 * ((n2**2 - 1) * (u_p**2 * (n1**2 - 1) + d**2 * n1**2) - (n2**2 - 1) * (d - d2)**2 - (n1**2 - 1) * (d2 - v_p)**2 + d**2 * n2**2 * (n1**2 - 1)) - 4 * d**4 * n1**2 * n2**2 * (d - d2)**2 * (d2 - v_p)**2 + 4 * d**4 * n1**2 * n2**2 * u_p * (2 * d**2 * n1**2 * u_p * (n2**2 - 1) + 2 * d**2 * n2**2 * u_p * (n1**2 - 1))
        s10 = -2 * d**4 * n1**2 * n2**2 * u_p**2 * (2 * d**2 * n1**2 * u_p * (n2**2 - 1) + 2 * d**2 * n2**2 * u_p * (n1**2 - 1)) - 4 * d**4 * n1**2 * n2**2 * u_p * (d**2 * n2**2 * (u_p**2 * (n1**2 - 1) + d**2 * n1**2) - d**2 * n2**2 * (d - d2)**2 - d**2 * n1**2 * (d2 - v_p)**2 + d**2 * n1**2 * u_p**2 * (n2**2 - 1))
        s11 = 4 * d**8 * n1**4 * n2**4 * u_p**2 + 2 * d**4 * n1**2 * n2**2 * u_p**2 * (d**2 * n2**2 * (u_p**2 * (n1**2 - 1) + d**2 * n1**2) - d**2 * n2**2 * (d - d2)**2 - d**2 * n1**2 * (d2 - v_p)**2 + d**2 * n1**2 * u_p**2 * (n2**2 - 1))
        s12 = (-4) * d**8 * n1**4 * n2**4 * u_p**3
        s13 = d**8 * n1**4 * n2**4 * u_p**4

        coeff[0] = s1; coeff[1] = s2 ;coeff[2] = s3; coeff[3] = s4
        coeff[4] = s5; coeff[5] = s6; coeff[6] = s7; coeff[7] = s8
        coeff[8] = s9; coeff[9] = s10; coeff[10] = s11; coeff[11] = s12
        coeff[12] = s13

        # Solve the polynomial using our custom solver
        x_sol = cuda.local.array(4, dtype=np.float64)
        ret = solve_polynomial(coeff, x_sol)
        if not ret:
            print("No real solution found")
            return
        
        new_normal = cuda.local.array((2,), dtype=np.float64)
        new_normal[0] = 0.0; new_normal[1] = -1.0
        
        for i in range(12):
            x = x_sol[i]
            vi = cuda.local.array((2,), dtype=np.float64)
            vi[0] = x; vi[1] = d
            
            # Inline the logic of refractedRay
            v2= cuda.local.array((2,), dtype=np.float64)
            refractedRayCUDA(vi, new_normal, 1.0, n1, v2)

            q2 = cuda.local.array((2,), dtype=np.float64)
            dot_vr_new_normal = dot2(v2, new_normal)
            for i in range(2):
                q2[i] = vi[i] + (d - d2) * v2[i] / dot_vr_new_normal

            # Inline the logic of refractedRay again
            v3 = cuda.local.array((2,), dtype=np.float64)
            refractedRayCUDA(v2, new_normal, n1, n2, v3)
            
            vrd = cuda.local.array((2,), dtype=np.float64)
            for i in range(2):
                vrd[i] = pt_p[i] - q2[i]

            error = math.fabs(vrd[0] * v3[1] - vrd[1] * v3[0])
            # With this:
            if error < 1e-4:
                for i in range(3):
                    m[i, idx] = x * z2[i] + d * z1[i]
                return
            

## Hybrid method ## 

@cuda.jit
def compute_poly_coeffs_kernel(d, d2, normal, n1, n2, pt_vs, coeffs_out):
    # Cuda
    idx = cuda.grid(1)
    num_pts = pt_vs.shape[1]
    if idx < num_pts:
        # Normal code execution
        # Load pt_v (3D vector) for this index.
        pt_v = cuda.local.array((3,), dtype=np.float64)
        for i in range(3):
            pt_v[i] = pt_vs[i, idx]
        
        # Compute coordinate system as in your original kernel:
        por = cuda.local.array((3,), dtype=np.float64)
        cross_product(normal, pt_v, por)
        por_norm = norm3(por)
        por[0] /= por_norm
        por[1] /= por_norm
        por[2] /= por_norm
        
        z1 = cuda.local.array((3,), dtype=np.float64)
        for i in range(3):
            z1[i] = normal[i]

        z1_norm = -1 * norm3(z1)
        z1[0] /= z1_norm
        z1[1] /= z1_norm
        z1[2] /= z1_norm

        # Manually compute the cross product
        z2 = cuda.local.array((3,), dtype=np.float64)
        cross_product(por, z1, z2)
        z2_norm = norm3(z1)
        z2[0] /= z2_norm
        z2[1] /= z2_norm
        z2[2] /= z2_norm

        v_p = dot3(pt_v, z1)
        u_p = dot3(pt_v, z2)
        pt_p = cuda.local.array((2,), dtype=np.float64)
        pt_p[0] = u_p; pt_p[1] = v_p

        s1 = (n1**2 - 1)**2 * (n1**2 - 1)**2
        s2 = (-4) * u_p * (n1**2 - 1)**2 * (n2**2 - 1)**2
        s3 = 4 * u_p**2 * (n1**2 - 1)**2 * (n2**2 - 1)**2 + 2 * (n1**2 - 1) * (n2**2 - 1) * ((n2**2 - 1) * (u_p**2 * (n1**2 - 1) + d**2 * n1**2) - (n2**2 - 1) * (d - d2)**2 - (n1**2 - 1) * (d2 - v_p)**2 + d**2 * n2**2 * (n1**2 - 1))
        s4 = -2 * (n1**2 - 1) * (n2**2 - 1) * (2 * d**2 * n1**2 * u_p * (n2**2 - 1) + 2 * d**2 * n2**2 * u_p * (n1**2 - 1)) - 4 * u_p * (n1**2 - 1) * (n2**2 - 1) * ((n2**2 - 1) * (u_p**2 * (n1**2 - 1) + d**2 * n1**2) - (n2**2 - 1) * (d - d2)**2 - (n1**2 - 1) * (d2 - v_p)**2 + d**2 * n2**2 * (n1**2 - 1))
        s5 = ((n2**2 - 1) * (u_p**2 * (n1**2 - 1) + d**2 * n1**2) - (n2**2 - 1) * (d - d2)**2 - (n1**2 - 1) * (d2 - v_p)**2 + d**2 * n2**2 * (n1**2 - 1))**2 + 2 * (n1**2 - 1) * (n2**2 - 1) * (d**2 * n2**2 * (u_p**2 * (n1**2 - 1) + d**2 * n1**2) - d**2 * n2**2 * (d - d2)**2 - d**2 * n1**2 * (d2 - v_p)**2 + d**2 * n1**2 * u_p**2 * (n2**2 - 1)) - 4 * (n1**2 - 1) * (n2**2 - 1) * (d - d2)**2 * (d2 - v_p)**2 + 4 * u_p * (n1**2 - 1) * (n2**2 - 1) * (2 * d**2 * n1**2 * u_p * (n2**2 - 1) + 2 * d**2 * n2**2 * u_p * (n1**2 - 1))
        s6 = -2 * (2 * d**2 * n1**2 * u_p * (n2**2 - 1) + 2 * d**2 * n2**2 * u_p * (n1**2 - 1)) * ((n2**2 - 1) * (u_p**2 * (n1**2 - 1) + d**2 * n1**2) - (n2**2 - 1) * (d - d2)**2 - (n1**2 - 1) * (d2 - v_p)**2 + d**2 * n2**2 * (n1**2 - 1)) - 4 * u_p * (n1**2 - 1) * (n2**2 - 1) * (d**2 * n2**2 * (u_p**2 * (n1**2 - 1) + d**2 * n1**2) - d**2 * n2**2 * (d - d2)**2 - d**2 * n1**2 * (d2 - v_p)**2 + d**2 * n1**2 * u_p**2 * (n2**2 - 1)) - 4 * d**4 * n1**2 * n2**2 * u_p * (n1**2 - 1) * (n2**2 - 1)
        s7 = 2 * ((n2**2 - 1) * (u_p**2 * (n1**2 - 1) + d**2 * n1**2) - (n2**2 - 1) * (d - d2)**2 - (n1**2 - 1) * (d2 - v_p)**2 + d**2 * n2**2 * (n1**2 - 1)) * (d**2 * n2**2 * (u_p**2 * (n1**2 - 1) + d**2 * n1**2) - d**2 * n2**2 * (d - d2)**2 - d**2 * n1**2 * (d2 - v_p)**2 + d**2 * n1**2 * u_p**2 * (n2**2 - 1)) + (2 * d**2 * n1**2 * u_p * (n2**2 - 1) + 2 * d**2 * n2**2 * u_p * (n1**2 - 1))**2 - 4 * (d - d2)**2 * (d**2 * n1**2 * (n2**2 - 1) + d**2 * n2**2 * (n1**2 - 1)) * (d2 - v_p)**2 + 10 * d**4 * n1**2 * n2**2 * u_p**2 * (n1**2 - 1) * (n2**2 - 1)
        s8 = -2 * (2 * d**2 * n1**2 * u_p * (n2**2 - 1) + 2 * d**2 * n2**2 * u_p * (n1**2 - 1)) * (d**2 * n2**2 * (u_p**2 * (n1**2 - 1) + d**2 * n1**2) - d**2 * n2**2 * (d - d2)**2 - d**2 * n1**2 * (d2 - v_p)**2 + d**2 * n1**2 * u_p**2 * (n2**2 - 1)) - 4 * d**4 * n1**2 * n2**2 * u_p * ((n2**2 - 1) * (u_p**2 * (n1**2 - 1) + d**2 * n1**2) - (n2**2 - 1) * (d - d2)**2 - (n1**2 - 1) * (d2 - v_p)**2 + d**2 * n2**2 * (n1**2 - 1)) - 4 * d**4 * n1**2 * n2**2 * u_p**3 * (n1**2 - 1) * (n2**2 - 1)
        s9 = (d**2 * n2**2 * (u_p**2 * (n1**2 - 1) + d**2 * n1**2) - d**2 * n2**2 * (d - d2)**2 - d**2 * n1**2 * (d2 - v_p)**2 + d**2 * n1**2 * u_p**2 * (n2**2 - 1))**2 + 2 * d**4 * n1**2 * n2**2 * u_p**2 * ((n2**2 - 1) * (u_p**2 * (n1**2 - 1) + d**2 * n1**2) - (n2**2 - 1) * (d - d2)**2 - (n1**2 - 1) * (d2 - v_p)**2 + d**2 * n2**2 * (n1**2 - 1)) - 4 * d**4 * n1**2 * n2**2 * (d - d2)**2 * (d2 - v_p)**2 + 4 * d**4 * n1**2 * n2**2 * u_p * (2 * d**2 * n1**2 * u_p * (n2**2 - 1) + 2 * d**2 * n2**2 * u_p * (n1**2 - 1))
        s10 = -2 * d**4 * n1**2 * n2**2 * u_p**2 * (2 * d**2 * n1**2 * u_p * (n2**2 - 1) + 2 * d**2 * n2**2 * u_p * (n1**2 - 1)) - 4 * d**4 * n1**2 * n2**2 * u_p * (d**2 * n2**2 * (u_p**2 * (n1**2 - 1) + d**2 * n1**2) - d**2 * n2**2 * (d - d2)**2 - d**2 * n1**2 * (d2 - v_p)**2 + d**2 * n1**2 * u_p**2 * (n2**2 - 1))
        s11 = 4 * d**8 * n1**4 * n2**4 * u_p**2 + 2 * d**4 * n1**2 * n2**2 * u_p**2 * (d**2 * n2**2 * (u_p**2 * (n1**2 - 1) + d**2 * n1**2) - d**2 * n2**2 * (d - d2)**2 - d**2 * n1**2 * (d2 - v_p)**2 + d**2 * n1**2 * u_p**2 * (n2**2 - 1))
        s12 = (-4) * d**8 * n1**4 * n2**4 * u_p**3
        s13 = d**8 * n1**4 * n2**4 * u_p**4

        # Store coefficients in the output array.
        coeffs_out[idx, 0] = s1
        coeffs_out[idx, 1] = s2
        coeffs_out[idx, 2] = s3
        coeffs_out[idx, 3] = s4
        coeffs_out[idx, 4] = s5
        coeffs_out[idx, 5] = s6
        coeffs_out[idx, 6] = s7
        coeffs_out[idx, 7] = s8
        coeffs_out[idx, 8] = s9
        coeffs_out[idx, 9] = s10
        coeffs_out[idx, 10] = s11
        coeffs_out[idx, 11] = s12
        coeffs_out[idx, 12] = s13

    return


@cuda.jit
def finish_projection_kernel(d, d2, normal, n1, n2, pt_vs, candidate_roots, m):
   
    idx = cuda.grid(1)
    if idx < pt_vs.shape[1]:

        pt_v = cuda.local.array((3,), dtype=np.float64)
        for i in range(3):
            pt_v[i] = pt_vs[i, idx]
        
        # Compute coordinate system as in your original kernel:
        por = cuda.local.array((3,), dtype=np.float64)
        cross_product(normal, pt_v, por)
        por_norm = norm3(por)
        por[0] /= por_norm
        por[1] /= por_norm
        por[2] /= por_norm
        
        z1 = cuda.local.array((3,), dtype=np.float64)
        for i in range(3):
            z1[i] = normal[i]

        z1_norm = -1 * norm3(z1)
        z1[0] /= z1_norm
        z1[1] /= z1_norm
        z1[2] /= z1_norm

        # Manually compute the cross product
        z2 = cuda.local.array((3,), dtype=np.float64)
        cross_product(por, z1, z2)
        z2_norm = norm3(z1)
        z2[0] /= z2_norm
        z2[1] /= z2_norm
        z2[2] /= z2_norm

        v_p = dot3(pt_v, z1)
        u_p = dot3(pt_v, z2)
        pt_p = cuda.local.array((2,), dtype=np.float64)
        pt_p[0] = u_p; pt_p[1] = v_p
        
        x_sol = cuda.local.array(12, dtype=np.float64)
        for i in range(12):
            x_sol[i] = candidate_roots[i, idx]

        new_normal = cuda.local.array((2,), dtype=np.float64)
        new_normal[0] = 0.0; new_normal[1] = -1.0
        
        for i in range(12):
            if math.isnan(x_sol[i]):
                break

            x = x_sol[i]
            vi = cuda.local.array((2,), dtype=np.float64)
            vi[0] = x; vi[1] = d
            
            # Inline the logic of refractedRay
            v2= cuda.local.array((2,), dtype=np.float64)
            refractedRayCUDA(vi, new_normal, 1.0, n1, v2)

            q2 = cuda.local.array((2,), dtype=np.float64)
            dot_vr_new_normal = dot2(v2, new_normal)
            for i in range(2):
                q2[i] = vi[i] + (d - d2) * v2[i] / dot_vr_new_normal

            # Inline the logic of refractedRay again
            v3 = cuda.local.array((2,), dtype=np.float64)
            refractedRayCUDA(v2, new_normal, n1, n2, v3)
            
            vrd = cuda.local.array((2,), dtype=np.float64)
            for i in range(2):
                vrd[i] = pt_p[i] - q2[i]

            error = math.fabs(vrd[0] * v3[1] - vrd[1] * v3[0])
            # With this:
            if error < 1e-4:
                for i in range(3):
                    m[i, idx] = x * z2[i] + d * z1[i]
                return
            
def solveForwardProjectionCase3Cuda(d, d2, normal, n1, n2, pt_vs):
    num_pts = pt_vs.shape[1]
    m = np.zeros((3, num_pts), dtype=np.float64)

    # Launch the CUDA kernel
    threads_per_block = 256
    blocks_per_grid = (num_pts + threads_per_block - 1) // threads_per_block
    solveForwardProjectionCase3Kernel[blocks_per_grid, threads_per_block](d, d2, normal, n1, n2, pt_vs, m)
    return m


def gpu_compute_roots(coeffs, tol_imag = 1e-6):
   
    num_pts = coeffs.shape[0]
    n = coeffs.shape[1] - 1  # degree (here, 12)
    
    # Normalize all polynomials: divide each row by its first element.
    coeffs_norm = coeffs / coeffs[:, 0:1]
    
    # Build batch of companion matrices (shape: (num_pts, n, n))
    comp_batch = np.zeros((num_pts, n, n), dtype=np.float64)
    comp_batch[:, 0, :] = -coeffs_norm[:, 1:]
    if n > 1:
        comp_batch[:, 1:, :-1] = np.eye(n - 1, dtype=np.float64)[None, :, :]
    
    # Compute eigenvalues for each companion matrix.
    eigvals = np.linalg.eigvals(comp_batch)  # shape: (num_pts, n)
    
    # Extract real parts and compute absolute imaginary parts.
    real_eigvals = np.real(eigvals)
    imag_eigvals = np.abs(np.imag(eigvals)) 
    real_eigvals[imag_eigvals > tol_imag] = np.nan
    sorted_roots = np.sort(real_eigvals, axis=1)    
    candidate_roots = sorted_roots.T

    return candidate_roots

def torch_batch_roots(coeffs, tol_imag=1e-3):
    """
    Given a numpy array of polynomial coefficients of shape (num_pts, 13)
    (coefficients in descending order, i.e. highest degree first),
    this function converts the coefficients to a torch tensor (on GPU if available),
    normalizes each polynomial (making it monic),
    builds a batch of companion matrices,
    computes eigenvalues using torch.linalg.eig,
    filters out eigenvalues with an absolute imaginary part greater than tol_imag,
    sorts the nearly-real candidate roots, and returns a 2D numpy array of shape (n, num_pts).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coeffs_t = torch.tensor(coeffs, dtype=torch.float64, device=device)  # shape (num_pts, 13)
    num_pts, m_val = coeffs_t.shape
    n = m_val - 1  # degree (here, 12)

    # Normalize: make each polynomial monic.
    coeffs_t = coeffs_t / coeffs_t[:, 0:1]

    # Build the batch of companion matrices: shape (num_pts, n, n)
    comp = torch.zeros((num_pts, n, n), dtype=torch.float64, device=device)
    comp[:, 0, :] = -coeffs_t[:, 1:]
    if n > 1:
        comp[:, 1:, :-1] = torch.eye(n - 1, dtype=torch.float64, device=device).unsqueeze(0).expand(num_pts, -1, -1)

    # Compute eigenvalues using torch.linalg.eig (general eigensolver)
    eigvals = torch.linalg.eig(comp)[0]  # shape (num_pts, n), complex tensor
    # Filter for nearly real eigenvalues using tol_imag.
    real_part = eigvals.real
    imag_part = eigvals.imag.abs()
    real_part[imag_part > tol_imag] = float('nan')

    # Sort the candidate roots along dimension 1.
    sorted_roots, _ = torch.sort(real_part, dim=1)  # shape: (num_pts, n)
    candidate_roots = sorted_roots.cpu().numpy().T  # shape (n, num_pts)
    return candidate_roots

 
# Function to compute the roots of polynomials using list comprehension and numpy.            
def batch_np_roots(coeffs, tol_imag=1e-6):
    num_pts = coeffs.shape[0]
    n = coeffs.shape[1] - 1  # polynomial degree, here 12
    
    # Normalize each polynomial (make monic)
    coeffs_norm = coeffs / coeffs[:, 0:1]  # broadcasting division

    # Compute roots for each polynomial using list comprehension.
    # This yields a list of arrays, one per polynomial.
    roots_list = [np.roots(poly) for poly in coeffs_norm]

    # For each polynomial, filter for nearly real roots (imaginary part below tol_imag),
    # take the real parts, and sort them.
    candidate_list = [
        np.sort(np.real(roots[np.abs(np.imag(roots)) < tol_imag])).astype(np.float64)
        for roots in roots_list
    ]
    
    # Prepare a result array of shape (num_pts, n) filled with NaN.
    candidate_roots = np.full((num_pts, n), np.nan, dtype=np.float64)
    
    # Pad each candidate array to length n.
    for i, cand in enumerate(candidate_list):
        length = cand.size
        if length > 0:
            candidate_roots[i, :length] = cand

    # Transpose so that the output shape is (n, num_pts)
    return candidate_roots.T

def solveForwardProjectionCase3Hybrid(d, d2, normal, n1, n2, pt_vs):
    num_pts = pt_vs.shape[1]
    
    # Stage 1: Compute polynomial coefficients for each input ray.
    coeffs = np.zeros((num_pts, 13), dtype=np.float64)
    d_coeffs = cuda.to_device(coeffs)
    threads_per_block = 256
    blocks_per_grid = (num_pts + threads_per_block - 1) // threads_per_block
    compute_poly_coeffs_kernel[blocks_per_grid, threads_per_block](d, d2, normal, n1, n2, pt_vs, d_coeffs)
    coeffs = d_coeffs.copy_to_host()
    
    # Stage 2: Use numpy to compute the roots.
    candidate_roots = gpu_compute_roots(coeffs)
    d_candidate_roots = cuda.to_device(candidate_roots)

    # Stage 3: Run second kernel to finish the projection computation.
    m = np.zeros((3, num_pts), dtype=np.float64)
    d_m = cuda.to_device(m)
    finish_projection_kernel[blocks_per_grid, threads_per_block](d, d2, normal, n1, n2, pt_vs, d_candidate_roots, d_m)
    m = d_m.copy_to_host()
    return m