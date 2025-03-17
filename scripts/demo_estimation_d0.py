import numpy as np
import pinax_utlis as pu
from scipy.optimize import least_squares

def main():

 
    # Example of camera matrix (Ori)
    matrixK = np.array([[900, 0, 512],
                        [0, 900, 384],
                        [0, 0, 1]], dtype=float)

    # Flat port parameters
    d1=10.0; #glass thicknes

    # Refraction indexes
    ng=1.492 #glass refraction index
    nw=1.345; #water refraction index
    normal=np.array([0.0, 0.0, 1.0], dtype=float) #normal vector    

    x0 = 0.1
    init_guess = np.array([x0, 0.0], dtype=float)

    res = least_squares(pu.optim_d0, init_guess, args=(matrixK, normal, ng, nw, d1))
    x0 = res.x[0]
    virtual_d_0 = res.x[1]
    approxErr = res.cost

    print(f'---- Optimization Results ----')
    print(f'Optimal physical d_0: {x0}')
    print(f'Optimal virtual d_0: {virtual_d_0}')
    print(f'Approximation error: {approxErr}')


    real_d0 = 1.5
    r_input = np.array([real_d0, 0.0], dtype=float)
    leght_caustic = pu.optim_d0(r_input, matrixK, normal, ng, nw, d1)
    real_virtual_d_0 = r_input[1]
    print(f'---- Real Values ----')
    print(f'Real physical d_0: {real_d0}')
    print(f'Virtual Plane: {real_virtual_d_0}')
    print(f'Caustic length: {leght_caustic}') 
    print(f'---------------------')


    return 0


if __name__ == "__main__":
    main()