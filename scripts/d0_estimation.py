import argparse
import yaml

import numpy as np
from scipy.optimize import least_squares

import pinax_utlis as pu



def parse_args():
    parser = argparse.ArgumentParser(description='Estimate d0')
    parser.add_argument('--cfg', type=str, default='./config.yaml', help='YAML Configuration file')
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()

def read_cfg(cfg_file):
    # Read the YAML configuration file
    with open(cfg_file, 'r') as file:
        try:
            cfg = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return cfg

def main():

    args = parse_args()
    cfg = read_cfg(args.cfg)

    # Camera Parameters
    matrixK = np.array(cfg['K_matrix'], dtype=np.float64)
    matrixK = matrixK.reshape(3,3)
    distCoeffs= np.array(cfg['D_matrix'], dtype=np.float64)

    # Flat port parameters
    d1=cfg['glass_thickness']
    normal=np.array(cfg['normal_vector'], dtype=np.float64) #normal vector

    # Refraction indexes
    ng=cfg['glass_refractive_index']
    nw=cfg['water_refractive_index']
        

    x0 = cfg['initial_guess']
    init_guess = np.array([x0, 0.0, 0.0], dtype=np.float64)
    res = least_squares(pu.optim_d0, init_guess, args=(matrixK, normal, ng, nw, d1))
    x0 = res.x[0]
    virtual_d_0 = res.x[1]
    optimal_caustic_length = res.x[2]
    approxErr = res.cost

    print(f'\n---- Optimization Results ----')
    if cfg['verbose']:
        print(f'Optimization Details: {res}')
    print(f'Optimal physical d_0: {x0}')
    print(f'Optimal virtual d_0: {virtual_d_0}')
    print(f'Optimal caustic length: {optimal_caustic_length}')
    print(f'Approximation error: {approxErr}')


    real_d0 = cfg['distance_to_glass']
    r_input = np.array([real_d0, 0.0, 0.0], dtype=np.float64)
    real_caustic_length = pu.optim_d0(r_input, matrixK, normal, ng, nw, d1)
    real_virtual_d_0 = r_input[1]
    print(f'---- Real Values ----')
    print(f'Real physical d_0: {real_d0}')
    print(f'Virtual Plane: {real_virtual_d_0}')
    print(f'Caustic length: {real_caustic_length}') 
    print(f'---------------------')


    return 0


if __name__ == "__main__":
    main()