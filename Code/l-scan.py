"""
Scan over the l quantum number for Hydrogen atom and find eigenvalues.
"""
import numpy as np
from src import *
import ray 
import argparse

N = 1000
r_max = 10.0 
granularity = 100000
n_scan = 500
l_values = np.arange(0, 10)

@ray.remote
def hydrogen_worker(params):
    i, l_val, r_max, n_scan, granularity = params
    r_range = np.linspace(1e-3, r_max, granularity)

    k_func = lambda r, E: k_hydrogen(r, E, l=l_val)
    seed_out = lambda r, E: seed_hydrogen(r, E, l=l_val)

    eigenvalues = find_eigenvalues(r_range, k_func, seed_out, shoot_par_range=(-0.6, -0.0001), 
                                   shoot_func=shoot, n_scan=n_scan, renorm_every=1000)
    
    return i, l_val, eigenvalues

def main():
    parser = argparse.ArgumentParser(description="Scan over l values for Hydrogen atom.")
    parser.add_argument("--l_min", type=int, default=0, help="Minimum l value to scan")
    parser.add_argument("--l_max", type=int, default=9, help="Maximum l value to scan")
    parser.add_argument("--r_max", type=float, default=10.0, help="Maximum r value for scanning")
    parser.add_argument("--granularity", type=int, default=100000, help="Number of points in r range")
    parser.add_argument("--n_scan", type=int, default=500, help="Number of energy values to scan for each l")
    args = parser.parse_args()

    l_vals = np.arange(args.l_min, args.l_max + 1)
    params = [(i, l_val, args.r_max, args.n_scan, args.granularity) for i, l_val in enumerate(l_vals)]
    print(f"Total parameter combinations: {len(params)}. Dispatching tasks to Ray...")

    context = init_ray()
    futures = [hydrogen_worker.remote(param) for param in params]

    result_dict = {}

    for future in ray.get(futures):
        try:
            i, l_val, eigenvalues = future
            result_dict[f"l_{l_val}"] = eigenvalues
        except Exception as e:
            print(f"Error processing l={l_val}: {e}")

    # Save results to file
    fn = f"./Data/hydrogen_eigenvalues_l{args.l_min}_to_{args.l_max}_r{args.r_max}_g{int(np.log10(args.granularity))}_ns{int(args.n_scan)}.npz"
    np.savez(fn, **result_dict, l_vals=l_vals)
    print(f"Results saved to {fn}")
    
if __name__ == "__main__":
    main()
    ray.shutdown()
    print("All tasks completed and Ray shutdown.")
