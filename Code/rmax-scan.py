"""
Scan over r_max values for Hydrogen atom and find eigenvalues.
"""
import numpy as np
from src import *
import ray 
import argparse

R_POINTS = 100000

@ray.remote
def hydrogen_worker_rmax(params):
    i, l_val, r_max, n_scan, granularity = params
    r_range = np.linspace(1e-3, r_max, R_POINTS)

    k_func = lambda r, E: k_hydrogen(r, E, l=l_val)
    seed_out = lambda r, E: seed_hydrogen(r, E, l=l_val)

    eigenvalues = find_eigenvalues(r_range, k_func, seed_out, shoot_par_range=(-0.6, -0.0001), 
                                   shoot_func=shoot, n_scan=n_scan, renorm_every=1000)
    
    return i, l_val, eigenvalues

def main():
    parser = argparse.ArgumentParser(description="Scan over r_max values for Hydrogen atom.")
    parser.add_argument("--l_val", type=int, default=0, help="l value to use for scanning")
    parser.add_argument("--r_scan_start", type=float, default=5.0, help="Starting r_max value for scanning")
    parser.add_argument("--r_scan_end", type=float, default=500.0, help="Ending r_max value for scanning")
    parser.add_argument("--granularity", type=int, default=100000, help="Number of points in r range")
    parser.add_argument("--n_scan", type=int, default=500, help="Number of energy values to scan for each r_max")
    args = parser.parse_args()

    r_scan_values = np.linspace(args.r_scan_start, args.r_scan_end, args.granularity)
    params = [(i, args.l_val, r_max, args.n_scan, args.granularity) for i, r_max in enumerate(r_scan_values)]
    print(f"Total parameter combinations: {len(params)}. Dispatching tasks to Ray...")

    context = init_ray()
    futures = [hydrogen_worker_rmax.remote(param) for param in params]

    result_dict = {}

    for future in ray.get(futures):
        try:
            i, r_max, eigenvalues = future
            result_dict[f"r_{i}"] = eigenvalues
        except Exception as e:
            print(f"Error processing r_max index {i}: {e}")

    # Save results to file
    fn = f"./Data/rmax_scan_l{args.l_val}_rscan{args.r_scan_start}_to_{args.r_scan_end}_g{int(np.log10(args.granularity))}.npz"
    np.savez(fn, **result_dict, r_scan_values=r_scan_values)
    print(f"Results saved to {fn}")

if __name__ == "__main__":
    main()
    ray.shutdown()
    print("All tasks completed and Ray shutdown.")

