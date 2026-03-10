"""
Scan over k_vals for fiber problem to find dispersion relation.
"""
import numpy as np
from src import *
import ray
import argparse


@ray.remote
def fiber_worker(params):
    i, k_val, x_max, n_scan, granularity = params

    x_range = np.linspace(1e-3, x_max, granularity)
    core_boundary_idx = np.searchsorted(x_range, 1.0)

    k_func = lambda r, lam: k_fiber(r, lam, k=k_val)
    lam_range = lambda a: (1.0 * a + 1e-3, 2.0 * a - 1e-3)
    seed_out = lambda r, lam: seed_fiber(r, lam, k=k_val)
    seed_in = lambda r, lam, idx: seed_fiber_inward(r, lam, k_val, idx)
    eigenvalues = find_eigenvalues(x_range, k_func, (seed_out, seed_in),
                               shoot_par_range=lam_range(k_val),
                               shoot_func=shoot_midpoint,
                               match_idx=core_boundary_idx,
                               inward_buffer=5.0, n_scan=n_scan, negate_k=True)

    return i, k_val, eigenvalues

def main():
    parser = argparse.ArgumentParser(description="Scan over k values for fiber problem.")
    parser.add_argument("--k_min", type=float, default=0.8, help="Minimum k value to scan")
    parser.add_argument("--k_max", type=float, default=10.0, help="Maximum k value to scan")
    parser.add_argument("--N", type=int, default=200, help="Number of k values to scan")
    parser.add_argument("--x_max", type=float, default=10.0, help="Maximum x value for scanning")
    parser.add_argument("--granularity", type=int, default=100000, help="Number of points in x range")
    parser.add_argument("--n_scan", type=int, default=100, help="Number of lambda values to scan for each k")
    args = parser.parse_args()

    k_vals = np.linspace(args.k_min, args.k_max, args.N)
    params = [(i, k_val, args.x_max, args.n_scan, args.granularity) for i, k_val in enumerate(k_vals)]
    print(f"Total parameter combinations: {len(params)}. Dispatching tasks to Ray...")

    context = init_ray()
    futures = [fiber_worker.remote(param) for param in params]

    result_dict = {}

    for future in ray.get(futures):
        try:
            i, k_val, eigenvalues = future
            result_dict[f"k_{i}"] = eigenvalues
        except Exception as e:
            print(f"Error processing k={k_val}: {e}")

    # Save results to file
    fn = f"./Data/fiber_dispersion_k{args.k_min}_to_{args.k_max}_N{args.N}_x{args.x_max}_g{int(np.log10(args.granularity))}.npz"
    np.savez(fn, **result_dict, k_vals=k_vals)
    print(f"Results saved to {fn}")

if __name__ == "__main__":
    main()
    ray.shutdown()
    print("All tasks completed and Ray shutdown.")
