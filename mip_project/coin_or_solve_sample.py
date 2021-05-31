import argparse
import time
from mip import *

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", required=True, help="mps file")
    ap.add_argument("-m", "--max_seconds", default=60, type=int, help="max seconds for the optimizer")
    ap.add_argument("-c", "--cuts", default=-1, type=int, help="generate cuts")
    ap.add_argument("-p", "--preprocess", default=-1, type=int, help="do preprocess")
    ap.add_argument("-l", "--lp_method", choices=['primal_simplex', 'dual_simplex', 'barrier'], 
        help="lp_method 0=auto/1=dual_simplex/2=primal_simplex/3=barrier")
    ap.add_argument("-s", "--solver", required=True, choices=['cbc', 'ido'])
    args = vars(ap.parse_args())
    print(f"args: {args}")
    if args['solver'] == 'cbc':
        cbc_solver(args)
    else:
        my_solver(args)

tr = {"dual_simplex": 1, "primal_simplex": 2, "barrier": 3}

def cbc_solver(args):
    m = Model(sense=MINIMIZE, solver_name=CBC)
    m.read(args['file'])
    m.max_gap = 0.1
    m.cuts = args['cuts']
    m.preprocess = args['preprocess']
    m.lp_method = LP_Method(tr[args['lp_method']])
    status = m.optimize(max_seconds=args['max_seconds'])


# globals (yuck) for my_solver
on_tree = []
best_solution = 9999999
best_possible = -9999999
fathomed = 0


def my_solver(args):
    """
    homegrown branch and bound implementation
    using the CLP linear relaxation solver at each node
    """
    m = Model(sense=MINIMIZE, solver_name=CBC)    
    m.read(args['file'])
    m.verbose = 1
    m.max_gap = 0.1
    m.cuts = args['cuts']
    m.preprocess = args['preprocess']
    m.lp_method = LP_Method(tr[args['lp_method']])
    m.optimize(max_seconds=10, relax=True)
    m.verbose = 0
    on_tree.append(m)
    iters = 0
    start, elapsed = time.time(), 0

    while elapsed < args['max_seconds'] and on_tree and abs((best_solution - best_possible)/best_solution) > 0.01:
        found_new_best_solution = handle_leaf(on_tree.pop(0)) 
        found_new_best_possible = update_best_possible()
        if found_new_best_solution or found_new_best_possible or iters % 1000 == 0:
            print(f'After {iters} nodes, {len(on_tree)} on tree, {best_solution} best solution, '
                f'best possible {best_possible} ({time.time() - start} seconds)')
        iters += 1
        if iters % 100 == 0:
            elapsed = time.time() - start
        on_tree.sort(key=lambda m: -num_ints_in_model(m))
    print(f'After {iters} nodes, {len(on_tree)} on tree, {best_solution} best solution, '
        f'best possible {best_possible} ({time.time() - start} seconds)')


def handle_leaf(m):
    """
    handle a leaf node - fathom it if possible, if not split to two new nodes
    """
    global best_solution, fathomed
    #print(m.status, m.objective_value, num_ints_in_model(m))
    m.num_ints_in_model = 0
    if m.objective_value > best_solution:
        #print(f'fathomed worse than incumbent(best_solution): {m.objective_value}')
        fathomed += 1
        return False
    if m.status == OptimizationStatus.OPTIMAL:
        for idx, v in enumerate(m.vars):
            if v.var_type == 'C':
                continue
            if not is_almost_integer(v.x):                
                m1, m2 = m.copy(), m.copy()
                m1.verbose = 0
                m2.verbose = 0
                m1 += v <= int(v.x)
                m1.optimize(max_seconds=10, relax=True)
                on_tree.append(m1)
                m2 += v >= int(v.x) + 1
                m2.optimize(max_seconds=10, relax=True)
                on_tree.append(m2)
                return False
        #print(f'fathomed mip solution: {m.objective_value}')
        fathomed += 1
        if m.objective_value < best_solution:
            best_solution = m.objective_value            
            return True
    else:
        #print(f'fathomed infeasible solution: {m.status}')
        fathomed += 1
        return False


def is_almost_integer(x):
    if x is None:
        return False
    delta = x - int(x)
    return delta < 0.01 or delta > 0.99


def num_ints_in_model(m):
    return len([1 for v in m.vars if is_almost_integer(v.x)])


def update_best_possible():
    """
    update the best_possible (the bound) to the best relaxed solution 
    on the current tree
    """
    global best_possible, on_tree
    new_best_possible = min([m.objective_value for m in on_tree if m.objective_value])
    if new_best_possible > best_possible:
        best_possible = new_best_possible
        return True
    else:
        return False


if __name__ == '__main__':
    main()