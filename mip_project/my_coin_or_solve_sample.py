import time
import argparse
from mip import *

front = []
best_solution = 9999999
best_possible = -9999999
fathomed = 0

def handle_leaf(m):
    global best_solution, fathomed
    #print(m.status, m.objective_value, num_ints(m))
    m.num_ints = 0
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
                front.append(m1)
                m2 += v >= int(v.x) + 1
                m2.optimize(max_seconds=10, relax=True)
                front.append(m2)
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

def num_ints(m):
    return len([1 for v in m.vars if is_almost_integer(v.x)])

def update_best_possible():
    global best_possible, front
    new_best_possible = min([m.objective_value for m in front if m.objective_value])
    if new_best_possible > best_possible:
        best_possible = new_best_possible
        return True
    else:
        return False
    
def main():
    global best_possible, best_solution, front
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", required=True, help="mps file")
    ap.add_argument("-m", "--max_seconds", default=60, type=int, help="max seconds for the optimizer")
    ap.add_argument("-c", "--cuts", default=-1, type=int, help="generate cuts")
    ap.add_argument("-p", "--preprocess", default=-1, type=int, help="do preprocess")
    args = vars(ap.parse_args())    

    m = Model(sense=MINIMIZE, solver_name=CBC)    
    m.read(args['file'])
    m.verbose = 1
    m.max_gap = 0.1
    m.optimize(max_seconds=10, relax=True)
    m.verbose = 0
    front.append(m)
    iters = 0
    start, elapsed = time.time(), 0

    while elapsed < args['max_seconds'] and front and abs((best_solution - best_possible)/best_solution) > 0.01:
        found_new_best_solution = handle_leaf(front.pop(0)) 
        found_new_best_possible = update_best_possible()
        if found_new_best_solution or found_new_best_possible or iters % 1000 == 0:
            print(f'After {iters} nodes, {len(front)} on tree, {best_solution} best solution, '
                f'best possible {best_possible} ({time.time() - start} seconds) gap {abs((best_solution - best_possible)/best_solution)}')
        iters += 1
        if iters % 100 == 0:
            elapsed = time.time() - start
        front.sort(key=lambda m: -num_ints(m))
    print(f'After {iters} nodes, {len(front)} on tree, {best_solution} best solution, '
        f'best possible {best_possible} ({time.time() - start} seconds) gap {abs((best_solution - best_possible)/best_solution)}')

if __name__ == '__main__':
    main()