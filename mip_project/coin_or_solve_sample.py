import argparse
from mip import *

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", required=True, help="mps file")
    ap.add_argument("-m", "--max_seconds", default=60, type=int, help="max seconds for the optimizer")
    ap.add_argument("-c", "--cuts", default=-1, type=int, help="generate cuts")
    ap.add_argument("-p", "--preprocess", default=-1, type=int, help="do preprocess")
    args = vars(ap.parse_args())    
    m = Model(sense=MINIMIZE, solver_name=CBC)
    m.read(args['file'])
    m.max_gap = 0.1
    m.cuts = args['cuts']
    m.preprocess = args['preprocess']
    m.pump_passes = 0
    status = m.optimize(max_seconds=args['max_seconds'])

if __name__ == '__main__':
    main()