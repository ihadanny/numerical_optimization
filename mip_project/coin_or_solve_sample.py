from mip import *

def main():
    m = Model(sense=MINIMIZE, solver_name=CBC)
    m.read('gen-ip021.mps')
    m.max_gap = 0.1
    status = m.optimize(max_seconds=60)

if __name__ == '__main__':
    main()