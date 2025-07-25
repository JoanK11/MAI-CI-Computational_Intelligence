# problem_utils.py
import tsplib95

def load_tsplib_problem(filename):
    return tsplib95.load(filename)
