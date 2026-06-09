import random

def simulate_input():
	cells = [random.randint(0, 1) for _ in range(20)]
	occupied_cells = [i+1 for i in range(20) if cells[i] == 1]
	return occupied_cells