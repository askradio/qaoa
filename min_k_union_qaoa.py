from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.circuit.library import qaoa_ansatz

import numpy as np


####################################################################################
# 1.Input dataset
####################################################################################
example_dataset = np.array([[1, 0, 0, 1, 0], [0, 0, 1, 0, 1], [0, 1, 0, 1, 1], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [1, 0, 1, 0, 1], [0, 0, 0, 0, 1], [0, 1, 1, 0, 0]])

address_bit = example_dataset.shape[1]
num_of_patterns = example_dataset.shape[0]


print("---- 1. Input dataset ----")
print("example_dataset")
print(example_dataset)

print("address_bit: ", address_bit, ", ", "num_of_patterns: ", num_of_patterns)
print("----\n")

####################################################################################
# 2. QUBO formulation
####################################################################################
# Constant term
input_dim = address_bit + num_of_patterns # number of qubits needed
S = address_bit # number of address bits
V = num_of_patterns # number of patterns
k = 3 # number of row address bits
C = 1
A = C*V + 1 # A > C * V
B = C*V + 1 # B > C * V
print("---- 2. QUBO formulation ----")
print("input_dim = ", input_dim, ", ", "S = ", S, ", ", "V = ", V, ", ", "C = ", C, ", ", "A = ", A, ", ", "B = ", B, ", ", "k = ", k)

def get_qubo_matrix(parameters):
    [A, B, C, S, V, k, input_dim] = parameters

    H_A = np.zeros((input_dim, input_dim))
    r_idx, c_idx = np.triu_indices(S, k=1)
    H_A[r_idx, c_idx] = 2 * A
    H_A[np.arange(S), np.arange(S)] = -(2*k - 1) * A

    H_B = np.zeros((input_dim, input_dim))
    for i in range(V):
        for j in range(S):
            if example_dataset[i, j] == 1:
                H_B[j, j] += B
                H_B[j, S+i] += - B

    H_C = np.zeros((input_dim, input_dim))
    H_C[np.arange(S, input_dim), np.arange(S, input_dim)] = C

    H = H_A + H_B + H_C
    return H

qubo_matrix = get_qubo_matrix([A, B, C, S, V, k, input_dim])

print("qubo_matrix")
print(qubo_matrix)

# Solving QUBO using classical solver
from pyqubo import solve_qubo

qubo_dict = {}
Q = qubo_matrix
n = Q.shape[0]
for i in range(n):
    for j in range(i, n): # Since Q is symmetric, it's sufficient to only include i<=j 
        val = Q[i, j]
        if val != 0.0:
            qubo_dict[(i, j)] = val

solution = solve_qubo(qubo_dict)
# print("---- solution ----")
# print("solution = ", solution)

def extract_bitstrings(solution_dict, S, V):
    sorted_keys = sorted(solution_dict.keys())
    first_S_bits = ''.join(str(solution_dict[key]) for key in sorted_keys[:S])
    last_V_bits = ''.join(str(solution_dict[key]) for key in sorted_keys[-V:])
    return first_S_bits, last_V_bits

optimal_row_address_bit, optimal_row_miss_address_request_index = extract_bitstrings(solution, S, V)

# 결과 출력

print(f"optimal row_address bit = {optimal_row_address_bit}")
print(f"optimal row-miss address request index = {optimal_row_miss_address_request_index}")
x = np.array([solution[i] for i in range(input_dim)])
optimal_cost_value = x @ Q @ x
print("optimal_cost_value = ", optimal_cost_value)
print("----\n")

####################################################################################
# 3. QAOA
####################################################################################
# QAOA Repetition
reps = 3

# Mixing Hamiltonian (index for pauli X basis)
H_m = - np.eye(input_dim)

# Cost Hamiltonian (index for pauli Z basis)
H_c = qubo_matrix

print("---- 3. QAOA ----")
print("mixer_hamiltonian")
print(H_m)
print("cost_hamiltonian")
print(H_c)

def get_cost_hamiltonian(H_c, input_dim):

    num_nodes = input_dim
    pauli_list_c = []
    coeff_list_c = []
    shift = 0

    for i in range(num_nodes):
        for j in range(num_nodes):
            if H_c[i, j] != 0:
                if i == j: # constant * (I-Z)/2
                    x_p = np.zeros(num_nodes, dtype=bool)
                    z_p = np.zeros(num_nodes, dtype=bool)
                    z_p[i] = True
                    pauli_list_c.append(Pauli((z_p, x_p)))
                    coeff_list_c.append(- 0.5 * H_c[i, i])
                    shift += 0.5 * H_c[i, i] # Identity term
                else: # constant * (I-Z)/2 * (I-Z)/2
                    x_p = np.zeros(num_nodes, dtype=bool)
                    z_p = np.zeros(num_nodes, dtype=bool)
                    z_p[i] = True
                    pauli_list_c.append(Pauli((z_p, x_p)))
                    coeff_list_c.append(- 0.25 * H_c[i, j])
                    x_p = np.zeros(num_nodes, dtype=bool)
                    z_p = np.zeros(num_nodes, dtype=bool)
                    z_p[j] = True
                    pauli_list_c.append(Pauli((z_p, x_p)))
                    coeff_list_c.append(- 0.25 * H_c[i, j])
                    x_p = np.zeros(num_nodes, dtype=bool)
                    z_p = np.zeros(num_nodes, dtype=bool)
                    z_p[i] = True
                    z_p[j] = True
                    pauli_list_c.append(Pauli((z_p, x_p)))
                    coeff_list_c.append(0.25 * H_c[i, j])
                    shift += 0.25 * H_c[i, j] # Identity term

    hamiltonian_c = SparsePauliOp(pauli_list_c, coeffs=coeff_list_c)

    return hamiltonian_c, shift

def get_mixer_hamiltonian(H_m, input_dim):
    num_nodes = input_dim
    pauli_list_m = []
    coeff_list_m = []
    for i in range(num_nodes):
        x_p = np.zeros(num_nodes, dtype=bool)
        z_p = np.zeros(num_nodes, dtype=bool)
        x_p[i] = True
        pauli_list_m.append(Pauli((z_p, x_p)))
        coeff_list_m.append(H_m[i, i])
    return SparsePauliOp(pauli_list_m, coeffs=coeff_list_m)

hamiltonian_c, offset = get_cost_hamiltonian(H_c, input_dim)
hamiltonian_m = get_mixer_hamiltonian(H_m, input_dim)

print("mixer_hamiltonian")
print(hamiltonian_m)
print("cost_hamiltonian")
print(hamiltonian_c)
print("offset")
print(offset)

ansatz = qaoa_ansatz(cost_operator=hamiltonian_c, reps=reps, mixer_operator=hamiltonian_m, insert_barriers=False)
print("num ansatz parameters = ", ansatz.num_parameters)
print("----\n")

####################################################################################
# 4. Simulation & Optimization
####################################################################################

# 4.1 COBYLA
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from scipy.optimize import minimize

def cost_func_qaoa(params, ansatz, observable, estimator):
    pub = (ansatz, observable, params)
    job = estimator.run([pub])
    result = job.result()[0]
    return result.data.evs

print("---- 4. Simulation & Optimization ----")
print("target_expectation_value = ", optimal_cost_value - offset)

initial_guess = np.random.uniform(-2*np.pi, 2*np.pi, size=ansatz.num_parameters)

parameters = {'maxiter': 200, 'tol': 1e-6, 'disp': True, 'rhobeg': 1.0}
print("initial_guess")
print(initial_guess)

estimator_sim = StatevectorEstimator()

### Progress Bar
from tqdm import tqdm
class OptCallback:
    def __init__(self, max_iter):
        self.pbar = tqdm(total=max_iter, 
                        desc="Optimization Progress",
                        position=0,
                        leave=True,
                        dynamic_ncols=True,  
                        ascii=True)          
        self.costs = []
        self.iter = 0
        
    def __call__(self, xk):
        self.iter += 1
        current_cost = cost_func_qaoa(xk, ansatz, hamiltonian_c, estimator_sim)
        self.costs.append(float(current_cost))
        self.pbar.update(1)
        self.pbar.set_postfix({'Cost': f'{float(current_cost):.6f}'})
###################
callback = OptCallback(parameters['maxiter'])

simulation_result = minimize(fun=cost_func_qaoa, x0=initial_guess, \
                  args=(ansatz, hamiltonian_c, estimator_sim), \
                    method='COBYLA', options=parameters, callback=callback)
# Close progress bar
callback.pbar.close()
# Optional: Print best cost found
print(f"Best cost found: {min(callback.costs):.6f}")

print("\n")
print("simulation_result")
print(simulation_result)
print("Result parameters")
print(simulation_result['x'])

# 4.2 Optimization result
best_ansatz = ansatz.assign_parameters(simulation_result['x'])
best_ansatz.measure_all()

shots = 1000
sampler_sim = StatevectorSampler()
job_sim = sampler_sim.run([best_ansatz], shots=shots)
data_pub_sim = job_sim.result()[0].data
bitstrings_sim = data_pub_sim.meas.get_bitstrings()
counts_sim = data_pub_sim.meas.get_counts()

sorted_counts = sorted(counts_sim.items(), key=lambda x: x[1], reverse=True)
print("\nTop 3 measurement results (Big Endian):")
for bitstring, count in sorted_counts[:3]:
    print(f"Bitstring: {bitstring[::-1]}, Measurement count: {count}")

####################################################################################
# 5. Real Quantum Device & Optimization
####################################################################################

