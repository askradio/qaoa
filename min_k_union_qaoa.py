# Import libraries 
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.circuit.library import qaoa_ansatz
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
import numpy as np
from pyqubo import solve_qubo
from scipy.optimize import minimize
from tqdm import tqdm

def load_dataset():
    """
    Load dataset and return related parameters.
    
    Returns:
        example_dataset (np.ndarray): Input dataset.
        address_bit (int): Number of address bits.
        num_of_patterns (int): Number of patterns.
        row_address_bit (int): Number of row address bits.
    """
    # Flipped bits as 1, otherwise 0
    example_dataset = np.array([
        [1, 0, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 1, 1, 0, 0]
    ])

    row_address_bit = 3
    address_bit = example_dataset.shape[1]
    num_of_patterns = example_dataset.shape[0]
    
    return example_dataset, address_bit, num_of_patterns, row_address_bit

def formulate_qubo_from_min_k_union(example_dataset, qubo_parameters):
    """
    Formulate QUBO matrix from min-k union problem.
    
    Args:
        example_dataset (np.ndarray): Input dataset.
        qubo_parameters (dict): Parameters used in QUBO formulation.
    
    Returns:
        qubo_matrix (np.ndarray): Formulated QUBO matrix.
    """
    input_dim = qubo_parameters['input_dim']  # Number of address bits + number of patterns
    S = qubo_parameters['S']  # Number of address bits
    V = qubo_parameters['V']  # Number of patterns
    k = qubo_parameters['k']  # Number of row address bits
    # Constant terms
    A = qubo_parameters['A']
    B = qubo_parameters['B']  
    C = qubo_parameters['C']  
   
    # Formulate QUBO matrix
    print("---- Formulating QUBO matrix ----")
    H_A = np.zeros((input_dim, input_dim)) # Penalty matrix1 to satisfy the constraint.
    r_idx, c_idx = np.triu_indices(S, k=1)
    H_A[r_idx, c_idx] = 2 * A
    H_A[np.arange(S), np.arange(S)] = -(2 * k - 1) * A

    H_B = np.zeros((input_dim, input_dim)) # Penalty matrix2 to satisfy the constraint.
    for i in range(V):
        for j in range(S):
            if example_dataset[i, j] == 1:
                H_B[j, j] += B
                H_B[j, S + i] += -B

    H_C = np.zeros((input_dim, input_dim)) # Cost matrix. To minimize the cost, we need to minimize the number of row-miss address request patterns.
    H_C[np.arange(S, input_dim), np.arange(S, input_dim)] = C

    H = H_A + H_B + H_C
    qubo_matrix = H

    print("---- QUBO matrix formulated ----")

    return qubo_matrix

def solve_qubo_classically(qubo_matrix):
    """
    Solve QUBO problem with classical solver.
    
    Args:
        qubo_matrix (np.ndarray): QUBO matrix.
    
    Returns:
        solution (dict): QUBO solution dictionary.
    """
    print("---- Solving QUBO with classical solver ----")
    qubo_dict = {}
    Q = qubo_matrix
    n = Q.shape[0]
    for i in range(n):
        for j in range(i, n):  # Q is symmetric, so only i <= j is checked
            val = Q[i, j]
            if val != 0.0:
                qubo_dict[(i, j)] = val

    solution = solve_qubo(qubo_dict)
    print("---- Classical QUBO solution obtained. ----\n")
    return solution

def extract_bitstrings(solution_dict, S, V):
    """
    Extract bitstrings from solution dictionary.
    
    Args:
        solution_dict (dict): QUBO solution dictionary.
        S (int): Number of address bits.
        V (int): Number of patterns.
    
    Returns:
        address_bits (str): Address bit string.
        pattern_indices (str): index strings of row-miss address request patterns.
    """
    sorted_keys = sorted(solution_dict.keys())
    address_bits = ''.join(str(solution_dict[key]) for key in sorted_keys[:S])
    pattern_indices = ''.join(str(solution_dict[key]) for key in sorted_keys[-V:])
    return address_bits, pattern_indices

def calculate_optimal_cost(Q, solution, input_dim):
    """
    Calculate optimal cost value.
    
    Args:
        Q (np.ndarray): QUBO matrix.
        solution (dict): QUBO solution dictionary.
        input_dim (int): Number of qubits.
    
    Returns:
        optimal_cost_value (float): Optimal cost value.
    """
    x = np.array([solution[i] for i in range(input_dim)])
    optimal_cost_value = x @ Q @ x
    return optimal_cost_value

def construct_hamiltonians(H_c_matrix, H_m_matrix, input_dim):
    """
    Construct cost and mixer Hamiltonians.
    
    Args:
        H_c_matrix (np.ndarray): Cost Hamiltonian matrix.
        H_m_matrix (np.ndarray): Mixer Hamiltonian matrix.
        input_dim (int): Number of qubits.
    
    Returns:
        hamiltonian_c (SparsePauliOp): Cost Hamiltonian.
        hamiltonian_m (SparsePauliOp): Mixer Hamiltonian.
        offset (float): Hamiltonian offset.
    """
    def get_cost_hamiltonian(H_c, input_dim): # Only Z-basis
        num_nodes = input_dim
        pauli_list_c = []
        coeff_list_c = []
        shift = 0

        for i in range(num_nodes):
            for j in range(num_nodes):
                if H_c[i, j] != 0:
                    if i == j:  # coeff * (I-Z_i)/2 
                        x_p = np.zeros(num_nodes, dtype=bool)
                        z_p = np.zeros(num_nodes, dtype=bool)
                        z_p[i] = True
                        pauli_list_c.append(Pauli((z_p, x_p)))
                        coeff_list_c.append(- 0.5 * H_c[i, i])
                        shift += 0.5 * H_c[i, i] # Constant expectation value from Identity term (from coeff * I/2)
                    else: 
                        # coeff * (I - Z_i)(I - Z_j) = coeff * (I - Z_i - Z_j + Z_i Z_j)
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

                        shift += 0.25 * H_c[i, j] # Constant expectation value from Identity term (from coeff * I/2 *I/2)

        hamiltonian_c = SparsePauliOp(pauli_list_c, coeffs=coeff_list_c)
        return hamiltonian_c, shift

    def get_mixer_hamiltonian(H_m, input_dim): # Only X-basis
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
    
    # Construct cost Hamiltonian
    print("---- Constructing cost Hamiltonian ----")
    hamiltonian_c, offset = get_cost_hamiltonian(H_c_matrix, input_dim)
    print("---- Cost Hamiltonian constructed ----\n")
        
    # Construct mixer Hamiltonian
    print("---- Constructing mixer Hamiltonian ----")
    hamiltonian_m = get_mixer_hamiltonian(H_m_matrix, input_dim)
    print("---- Mixer Hamiltonian constructed ----\n")
    
    return hamiltonian_c, hamiltonian_m, offset


def build_ansatz(hamiltonian_c, hamiltonian_m, reps):
    """
    Build QAOA Ansatz.
    
    Args:
        hamiltonian_c (SparsePauliOp): Cost Hamiltonian.
        hamiltonian_m (SparsePauliOp): Mixer Hamiltonian.
        reps (int): Number of QAOA repetitions.
    
    Returns:
        ansatz (QuantumCircuit): QAOA Ansatz.
    """
    print("---- Building QAOA Ansatz ----")
    ansatz = qaoa_ansatz(cost_operator=hamiltonian_c, reps=reps, mixer_operator=hamiltonian_m, insert_barriers=False)
    print("---- QAOA Ansatz built ----\n")
    return ansatz

def cost_func_qaoa(params, ansatz, observable, estimator):
    """
    Define cost function for QAOA.
    
    Args:
        params (np.ndarray): Parameter vector for ansatz circuit.
        ansatz (QuantumCircuit): QAOA Ansatz.
        observable (SparsePauliOp): Hamiltonian to observe.
        estimator (StatevectorEstimator): State vector estimator.
    
    Returns:
        float: Expectation value.
    """
    pub = (ansatz, observable, params)
    job = estimator.run([pub])
    result = job.result()[0]
    return result.data.evs

# tqdm is used to display the optimization progress.
class OptCallback:
    """
    Callback class to display optimization progress.
    """
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
        current_cost = cost_func_qaoa(xk, self.ansatz, self.observable, self.estimator)
        self.costs.append(float(current_cost))
        self.pbar.update(1)
        self.pbar.set_postfix({'Cost': f'{float(current_cost):.6f}'})

    def set_components(self, ansatz, observable, estimator):
        self.ansatz = ansatz
        self.observable = observable
        self.estimator = estimator

def run_qaoa(hamiltonian_c, hamiltonian_m, initial_guess, reps=3, maxiter=200, tol=1e-4, rhobeg=1.0):
    """
    Run QAOA algorithm and return optimization results.
    
    Args:
        hamiltonian_c (SparsePauliOp): Cost Hamiltonian.
        hamiltonian_m (SparsePauliOp): Mixer Hamiltonian.
        offset (float): Hamiltonian offset.
        qubo_matrix (np.ndarray): QUBO matrix.
        parameters (dict): QUBO configuration parameters.
        reps (int): Number of QAOA repetitions.
        maxiter (int): Maximum number of optimization iterations.
        tol (float): Optimization tolerance.
    
    Returns:
        dict: Optimization results and simulation results.
    """
    
    # Initialize optimization
    print("----------------------------------------")
    print("-------- Run QAOA by simulation --------")
    print("----------------------------------------")

    estimator_sim = StatevectorEstimator() # StatevectorEstimator is used for simulation
    ansatz = build_ansatz(hamiltonian_c, hamiltonian_m, reps) # Build QAOA Ansatz
    observable = hamiltonian_c # Hamiltonian to observe

    print("initial_guess")
    print(initial_guess, "\n")
    print("---- optimization process started. ----")
    # Set callback
    callback = OptCallback(maxiter)
    callback.set_components(ansatz, observable, estimator_sim)

    # Run optimization
    simulation_result = minimize(
        fun=cost_func_qaoa,
        x0=initial_guess,
        args=(ansatz, observable, estimator_sim),
        method='COBYLA',
        options={'maxiter': maxiter, 'tol': tol, 'disp': True, 'rhobeg': rhobeg},
        callback=callback
    )
    
    # Close progress bar
    callback.pbar.close()
    
    print("---- Optimization process finished. ----\n") 
    print("simulation_result")
    print(simulation_result)
    print("\n")
    
    print("--- Simulation sampling of ansatz with optimized parameters ---")
    # Optimized ansatz
    best_ansatz = ansatz.assign_parameters(simulation_result['x'])
    best_ansatz.measure_all()
    
    # Simulation sampling
    shots = 1000
    sampler_sim = StatevectorSampler()
    job_sim = sampler_sim.run([best_ansatz], shots=shots)
    data_pub_sim = job_sim.result()[0].data
    counts_sim = data_pub_sim.meas.get_counts()
    
    sorted_counts = sorted(counts_sim.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 3 measurement results (Big Endian):")
    for bitstring, count in sorted_counts[:3]:
        print(f"Bitstring: {bitstring[::-1]}, Measurement count: {count}")
    
    return {
        'simulation_result': simulation_result,
        'best_parameters': simulation_result['x'],
        'measurement_counts': counts_sim
    }

def main():
    """
    Main function: Run all steps sequentially.
    """
    # 1. Load dataset
    example_dataset, num_of_address_bit, num_of_patterns, num_of_row_address_bit = load_dataset()
    print("example_dataset")
    print(example_dataset)
    print("\n")
    
    # 2. QUBO formulation
    S = num_of_address_bit
    V = num_of_patterns
    k = num_of_row_address_bit  # Number of row address bits
    C = 1
    A = C * V + 1  # A > C * V
    B = C * V + 1  # B > C * V
    input_dim = S + V # Number of qubits
    qubo_parameters = {
        'S': S,
        'V': V,
        'k': k,
        'A': A,
        'B': B,
        'C': C,
        'input_dim': input_dim
    }
    qubo_matrix = formulate_qubo_from_min_k_union(example_dataset, qubo_parameters)
    print("qubo_matrix")
    print(qubo_matrix)
    print("----\n")
    
    # 3. Solve QUBO with classical solver
    solution = solve_qubo_classically(qubo_matrix)
    S = qubo_parameters['S']
    V = qubo_parameters['V']

    optimal_row_address_bit, optimal_row_miss_address_request_index = extract_bitstrings(solution, S, V) # Optimal solution
    print(f"optimal row_address bit = {optimal_row_address_bit}")
    print(f"optimal row-miss address request index = {optimal_row_miss_address_request_index}")
    
    optimal_cost_value = calculate_optimal_cost(qubo_matrix, solution, qubo_parameters['input_dim']) # Optimal cost value
    print(f"optimal_cost_value = {optimal_cost_value}")
    print("\n")
    
    # 4. QAOA setup
    reps = 3  # Number of QAOA repetitions
    H_m = -np.eye(qubo_parameters['input_dim'])  # Mixer Hamiltonian
    H_c = qubo_matrix  # Cost Hamiltonian
    hamiltonian_c, hamiltonian_m, offset = construct_hamiltonians(H_c, H_m, qubo_parameters['input_dim'])
    
    print("mixer_hamiltonian")
    print(hamiltonian_m)
    print("cost_hamiltonian")
    print(hamiltonian_c)
    print(f"offset = {offset}")
    print("\n")
    
    # 5. Run QAOA
    print(f"target_expectation_value = {optimal_cost_value - offset}")
    reps = 3  # Number of QAOA repetitions
    maxiter = 200  # Maximum number of optimization iterations
    tol = 1e-4  # Optimization tolerance
    rhobeg = 1.0  # Initial step size
    initial_guess = np.random.uniform(-2*np.pi, 2*np.pi, size=reps*2)
    optimization_results = run_qaoa(hamiltonian_c=hamiltonian_c, hamiltonian_m=hamiltonian_m, initial_guess=initial_guess
                                    , reps=reps, maxiter=maxiter, tol=tol, rhobeg=rhobeg)
    
    # 6. Run QAOA on Real Quantum Device

if __name__ == "__main__":
    main()
