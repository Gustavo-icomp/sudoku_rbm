from pysat.solvers import Glucose3
import numpy as np

def initialize_board(board):
    global N, block_size
    N = board.shape[0]  # Dynamically determine the size of the board
    block_size = int(np.sqrt(N))  # Block size is the square root of N
    if block_size * block_size != N:
        raise ValueError("The board size must be a perfect square (e.g., 4x4, 9x9).")
    return board

def encode_variable(i, j, k, N):
    """Encode the propositional variable b_i_j_k as a unique integer."""
    return int(i) * int(N) * int(N) + int(j) * int(N) + int(k) + 1

def decode_variable(var, N):
    """Decode the unique integer back to (i, j, k)."""
    var -= 1  # Adjust for 1-based indexing
    i = var // (N * N)
    j = (var // N) % N
    k = var % N
    return i, j, k

def generate_cnf_clauses(board):
    clauses = []
    # Helper function to add "exactly one" constraint
    def exactly_one(variables):
        clauses.append(variables)  # At least one is true
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                clauses.append([-variables[i], -variables[j]])  # At most one is true

    # Cell constraint: Each cell must contain exactly one number
    for i in range(N):
        for j in range(N):
            variables = [encode_variable(i, j, k, N) for k in range(N)]
            exactly_one(variables)

    # Row constraint: Each number appears exactly once in every row
    for i in range(N):
        for k in range(N):
            variables = [encode_variable(i, j, k, N) for j in range(N)]
            exactly_one(variables)

    # Column constraint: Each number appears exactly once in every column
    for j in range(N):
        for k in range(N):
            variables = [encode_variable(i, j, k, N) for i in range(N)]
            exactly_one(variables)

    # Block constraint: Each number appears exactly once in every block
    for block_row in range(block_size):
        for block_col in range(block_size):
            for k in range(N):
                variables = [
                    encode_variable(block_row * block_size + i, block_col * block_size + j, k, N)
                    for i in range(block_size) for j in range(block_size)
                ]
                exactly_one(variables)

    # Initial board state: Pre-filled cells are fixed
    for i in range(N):
        for j in range(N):
            if board[i, j] != 0:
                k = board[i, j] - 1  # Convert to zero-based index
                clauses.append([encode_variable(i, j, k, N)])  # Ensure this is a list of integers

    return clauses

def solve_sudoku_with_sat(board):
    """Solve the Sudoku using a SAT solver."""
    clauses = generate_cnf_clauses(board)
    solver = Glucose3()

    # Add all clauses to the solver
    for clause in clauses:
        solver.add_clause(clause)

    if not solver.solve():
        print("No solution found.")
        return None

    # Extract the solution
    solution = np.zeros((N, N), dtype=int)
    model = solver.get_model()
    for var in model:
        if var > 0:  # Only consider positive literals
            i, j, k = decode_variable(var, N)
            solution[i, j] = k + 1  # Convert back to 1-based number

    return solution

# Example usage
if __name__ == "__main__":
    sudoku_boards = [
        # Base
        np.array([ 
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9]
        ]),
        # Easy
        np.array([
            [9, 2, 0, 3, 4, 0, 0, 8, 0],
            [4, 7, 3, 8, 0, 1, 0, 0, 5],
            [8, 0, 0, 7, 0, 6, 0, 3, 4],
            [0, 8, 5, 0, 0, 9, 2, 0, 0],
            [0, 0, 9, 0, 0, 0, 4, 7, 0],
            [7, 3, 0, 0, 6, 0, 8, 0, 9],
            [0, 0, 0, 0, 0, 0, 0, 0, 2],
            [3, 4, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 7, 2, 5, 4, 3, 6, 0]
        ]),
        # Medium
        np.array([
            [1, 0, 0,  4, 0, 0,  0, 0, 0],
            [3, 9, 6,  0, 0, 0,  0, 4, 0],
            [2, 0, 0,  7, 0, 3,  0, 0, 0],
            
            [6, 7, 2,  0, 3, 0,  9, 8, 0],
            [0, 3, 1,  6, 0, 9,  0, 0, 7],
            [0, 4, 9,  8, 0, 0,  6, 0, 3],
            
            [0, 6, 0,  1, 0, 0,  4, 2, 0],
            [0, 1, 0,  0, 6, 2,  3, 0, 0],
            [9, 2, 8,  0, 0, 4,  0, 0, 1]
        ]), 
        # Hard
        np.array([
            [3, 0, 8,  0, 6, 2,  0, 1, 5],
            [4, 0, 0,  0, 0, 0,  0, 6, 0],
            [0, 6, 0,  0, 7, 0,  0, 0, 8],
            
            [8, 0, 2,  4, 0, 0,  1, 0, 7],
            [0, 0, 0,  0, 0, 8,  3, 0, 0],
            [0, 0, 0,  0, 5, 3,  2, 8, 0],
            
            [0, 0, 0,  5, 0, 0,  6, 7, 0],
            [2, 0, 0,  0, 0, 0,  0, 0, 0],
            [0, 1, 0,  0, 2, 0,  5, 4, 0]
        ]),
        # Specialist
        np.array([
            [0, 3, 5,  8, 0, 0,  9, 2, 0],
            [2, 0, 6,  0, 9, 4,  0, 5, 0],
            [0, 8, 0,  2, 6, 0,  4, 7, 0],
            
            [0, 0, 0,  0, 2, 0,  6, 0, 0],
            [0, 0, 0,  5, 8, 0,  7, 0, 0],
            [3, 0, 8,  7, 0, 0,  0, 0, 9],
            
            [0, 9, 0,  0, 0, 0,  0, 0, 0],
            [0, 0, 0,  0, 0, 1,  3, 0, 2],
            [0, 0, 0,  0, 0, 0,  1, 9, 0]
        ]),
         # Master
        np.array([
            [8, 0, 0,  0, 0, 9,  0, 0, 0],
            [9, 6, 2,  0, 7, 0,  3, 0, 0],
            [0, 3, 7,  2, 5, 6,  0, 0, 0],
            
            [0, 0, 0,  0, 3, 2,  0, 7, 9],
            [0, 0, 1,  7, 0, 0,  0, 0, 0],
            [0, 0, 0,  5, 0, 0,  0, 6, 0],
            
            [0, 8, 0,  4, 2, 0,  9, 0, 0],
            [0, 0, 0,  0, 0, 0,  0, 0, 7],
            [0, 0, 9,  6, 0, 0,  0, 5, 0]
        ]),
        # Extreme
        np.array([
            [0, 0, 0,  0, 7, 0,  0, 0, 0],
            [3, 6, 0,  0, 0, 9,  1, 8, 0],
            [0, 2, 0,  5, 0, 0,  3, 0, 0],
            
            [0, 0, 0,  0, 0, 0,  0, 0, 0],
            [0, 0, 8,  2, 0, 0,  0, 4, 0],
            [0, 4, 0,  6, 0, 3,  0, 0, 0],
            
            [0, 5, 0,  0, 0, 0,  0, 0, 6],
            [6, 0, 3,  0, 0, 0,  2, 0, 0],
            [0, 0, 0,  0, 9, 0,  7, 0, 0]
        ])
        
    ]

    for idx, board in enumerate(sudoku_boards, start=1):
        print(f"Solving Board {idx}...")
        initial_board = initialize_board(board)
        solved_sudoku = solve_sudoku_with_sat(initial_board)
        if solved_sudoku is not None:
            print("Final Solved Sudoku:")
            print(solved_sudoku)
        else:
            print("No solution found.")
        print("-" * 40)