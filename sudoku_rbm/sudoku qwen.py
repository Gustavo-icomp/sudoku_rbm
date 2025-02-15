import numpy as np

class SudokuLBM:
    def __init__(self, initial_board):
        """
        Initialize the LBM with the given Sudoku board.
        :param initial_board: A 4x4 matrix representing the initial state of the Sudoku puzzle.
                               Empty cells are represented by 0.
        """
        self.board = np.array(initial_board)
        self.size = 4  # Size of the board (4x4)
        self.num_values = 4  # Numbers range from 1 to 4
        self.temperature = 1.0  # Temperature for Gibbs sampling
        self.iterations = 10000  # Number of sampling iterations
        self.variables = np.zeros((self.size, self.size, self.num_values))  # Boolean variables x_{i,j,k}

    def initialize_variables(self):
        """
        Initialize the boolean variables based on the initial board.
        """
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] != 0:
                    k = self.board[i, j] - 1  # Convert number to 0-based index
                    self.variables[i, j, :] = 0
                    self.variables[i, j, k] = 1

    def energy(self):
        """
        Compute the energy of the current configuration.
        """
        energy = 0

        # Constraint 1: Each cell contains exactly one number
        for i in range(self.size):
            for j in range(self.size):
                energy -= np.sum(self.variables[i, j, :])  # At least one number
                energy += np.sum(self.variables[i, j, :] * self.variables[i, j, :])  # No two numbers

        # Constraint 2: Each row contains all numbers
        for i in range(self.size):
            for k in range(self.num_values):
                energy -= np.sum(self.variables[i, :, k])  # Number appears at least once
                energy += np.sum(self.variables[i, :, k] * self.variables[i, :, k])  # No repetition

        # Constraint 3: Each column contains all numbers
        for j in range(self.size):
            for k in range(self.num_values):
                energy -= np.sum(self.variables[:, j, k])  # Number appears at least once
                energy += np.sum(self.variables[:, j, k] * self.variables[:, j, k])  # No repetition

        # Constraint 4: Each block contains all numbers
        for b in range(4):  # Four blocks
            block_i_start = (b // 2) * 2
            block_j_start = (b % 2) * 2
            block = self.variables[block_i_start:block_i_start+2, block_j_start:block_j_start+2, :]
            for k in range(self.num_values):
                energy -= np.sum(block[:, :, k])  # Number appears at least once
                energy += np.sum(block[:, :, k] * block[:, :, k])  # No repetition

        return energy

    def gibbs_sampling(self):
        """
        Perform Gibbs sampling to optimize the RBM and generate solutions.
        """
        for _ in range(self.iterations):
            for i in range(self.size):
                for j in range(self.size):
                    if self.board[i, j] == 0:  # Only update empty cells
                        probabilities = []
                        for k in range(self.num_values):
                            self.variables[i, j, :] = 0
                            self.variables[i, j, k] = 1
                            probabilities.append(np.exp(-self.energy()))
                        probabilities /= np.sum(probabilities)
                        sampled_k = np.random.choice(self.num_values, p=probabilities)
                        self.variables[i, j, :] = 0
                        self.variables[i, j, sampled_k] = 1

    def get_solution(self):
        """
        Extract the final solution from the boolean variables.
        """
        solution = np.zeros((self.size, self.size), dtype=int)
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] != 0:
                    solution[i, j] = self.board[i, j]
                else:
                    solution[i, j] = np.argmax(self.variables[i, j, :]) + 1
        return solution

# Example usage
if __name__ == "__main__":
    # Define an initial 4x4 Sudoku board with some numbers filled in (0 represents empty cells)
    initial_board = [
        [1, 0, 0, 4],
        [0, 3, 2, 0],
        [0, 4, 0, 2],
        [3, 0, 0, 1]
    ]

    lbm = SudokuLBM(initial_board)
    lbm.initialize_variables()
    lbm.gibbs_sampling()
    solution = lbm.get_solution()
    print("Solved Board:")
    print(solution)