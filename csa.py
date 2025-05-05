"""
Crow Search Algorithm (CSA) for Kernel Scheduling on FPGAs

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : May 2025
Description : This file contains the implementation of the Crow Search Algorithm (CSA) for
              optimizing kernel scheduling on FPGAs.

"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec


class CrowSearchAlgorithm:
    def __init__(self, num_kernels, num_slots, obj_func, kernel_names=None):
        """
        Initialize the Crow Scheduler.

        Args:
            num_kernels (int): Number of different kernel types.
            num_slots (int): Total number of FPGA slots.
            obj_func (callable): Objective function to evaluate kernel assignments.
            kernel_names (list): Optional list of kernel names for plotting.
        """
        self.num_kernels = num_kernels
        self.num_slots = num_slots
        self.obj_func = obj_func
        self.kernel_names = kernel_names or [f"Kernel {i}" for i in range(num_kernels)]

        self.history = None
        self.best_position = None
        self.best_score = None
        self.best_crow_idx = None

    def normalize_to_slot_sum(self, kernel_counts):
        """
        Normalize a kernel count vector to ensure:
        - Non-negative integers
        - Total slot usage equals self.num_slots
        - Distribution is as even as possible (might not be the best solution)
        """
        kernel_counts = np.clip(np.round(kernel_counts), 0, None)
        total = int(np.sum(kernel_counts))
        while total > self.num_slots:
            idx = np.argmax(kernel_counts)
            kernel_counts[idx] -= 1
            total -= 1
        while total < self.num_slots:
            idx = np.argmin(kernel_counts)
            kernel_counts[idx] += 1
            total += 1
        return kernel_counts.astype(int)

    def run(self, n_crows=20, max_iter=100, awareness_prob=0.1, flight_length=2):
        """
        Execute the Crow Search Algorithm.

        Args:
            n_crows (int): Number of crows in the population.
            max_iter (int): Number of iterations to run.
            awareness_prob (float): Probability of random search.
            flight_length (float): Step size toward memory solutions.
        """
        # Variables to optimize (kernel counts)
        dim = self.num_kernels
        # Bound for each variable is [0, num_slots]
        bounds = (0, self.num_slots)

        # Initialize positions and memory
        positions = np.random.randint(0, self.num_slots + 1, (n_crows, dim))
        for i in range(n_crows):
            positions[i] = self.normalize_to_slot_sum(positions[i])
        memory = positions.copy()

        # Track history for plotting
        history = np.zeros((max_iter + 1, n_crows, dim), dtype=int)
        history[0] = positions

        # Evaluate initial fitness
        fitness = np.array([self.obj_func(pos) for pos in positions])
        best_idx = np.argmin(fitness)
        self.best_position = positions[best_idx].copy()
        self.best_score = fitness[best_idx]
        self.best_crow_idx = best_idx

        # Main optimization loop
        for iter in range(1, max_iter + 1):
            new_positions = positions.copy()
            for crow_idx in range(n_crows):
                random_crow_idx = np.random.randint(n_crows)
                if np.random.rand() >= awareness_prob:
                    # Move toward another crow's memory
                    diff = memory[random_crow_idx] - positions[crow_idx]
                    rand_factor = np.random.rand(dim)
                    new_pos = positions[crow_idx] + flight_length * rand_factor * diff
                else:
                    # Perform random exploration
                    new_pos = np.random.uniform(bounds[0], bounds[1], dim)

                # Ensure new position is valid
                new_pos = self.normalize_to_slot_sum(new_pos)
                new_fit = self.obj_func(new_pos)

                # Update memory if better
                if new_fit < self.obj_func(memory[crow_idx]):
                    memory[crow_idx] = new_pos

                new_positions[crow_idx] = new_pos

            # Update current population
            positions = new_positions
            fitness = np.array([self.obj_func(pos) for pos in positions])
            iter_best_idx = np.argmin(fitness)

            # Update global best if found
            if fitness[iter_best_idx] < self.best_score:
                self.best_score = fitness[iter_best_idx]
                self.best_position = positions[iter_best_idx].copy()
                self.best_crow_idx = iter_best_idx

            # Save this iteration's positions
            history[iter] = positions

        self.history = history

    def plot_crow(self, crow_id):
        """
        Plot heatmap of kernel assignments and fitness evolution for a specific crow.

        Args:
            crow_id (int): ID of the crow to plot.
        """
        if self.history is None:
            raise ValueError("No history found. Run the algorithm first.")

        num_iters, _, num_kernels = self.history.shape
        crow_data = self.history[:, crow_id, :].T.astype(int)
        fitness = np.array([self.obj_func(self.history[iter, crow_id, :]) for iter in range(num_iters)])
        best_iter = int(np.argmin(fitness))

        # Set up grid with 3 rows: colorbar, heatmap, and fitness line plot
        fig = plt.figure(figsize=(14, 6))
        spec = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[0.15, 3, 1], width_ratios=[0.96, 0.04], hspace=0.3)

        cbar_ax = fig.add_subplot(spec[0, 0]) # Colorbar axis
        ax1 = fig.add_subplot(spec[1, 0])   # Heatmap axis
        ax2 = fig.add_subplot(spec[2, 0], sharex=ax1)   # Fitness plot axis

        # --- Heatmap of kernel allocation ---
        sns.heatmap(
            crow_data,
            ax=ax1,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            cbar_ax=cbar_ax,
            cbar_kws={"orientation": "horizontal", "label": "Slots Used"},
            xticklabels=np.arange(num_iters),
            yticklabels=self.kernel_names
        )
        ax1.axvline(x=best_iter, color='red', linestyle='--', lw=2)
        ax1.set_ylabel("Kernel Type")
        ax1.set_title(f"Crow #{crow_id} - Kernel Assignment and Fitness Over Time")

        # --- Line plot of fitness ---
        ax2.plot(np.arange(num_iters), fitness, marker='o', color='black', label="Fitness")
        ax2.axvline(x=best_iter, color='red', linestyle='--', lw=2, label=f"Best Iter ({best_iter})")
        ax2.set_ylabel("Fitness")
        ax2.set_xlabel("Iteration")
        ax2.grid(True)
        ax2.legend()

        # plt.tight_layout()
        plt.show()


# === Example usage ===
def power_performance_model(kernel_counts):
    """
    Simulated base cost model combining power and performance heuristics.
    """

    num_active_kernels = np.count_nonzero(kernel_counts)

    # Simulated power cost: 5 units per active kernel
    power_cost = num_active_kernels * 5
    # Simulated performance cost: 10 units per kernel count
    # Assume performance improves with parallelism,
    # but suffers if the slot use is highly unbalanced (std dev high)
    balance_penalty = np.std(kernel_counts)
    inverse_parallelism = 1 / (np.max(kernel_counts) + 1e-3)
    performance_cost = 10 * inverse_parallelism + 2 * balance_penalty

    # Combine power and performance costs
    return power_cost + performance_cost

def penalized_model(kernel_counts):
    """
    Full cost function with soft penalties:
    - Base cost from power/performance
    - Penalty for too many distinct kernels
    - Penalty for non-power-of-2 slot counts
    """
    # Base cost from power/performance model
    base_cost = power_performance_model(kernel_counts)

    # Soft penalty: >4 kernel types
    num_active_kernels = np.count_nonzero(kernel_counts)
    penalty_distinct = 10 * max(0, num_active_kernels - 4)

    # Soft penalty: non-power-of-2 kernel counts
    penalty_power_of_two = sum(5 for k in kernel_counts if k != 0 and not (k & (k - 1)) == 0)

    return base_cost + penalty_distinct + penalty_power_of_two

if __name__ == "__main__":

    random_seed = 42
    np.random.seed(random_seed)

    num_kernels = 11
    num_slots = 8
    scheduler = CrowSearchAlgorithm(num_kernels, num_slots, penalized_model, kernel_names=list("ABCDEFGHIJK"))
    scheduler.run(n_crows=40, max_iter=100, awareness_prob=0.4, flight_length=1.5)

    print("Best Kernel Counts:", scheduler.best_position)
    print("Best Score:", scheduler.best_score)
    print("Best Crow:", scheduler.best_crow_idx)

    scheduler.plot_crow(crow_id=scheduler.best_crow_idx)
