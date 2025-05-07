"""
Crow Search Algorithm (CSA) for Kernel Scheduling on FPGAs

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : May 2025
Description : This file contains the implementation of the Crow Search Algorithm (CSA) for
              optimizing kernel scheduling on FPGAs.

"""


import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

from incremental_learning import online_models as om


class CrowSearchAlgorithm:
    def __init__(self, num_kernels, num_slots, kernel_names=None, obj_func=None, models=None):
        """
        Initialize the Crow Scheduler.

        Args:
            num_kernels (int): Number of different kernel types.
            num_slots (int): Total number of FPGA slots.
            obj_func (callable): Objective function to evaluate kernel assignments.
            kernel_names (list): Optional list of kernel names for plotting.
        """
        print("Initializing Crow Search Algorithm...")
        self.num_kernels = num_kernels
        self.num_slots = num_slots
        self.models = models
        self.kernel_names = kernel_names or [f"Kernel {i}" for i in range(num_kernels)]

        self.history_positions = None
        self.history_fitness = None
        self.best_position = None
        self.best_score = None
        self.best_crow_idx = None

        self.obj_func = self.model_optimization_function if obj_func is None else obj_func
        self.models = models

    def normalize_to_slot_sum(self, kernel_counts, max_slots):
        """
        Normalize a kernel count vector to ensure:
        - Non-negative integers
        - Total slot usage equals max_slots
        - Distribution is as even as possible (might not be the best solution)
        """
        kernel_counts = np.clip(np.round(kernel_counts), 0, None)
        total = int(np.sum(kernel_counts))
        while total > max_slots:
            idx = np.argmax(kernel_counts)
            kernel_counts[idx] -= 1
            total -= 1
        while total < max_slots:
            idx = np.argmin(kernel_counts)
            kernel_counts[idx] += 1
            total += 1
        return kernel_counts.astype(int)

    def build_candidate_schedule(self, base_schedule: np.ndarray, kernel_indices: list, slot_counts: list) -> np.ndarray:
        """
        Build a candidate kernel schedule by applying new slot assignments to a base schedule.

        Parameters:
            base_schedule (np.ndarray): The current kernel slot configuration.
            kernel_indices (list[int]): Indices of schedulable kernels to update.
            slot_counts (list[int]): Slot counts corresponding to each schedulable kernel.

        Returns:
            np.ndarray: A new kernel schedule with the updates applied.
        """
        result = base_schedule.copy()
        result[kernel_indices] = slot_counts
        return result

    def run_no_running_kernels(self, n_crows=20, max_iter=100, awareness_prob=0.1, flight_length=2):
        """
        Execute the Crow Search Algorithm. (No kernels already in execution)

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
            positions[i] = self.normalize_to_slot_sum(positions[i], self.num_slots)
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
                new_pos = self.normalize_to_slot_sum(new_pos, self.num_slots)
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

        self.history_positions = history

    def plot_crow_no_running_kernels(self, crow_id):
        """
        Plot heatmap of kernel assignments and fitness evolution for a specific crow. (No running kernels)

        Args:
            crow_id (int): ID of the crow to plot.
        """
        if self.history_positions is None:
            raise ValueError("No history found. Run the algorithm first.")

        num_iters, _, num_kernels = self.history_positions.shape
        crow_data = self.history_positions[:, crow_id, :].T.astype(int)
        fitness = np.array([self.obj_func(self.history_positions[iter, crow_id, :]) for iter in range(num_iters)])
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

    def run_no_models(self, n_crows=20, max_iter=100, awareness_prob=0.1, flight_length=2, running_kernels: np.array = None, schedulable_kernels: list = None):
        """
        Execute the Crow Search Algorithm.

        This version is designed to search for kernel assignments while considering the
        constraints of running kernels and schedulable kernels. (no models used for prediction)

        Args:
            n_crows (int): Number of crows in the population.
            max_iter (int): Number of iterations to run.
            awareness_prob (float): Probability of random search.
            flight_length (float): Step size toward memory solutions.
            running_kernels (list): List of running kernels. kernels not running shown as 0.
            schedulable_kernels (list): List of schedulable kernels. Each element represents each kernel type.
        """

        # Check if running_kernels and schedulable_kernels are provided
        if running_kernels is None or schedulable_kernels is None:
            raise ValueError("Both running_kernels and schedulable_kernels must be provided.")
        # Ensure running_kernels sum is less than num_slots
        if np.sum(running_kernels) > self.num_slots:
            raise ValueError("The sum of running_kernels exceeds the total number of slots.")

        # Calculate the number of free slots
        free_slots = self.num_slots - np.sum(running_kernels)
        # Variables to optimize (schedulable_kernels)
        dim = len(schedulable_kernels)
        # Bound for each variable is [0, free_slots]
        bounds = (0, free_slots)

        # Initialize positions
        positions = np.random.randint(0, free_slots + 1, (n_crows, dim))
        for crow_idx in range(n_crows):
            positions[crow_idx] = self.normalize_to_slot_sum(positions[crow_idx], free_slots)

        # Track history of positions for plotting
        history_positions = np.zeros((max_iter + 1, n_crows, dim), dtype=int)
        history_positions[0] = positions

        # Evaluate initial fitness
        # The positions to evaluate will include the running kernels and the schedulable kernels (for the model prediction)
        fitness = np.array([
            self.obj_func(self.build_candidate_schedule(running_kernels, schedulable_kernels, pos))
            for pos in positions
            ])


        # Find the best initial position
        best_idx = np.argmin(fitness)
        self.best_position = positions[best_idx].copy()
        self.best_score = fitness[best_idx]
        self.best_crow_idx = best_idx

        # Initialize memory
        crow_position_memory = positions.copy()
        crow_fitness_memory = fitness.copy()

        # Track history of fitness for plotting
        history_fitness = np.zeros((max_iter + 1, n_crows), dtype=float)
        history_fitness[0] = fitness

        # Main optimization loop
        for iter in range(1, max_iter + 1):
            new_positions = positions.copy()
            new_fitness = fitness.copy()
            for crow_idx in range(n_crows):
                random_crow_idx = np.random.randint(n_crows)
                if np.random.rand() >= awareness_prob:
                    # Move toward another crow's memory
                    diff = crow_position_memory[random_crow_idx] - positions[crow_idx]
                    rand_factor = np.random.rand(dim)
                    new_pos = positions[crow_idx] + flight_length * rand_factor * diff
                else:
                    # Perform random exploration
                    new_pos = np.random.uniform(bounds[0], bounds[1], dim)

                # Ensure new position is valid
                new_pos = self.normalize_to_slot_sum(new_pos, free_slots)

                # Evaluate fitness of the new position
                # The positions to evaluate will include the running kernels and the schedulable kernels (for the model prediction)
                kernels_to_evaluate = self.build_candidate_schedule(running_kernels, schedulable_kernels, new_pos)
                # Compute the fitness of the new position
                new_fit = self.obj_func(kernels_to_evaluate)

                # Update memory if better
                if new_fit < crow_fitness_memory[crow_idx]:
                    crow_position_memory[crow_idx] = new_pos
                    crow_fitness_memory[crow_idx] = new_fit

                new_positions[crow_idx] = new_pos
                new_fitness[crow_idx] = new_fit

            # Update current population
            positions = new_positions
            fitness = new_fitness
            iter_best_idx = np.argmin(fitness)

            # Update global best if found
            if fitness[iter_best_idx] < self.best_score:
                self.best_score = fitness[iter_best_idx]
                self.best_position = positions[iter_best_idx].copy()
                self.best_crow_idx = iter_best_idx

            # Save this iteration's positions and fitness
            history_positions[iter] = positions
            history_fitness[iter] = fitness

        self.history_positions = history_positions
        self.history_fitness = history_fitness

    def run_with_models(self, n_crows=20, max_iter=100, awareness_prob=0.1, flight_length=2, running_kernels: np.array = None, schedulable_kernels: list = None, cpu_usage: dict = None):
        """
        Execute the Crow Search Algorithm.

        This version is designed to search for kernel assignments while considering the
        constraints of running kernels and schedulable kernels. (with models used for prediction)

        Args:
            n_crows (int): Number of crows in the population.
            max_iter (int): Number of iterations to run.
            awareness_prob (float): Probability of random search.
            flight_length (float): Step size toward memory solutions.
            running_kernels (list): List of running kernels. kernels not running shown as 0.
            schedulable_kernels (list): List of schedulable kernels. Each element represents each kernel type.
            cpu_usage (dict): Dictionary with CPU usage information.
        """

        # Check if running_kernels and schedulable_kernels are provided
        if running_kernels is None or schedulable_kernels is None:
            raise ValueError("Both running_kernels and schedulable_kernels must be provided.")
        # Ensure running_kernels sum is less than num_slots
        if np.sum(running_kernels) > self.num_slots:
            raise ValueError("The sum of running_kernels exceeds the total number of slots.")

        # Calculate the number of free slots
        free_slots = self.num_slots - np.sum(running_kernels)
        # Variables to optimize (schedulable_kernels)
        dim = len(schedulable_kernels)
        # Bound for each variable is [0, free_slots]
        bounds = (0, free_slots)

        # Initialize positions
        positions = np.random.randint(0, free_slots + 1, (n_crows, dim))
        for crow_idx in range(n_crows):
            positions[crow_idx] = self.normalize_to_slot_sum(positions[crow_idx], free_slots)

        # Track history of positions for plotting
        history_positions = np.zeros((max_iter + 1, n_crows, dim), dtype=int)
        history_positions[0] = positions

        # Evaluate initial fitness
        # The positions to evaluate will include the running kernels and the schedulable kernels (for the model prediction)
        fitness = np.array([
            self.obj_func(running_kernels, self.build_candidate_schedule(running_kernels, schedulable_kernels, pos), cpu_usage)
            for pos in positions
            ])

        # Find the best initial position
        best_idx = np.argmin(fitness)
        self.best_position = positions[best_idx].copy()
        self.best_score = fitness[best_idx]
        self.best_crow_idx = best_idx

        # Initialize memory
        crow_position_memory = positions.copy()
        crow_fitness_memory = fitness.copy()

        # Track history of fitness for plotting
        history_fitness = np.zeros((max_iter + 1, n_crows), dtype=float)
        history_fitness[0] = fitness

        # Main optimization loop
        for iter in range(1, max_iter + 1):
            new_positions = positions.copy()
            new_fitness = fitness.copy()
            for crow_idx in range(n_crows):
                random_crow_idx = np.random.randint(n_crows)
                if np.random.rand() >= awareness_prob:
                    # Move toward another crow's memory
                    diff = crow_position_memory[random_crow_idx] - positions[crow_idx]
                    rand_factor = np.random.rand(dim)
                    new_pos = positions[crow_idx] + flight_length * rand_factor * diff
                else:
                    # Perform random exploration
                    new_pos = np.random.uniform(bounds[0], bounds[1], dim)

                # Ensure new position is valid
                new_pos = self.normalize_to_slot_sum(new_pos, free_slots)

                # Evaluate fitness of the new position
                # The positions to evaluate will include the running kernels and the schedulable kernels (for the model prediction)
                kernels_to_evaluate = self.build_candidate_schedule(running_kernels, schedulable_kernels, new_pos)
                # Compute the fitness of the new position
                new_fit = self.obj_func(running_kernels, kernels_to_evaluate, cpu_usage)

                # Update memory if better
                if new_fit < crow_fitness_memory[crow_idx]:
                    crow_position_memory[crow_idx] = new_pos
                    crow_fitness_memory[crow_idx] = new_fit

                new_positions[crow_idx] = new_pos
                new_fitness[crow_idx] = new_fit

            # Update current population
            positions = new_positions
            fitness = new_fitness
            iter_best_idx = np.argmin(fitness)

            # Update global best if found
            if fitness[iter_best_idx] < self.best_score:
                self.best_score = fitness[iter_best_idx]
                self.best_position = positions[iter_best_idx].copy()
                self.best_crow_idx = iter_best_idx

            # Save this iteration's positions and fitness
            history_positions[iter] = positions
            history_fitness[iter] = fitness

        self.history_positions = history_positions
        self.history_fitness = history_fitness

    def run_standalone(self, n_crows=20, max_iter=100, awareness_prob=0.1, flight_length=2, running_kernels: np.array = None, schedulable_kernels: list = None, cpu_usage: dict = None):
        """
        Execute the Crow Search Algorithm.

        This version is designed to search for kernel assignments while considering the
        constraints of running kernels and schedulable kernels. (with models used for prediction)
        ** This is a standalone run, it basically cleans history and runs the algorithm again. **

        Args:
            n_crows (int): Number of crows in the population.
            max_iter (int): Number of iterations to run.
            awareness_prob (float): Probability of random search.
            flight_length (float): Step size toward memory solutions.
            running_kernels (list): List of running kernels. kernels not running shown as 0.
            schedulable_kernels (list): List of schedulable kernels. Each element represents each kernel type.
            cpu_usage (dict): Dictionary with CPU usage information.
        """

        # Initialize history variables (since it is a standalone run)
        self.history_positions = None
        self.history_fitness = None
        self.best_position = None
        self.best_score = None
        self.best_crow_idx = None

        # Check if running_kernels and schedulable_kernels are provided
        if running_kernels is None or schedulable_kernels is None:
            raise ValueError("Both running_kernels and schedulable_kernels must be provided.")
        # Ensure running_kernels sum is less than num_slots
        if np.sum(running_kernels) > self.num_slots:
            raise ValueError("The sum of running_kernels exceeds the total number of slots.")

        # Calculate the number of free slots
        free_slots = self.num_slots - np.sum(running_kernels)
        # Variables to optimize (schedulable_kernels)
        dim = len(schedulable_kernels)
        # Bound for each variable is [0, free_slots]
        bounds = (0, free_slots)

        # Initialize positions
        positions = np.random.randint(0, free_slots + 1, (n_crows, dim))
        for crow_idx in range(n_crows):
            positions[crow_idx] = self.normalize_to_slot_sum(positions[crow_idx], free_slots)

        # Track history of positions for plotting
        history_positions = np.zeros((max_iter + 1, n_crows, dim), dtype=int)
        history_positions[0] = positions

        # Evaluate initial fitness
        # The positions to evaluate will include the running kernels and the schedulable kernels (for the model prediction)
        fitness = np.array([
            self.obj_func(running_kernels, self.build_candidate_schedule(running_kernels, schedulable_kernels, pos), cpu_usage)
            for pos in positions
            ])

        # Find the best initial position
        best_idx = np.argmin(fitness)
        self.best_position = positions[best_idx].copy()
        self.best_score = fitness[best_idx]
        self.best_crow_idx = best_idx

        # Initialize memory
        crow_position_memory = positions.copy()
        crow_fitness_memory = fitness.copy()

        # Track history of fitness for plotting
        history_fitness = np.zeros((max_iter + 1, n_crows), dtype=float)
        history_fitness[0] = fitness

        # Main optimization loop
        for iter in range(1, max_iter + 1):
            new_positions = positions.copy()
            new_fitness = fitness.copy()
            for crow_idx in range(n_crows):
                random_crow_idx = np.random.randint(n_crows)
                if np.random.rand() >= awareness_prob:
                    # Move toward another crow's memory
                    diff = crow_position_memory[random_crow_idx] - positions[crow_idx]
                    rand_factor = np.random.rand(dim)
                    new_pos = positions[crow_idx] + flight_length * rand_factor * diff
                else:
                    # Perform random exploration
                    new_pos = np.random.uniform(bounds[0], bounds[1], dim)

                # Ensure new position is valid
                new_pos = self.normalize_to_slot_sum(new_pos, free_slots)

                # Evaluate fitness of the new position
                # The positions to evaluate will include the running kernels and the schedulable kernels (for the model prediction)
                kernels_to_evaluate = self.build_candidate_schedule(running_kernels, schedulable_kernels, new_pos)
                # Compute the fitness of the new position
                new_fit = self.obj_func(running_kernels, kernels_to_evaluate, cpu_usage)

                # Update memory if better
                if new_fit < crow_fitness_memory[crow_idx]:
                    crow_position_memory[crow_idx] = new_pos
                    crow_fitness_memory[crow_idx] = new_fit

                new_positions[crow_idx] = new_pos
                new_fitness[crow_idx] = new_fit

            # Update current population
            positions = new_positions
            fitness = new_fitness
            iter_best_idx = np.argmin(fitness)

            # Update global best if found
            if fitness[iter_best_idx] < self.best_score:
                self.best_score = fitness[iter_best_idx]
                self.best_position = positions[iter_best_idx].copy()
                self.best_crow_idx = iter_best_idx

            # Save this iteration's positions and fitness
            history_positions[iter] = positions
            history_fitness[iter] = fitness

        self.history_positions = history_positions
        self.history_fitness = history_fitness

        return self.best_position

    def model_optimization_function(self, running_kernels: np.ndarray, schedulable_candidate: np.ndarray, cpu_usage: dict = None):
        """
        TODO
        ideas (aplicar un factor para cada una y luego hacer análisis sensibilidad):
        - Evaluar impacto sobre cada kernel en ejecución
        - Evaluar impacto en kernels schedulables (por si otro set se vería menos afectado y es más conveniente)
        - Penalizar que no se ejecuten kernels schedulizables (puesto que se van a limitar a 2/3 opciones) (diversity (std kernel counts))
        - Penalizar que se dejen slots vacíos (num_slots - sum(kernel_counts))
        -
        """

        #
        # Evaluate impact of new scheduling candidate in running kernels
        #

        # Create an empty configuration
        empty_configuration = dict.fromkeys(self.kernel_names, 0)

        # print(f"Empty Configuration: {empty_configuration}")

        # Create configuration of running kernels
        running_configuration = dict(zip(self.kernel_names, running_kernels))

        # print(f"Running Configuration: {running_configuration}")

        # Create configuration inclusing schedulable kernels
        schedulable_configuration = dict(zip(self.kernel_names, schedulable_candidate))

        # print(f"Schedulable Configuration: {schedulable_configuration}")

        interaction_on_running_kernels = []
        for kernel_idx in running_kernels.nonzero()[0]:

            # Create running queue feature
            feature = {**cpu_usage, **running_configuration}
            feature["Main"] = kernel_idx

            # print(f"Feature: {feature}")

            # Predict execution time of each kernel in the running queue (without schedulable kernels)
            actual_configuration_predicted_time = round(self.models[-1].predict_one(feature), 3)
            # print(f"Actual Configuration Predicted Time: {actual_configuration_predicted_time}")

            # Create schedulable candidate feature
            feature = {**cpu_usage, **schedulable_configuration}
            feature["Main"] = kernel_idx
            # print(f"Feature: {feature}")

            # Predict execution time of each kernel in the running queue (with schedulable kernels)
            candidate_configuration_predicted_time = round(self.models[-1].predict_one(feature), 3)
            # print(f"Candidate Configuration Predicted Time: {candidate_configuration_predicted_time}")

            # Compute the relative difference in execution time of each kernel in the running queue
            relative_difference = (candidate_configuration_predicted_time - actual_configuration_predicted_time) / actual_configuration_predicted_time

            # Accumulate the interaction of each kernel in the running queue
            interaction_on_running_kernels.append(relative_difference)


        #
        # Evaluate impact of new scheduling candidate in schedulable kernels
        #

        # Get the set of schedulable kernels
        schedulable_kernels = schedulable_candidate - running_kernels

        interaction_on_schedulable_kernels = []
        gain = []
        for kernel_idx in schedulable_kernels.nonzero()[0]:

            # Create feature for each schedulable kernel alone
            feature = {**cpu_usage, **empty_configuration}
            feature[self.kernel_names[kernel_idx]] = schedulable_kernels[kernel_idx]
            feature["Main"] = kernel_idx
            # print(f"Feature: {feature}")

            # Predict execution time of each kernel in the schedulable set (when alone)
            alone_predicted_time = round(self.models[-1].predict_one(feature), 3)
            # print(f"Alone Predicted Time: {alone_predicted_time}")

            # Create schedulable candidate feature
            feature = {**cpu_usage, **schedulable_configuration}
            feature["Main"] = kernel_idx
            # print(f"Feature: {feature}")

            # Predict execution time of each kernel in the schedulable set (when in the candidate configuration)
            candidate_configuration_predicted_time = round(self.models[-1].predict_one(feature), 3)
            # print(f"Candidate Configuration Predicted Time: {candidate_configuration_predicted_time}")

            # Compute the relative difference in execution time of each kernel in the schedulable set
            relative_difference = (candidate_configuration_predicted_time - alone_predicted_time) / alone_predicted_time

            # TODO: introduce a factor to penalize no parallelism (divide by cus?)
            gain.append((schedulable_kernels[kernel_idx] / self.num_kernels) * (1 / candidate_configuration_predicted_time))

            # Accumulate the interaction of each kernel in the running queue
            interaction_on_schedulable_kernels.append(relative_difference)


        #
        # Fairness (promote diversity, otherwise fast kernels will always be used)
        #

        # TODO

        #
        # Promote parallelism (with the above division?)
        #

        # TODO

        #
        # Penalize empty slots (maximize resource utilization)
        #

        running_kernel_fitness = np.mean(interaction_on_running_kernels) if len(interaction_on_running_kernels) > 0 else 0
        schedulable_kernel_fitness = np.mean(interaction_on_schedulable_kernels) if len(interaction_on_schedulable_kernels) > 0 else 0
        gain_fitness = np.mean(gain) if len(gain) > 0 else 0
        balance_penalty = np.std(schedulable_kernels)

        fitness = 2 * np.mean(running_kernel_fitness) + 2 * np.mean(schedulable_kernel_fitness) - 1 * np.mean(gain_fitness) + 0.5 * balance_penalty


        # print(f"Schedule Candidate: {schedulable_candidate}")
        # print(f"Running Interaction: {running_kernel_fitness}")
        # print(f"Schedulable Interaction: {schedulable_kernel_fitness}")
        # print(f"Gain: {gain_fitness}")
        # print(f"Balance Penalty: {balance_penalty}")
        # print(f"Fitness: {fitness}")

        return fitness

    def plot_crow(self, running_kernels: np.ndarray, schedulable_kernels: list, crow_id):
        """
        Plot heatmap of kernel assignments and fitness evolution for a specific crow.

        Args:
            crow_id (int): ID of the crow to plot.
        """
        if self.history_positions is None or self.history_fitness is None:
            raise ValueError("No history found. Run the algorithm first.")

        num_iters, _, num_kernels = self.history_positions.shape

        # Flatten the history_positions to apply the build_candidate_schedule function (-1: 1D array)
        flat_history_positions = self.history_positions.reshape(-1, self.history_positions.shape[-1])
        # Apply the build_candidate_schedule function to each row of the flattened history
        flat_updated_history_positions = np.array([
            self.build_candidate_schedule(running_kernels, schedulable_kernels, pos)
            for pos in flat_history_positions
            ])
        # Reshape the updated history back to its original shape (*self.history_positions.shape[:-1]: all but last dim, -1: infer last dim)
        updated_history_positions = flat_updated_history_positions.reshape(*self.history_positions.shape[:-1], -1)

        # Extract crow data for the specified crow_id
        crow_data = updated_history_positions[:, crow_id, :].T.astype(int)

        # Extract fitness data for the specified crow_id
        num_iters, _ = self.history_fitness.shape
        fitness = self.history_fitness[:, crow_id]
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

    # TODO: Save each prediction so it is not recomputed

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

    print(f"Kernel Counts: {kernel_counts}")
    print(f"Base Cost: {base_cost}")
    print(f"Penalty Distinct: {penalty_distinct}")
    print(f"Penalty Power of Two: {penalty_power_of_two}")

    return base_cost + penalty_distinct + penalty_power_of_two


if __name__ == "__main__":

    # Open models files
    with open("models/adapt_models.pkl", 'rb') as file:
        online_models_list = pickle.load(file)

    random_seed = 42
    np.random.seed(random_seed)

    num_kernels = 11
    num_slots = 8
    scheduler = CrowSearchAlgorithm(num_kernels, num_slots, obj_func=None, kernel_names=["aes", "bulk", "crs", "kmp", "knn", "merge", "nw", "queue", "stencil2d", "stencil3d", "strided"], models=online_models_list)

    # No running kernels example
    # scheduler.run_no_running_kernels(n_crows=40, max_iter=100, awareness_prob=0.4, flight_length=1.5)
    # scheduler.plot_crow_no_running_kernels(crow_id=scheduler.best_crow_idx)

    # Equivalent to the previous example
    # running_kernels = np.zeros(num_kernels, dtype=int)
    # schedulable_kernels=[0,1,2,3,4,5,6,7,8,9,10]
    # scheduler.run(n_crows=40, max_iter=100, awareness_prob=0.4, flight_length=1.5,
    #                    running_kernels=running_kernels,
    #                    schedulable_kernels=schedulable_kernels)
    # scheduler.plot_crow(running_kernels=running_kernels, schedulable_kernels=schedulable_kernels, crow_id=scheduler.best_crow_idx)

    running_kernels = np.zeros(num_kernels, dtype=int)
    running_kernels[[1,4]] = 1
    schedulable_kernels=[5,6,10]
    for i in range(2):
        print(scheduler.run_standalone(n_crows=40, max_iter=150, awareness_prob=0.4, flight_length=5.5,
                    running_kernels=running_kernels,
                    schedulable_kernels=schedulable_kernels,
                    cpu_usage={"user":40, "kernel":20, "idle":40}))
        scheduler.plot_crow(running_kernels=running_kernels, schedulable_kernels=schedulable_kernels, crow_id=scheduler.best_crow_idx)

        print("Best Kernel Counts:", scheduler.best_position)
        print("Best Score:", scheduler.best_score)
        print("Best Crow:", scheduler.best_crow_idx)
