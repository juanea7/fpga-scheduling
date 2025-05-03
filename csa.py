import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
import seaborn as sns


def normalize_to_slot_sum(kernel_counts, num_slots):
    """
    Adjusts the vector of kernel counts to ensure:
    - All values are >= 0
    - All values are integers
    - The sum of values equals num_slots
    - The distribution is as even as possible (might not be the best solution)
    """
    kernel_counts = np.clip(np.round(kernel_counts), 0, None)
    total = int(np.sum(kernel_counts))

    # Reduce total if over budget
    while total > num_slots:
        idx = np.argmax(kernel_counts)
        kernel_counts[idx] -= 1
        total -= 1

    # Increase total if under budget
    while total < num_slots:
        idx = np.argmin(kernel_counts)
        kernel_counts[idx] += 1
        total += 1

    return kernel_counts.astype(int)


def crow_search_kernel_counts(
    obj_func,
    num_kernels,
    num_slots,
    n_crows=20,
    max_iter=100,
    awareness_prob=0.1,
    flight_length=2
):
    """
    Performs task scheduling optimization using the Crow Search Algorithm.
    Each crow suggests a combination of kernel counts that fill the available slots.
    """
    # Variables to optimize (kernel counts)
    dim = num_kernels
    # Bound for each variable is [0, num_slots]
    bounds = (0, num_slots)

    # Initialize random positions and normalize them
    positions = np.random.randint(0, num_slots + 1, (n_crows, dim))
    for crow_idx in range(n_crows):
        positions[crow_idx] = normalize_to_slot_sum(positions[crow_idx], num_slots)
    memory = positions.copy()  # Each crow remembers its own best solution

    # Track position history across iterations (for visualization)
    history = np.zeros((max_iter + 1, n_crows, num_kernels), dtype=int)
    history[0] = positions

    # Evaluate initial solutions
    fitness = np.array([obj_func(pos) for pos in positions])
    best_fitness_idx = np.argmin(fitness)
    best_position = positions[best_fitness_idx].copy()
    best_score = fitness[best_fitness_idx]

    # Main optimization loop
    for iter in range(1, max_iter + 1):
        new_positions = positions.copy()

        for crow_idx in range(n_crows):
            random_crow_idx = np.random.randint(n_crows)
            if np.random.rand() >= awareness_prob:
                # Follow another crow's memory
                diff = memory[random_crow_idx] - positions[crow_idx]
                rand_factor = np.random.rand(dim)
                new_pos = positions[crow_idx] + flight_length * rand_factor * diff
            else:
                # Perform a random search
                new_pos = np.random.uniform(bounds[0], bounds[1], dim)

            # Normalize position to ensure it fits within the slot budget
            new_pos = normalize_to_slot_sum(new_pos, num_slots)
            new_fit = obj_func(new_pos)

            # Update memory if improvement found
            if new_fit < obj_func(memory[crow_idx]):
                memory[crow_idx] = new_pos

            new_positions[crow_idx] = new_pos

        positions = new_positions
        fitness = np.array([obj_func(pos) for pos in positions])
        iteration_best_crow_idx = np.argmin(fitness)

        # Update global best if found
        if fitness[iteration_best_crow_idx] < best_score:
            best_score = fitness[iteration_best_crow_idx]
            best_position = positions[iteration_best_crow_idx].copy()
            best_crow_idx = iteration_best_crow_idx

        # Save this iteration's positions
        history[iter] = positions

    return best_position, best_score, best_crow_idx, history

def power_performance_model(kernel_counts):
    """
    Placeholder for a power/performance model.
    """
    total_slots = np.sum(kernel_counts)
    num_active_kernels = np.count_nonzero(kernel_counts)

    # --- Simulate power cost ---
    # More kernel types = more static/dynamic diversity = more power
    power_cost = num_active_kernels * 5

    # --- Simulate performance (time) ---
    # Assume performance improves with parallelism,
    # but suffers if the slot use is highly unbalanced (std dev high)
    balance_penalty = np.std(kernel_counts)
    inverse_parallelism = 1 / (np.max(kernel_counts) + 1e-3)  # avoid div by zero
    performance_cost = 10 * inverse_parallelism + 2 * balance_penalty

    # --- Base cost is sum of power and performance components ---
    base_cost = power_cost + performance_cost

    return base_cost

def penalized_model(kernel_counts):
    """
    The cost function for the Crow Search Algorithm.
    - Base cost should come from power/performance model
    - Soft penalty for >4 kernel types
    - Soft penalty for non-power-of-2 kernel counts
    """
    base_cost = power_performance_model(kernel_counts)

    # Soft penalty: >4 kernel types
    num_active_kernels = np.count_nonzero(kernel_counts)
    penalty_distinct = 10 * max(0, num_active_kernels - 4)

    # Soft penalty: non-power-of-2 kernel counts
    penalty_power_of_two = 0
    for k in kernel_counts:
        if k != 0 and not (k & (k - 1)) == 0:
            penalty_power_of_two += 5

    return base_cost + penalty_distinct + penalty_power_of_two


def plot_crow(history, crow_id, obj_func, kernel_names=None):
    """
    - Heatmap of kernel assignments per iteration (top)
    - Line plot of fitness evolution (bottom)
    """
    num_iters, _, num_kernels = history.shape
    crow_data = history[:, crow_id, :].T.astype(int)
    fitness = np.array([obj_func(history[t, crow_id, :]) for t in range(num_iters)])
    best_iter = int(np.argmin(fitness))

    # Set up grid with 3 rows: heatmap, colorbar, fitness
    fig = plt.figure(figsize=(14, 6))
    spec = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[0.15, 3, 1], width_ratios=[0.96, 0.04], hspace=0.3)

    cbar_ax = fig.add_subplot(spec[0, 0])  # horizontal colorbar
    ax1 = fig.add_subplot(spec[1, 0])  # heatmap
    ax2 = fig.add_subplot(spec[2, 0], sharex=ax1)  # fitness plot

    # --- Heatmap ---
    sns.heatmap(
        crow_data,
        ax=ax1,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        cbar_ax=cbar_ax,
        cbar_kws={"orientation": "horizontal", "label": "Slots Used"},
        xticklabels=np.arange(num_iters),
        yticklabels=kernel_names or [f"Kernel {i}" for i in range(num_kernels)]
    )
    ax1.axvline(x=best_iter, color='red', linestyle='--', lw=2)
    ax1.set_ylabel("Kernel Type")
    ax1.set_title(f"Crow #{crow_id} - Kernel Assignment and Fitness Over Time")

    # --- Fitness plot ---
    ax2.plot(np.arange(num_iters), fitness, marker='o', color='black', label="Fitness")
    ax2.axvline(x=best_iter, color='red', linestyle='--', lw=2, label=f"Best Iter ({best_iter})")
    ax2.set_ylabel("Fitness")
    ax2.set_xlabel("Iteration")
    ax2.grid(True)
    ax2.legend()

    # plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    num_kernels = 11     # Number of different kernel types
    num_slots = 8       # Total FPGA slots available

    # Run the crow search algorithm
    best_combo, best_score, best_crow_idx, history = crow_search_kernel_counts(
        obj_func=penalized_model,
        num_kernels=num_kernels,
        num_slots=num_slots,
        n_crows=40,
        max_iter=100,
        awareness_prob=0.4,
        flight_length=1.5
    )

    print("Best Kernel Counts:", best_combo)
    print("Best Score:", best_score)
    print("Best Crow:", best_crow_idx)

    # Visualize the behavior of one crow over time
    plot_crow(
        history,
        crow_id=best_crow_idx,
        obj_func=penalized_model,
        kernel_names=[kernel for kernel in "ABCDEFGHIJK"[:num_kernels]]
    )
