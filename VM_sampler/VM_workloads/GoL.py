import numpy as np
import time

# Define the size of the grid
grid_size = 500

# Initialize the grid with random states (0 or 1)
grid = np.random.randint(2, size=(grid_size, grid_size))

iteration = 0

# Function to update the grid based on the rules of Conway's Game of Life
def update_grid(grid):
    new_grid = np.copy(grid)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            # Count the live neighbors
            live_neighbors = np.sum(grid[max(0, i-1):min(grid_size, i+2), max(0, j-1):min(grid_size, j+2)]) - grid[i, j]
            # Apply the rules of the game
            if grid[i, j] == 1:
                if live_neighbors < 2 or live_neighbors > 3:
                    new_grid[i, j] = 0
            else:
                if live_neighbors == 3:
                    new_grid[i, j] = 1
    return new_grid

# Infinite loop to simulate the game
while True:
    grid = update_grid(grid)
    iteration += 1
    print(f"Iteration {iteration}: Updated grid state")
    time.sleep(0.05)
