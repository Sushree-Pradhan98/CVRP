import numpy as np
import random
from scipy.spatial import distance_matrix

def load_cvrp_data(file_path):
    locations = []
    demands = []
    vehicle_capacity = 0
    num_customers = 0

    with open(file_path, 'r') as file:
        section = None
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'NODE_COORD_SECTION':
                section = 'nodes'
                continue
            elif parts[0] == 'DEMAND_SECTION':
                section = 'demands'
                continue
            elif parts[0] == 'DEPOT_SECTION':
                section = 'depot'
                continue
            elif parts[0] == 'CAPACITY':
                vehicle_capacity = int(parts[-1])
                continue
            elif parts[0] == 'EOF':
                break

            if section == 'nodes':
                locations.append((int(parts[1]), int(parts[2])))
            elif section == 'demands':
                if len(demands) == 0:
                    num_customers = len(locations)
                    demands = [0] * num_customers
                demands[int(parts[0]) - 1] = int(parts[1])

    return locations, demands, vehicle_capacity

class RandomCVRP:
    def __init__(self, locations, demands, vehicle_capacity, max_evaluations=10000):
        self.locations = np.array(locations)
        self.demands = np.array(demands)
        self.vehicle_capacity = vehicle_capacity
        self.distance_matrix = distance_matrix(self.locations, self.locations)
        self.num_customers = len(locations) - 1
        self.max_evaluations = max_evaluations
        self.evaluations = 0

        # Convergence tracking
        self.history_best = []
        self.history_avg = []
        self.history_worst = []

    def total_distance(self, solution):
        self.evaluations += 1
        total_dist = 0
        for route in solution:
            if route:
                total_dist += self.distance_matrix[0, route[0]]
                for i in range(len(route) - 1):
                    total_dist += self.distance_matrix[route[i], route[i + 1]]
                total_dist += self.distance_matrix[route[-1], 0]
        return total_dist

    def random_solution(self):
        customers = list(range(1, self.num_customers + 1))
        random.shuffle(customers)
        solution = []
        while customers:
            route = []
            capacity = self.vehicle_capacity
            while customers:
                next_customer = customers[0]
                if self.demands[next_customer] <= capacity:
                    route.append(next_customer)
                    capacity -= self.demands[next_customer]
                    customers.pop(0)
                else:
                    break
            solution.append(route)
        return solution

    def run(self):
        best_solution = None
        best_distance = float('inf')

        while self.evaluations < self.max_evaluations:
            candidates = [self.random_solution() for _ in range(10)]
            distances = [self.total_distance(sol) for sol in candidates]

            # Log convergence
            self.history_best.append(min(distances))
            self.history_avg.append(sum(distances) / len(distances))
            self.history_worst.append(max(distances))

            min_index = np.argmin(distances)
            if distances[min_index] < best_distance:
                best_solution = candidates[min_index]
                best_distance = distances[min_index]

        return best_solution, best_distance

    def plot_convergence(self):
        import matplotlib.pyplot as plt
        iterations = range(len(self.history_best))
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, self.history_best, label='Best', color='blue')
        plt.plot(iterations, self.history_avg, label='Average', color='orange', linestyle='--')
        plt.plot(iterations, self.history_worst, label='Worst', color='green', linestyle=':')
        plt.fill_between(iterations, self.history_best, alpha=0.2, color='blue')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.title('Fitness Convergence: Random Search')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
