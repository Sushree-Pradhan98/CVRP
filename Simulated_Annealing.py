import numpy as np
import random
import math
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

def load_cvrp_data(file_path):
    locations = []
    demands = []
    vehicle_capacity = 0

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
                demands.append(int(parts[1]))

    return locations, demands, vehicle_capacity

class SimulatedAnnealingCVRP:
    def __init__(self, locations, demands, vehicle_capacity,
                 initial_temp=1000, cooling_rate=0.99, min_temp=1, no_improvement_limit=100):
        self.locations = np.array(locations)
        self.demands = np.array(demands)
        self.vehicle_capacity = vehicle_capacity
        self.distance_matrix = distance_matrix(self.locations, self.locations)
        self.num_customers = len(locations) - 1
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.no_improvement_limit = no_improvement_limit
        self.history_best = []
        self.history_avg = []
        self.history_worst = []

    def total_distance(self, solution):
        total_dist = 0
        for route in solution:
            if route:
                total_dist += self.distance_matrix[0, route[0]]
                for i in range(len(route) - 1):
                    total_dist += self.distance_matrix[route[i], route[i + 1]]
                total_dist += self.distance_matrix[route[-1], 0]
        return total_dist

    def initial_solution(self):
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

    def random_neighbor(self, solution):
        new_solution = [list(route) for route in solution]
        if len(new_solution) > 1:
            i, j = random.sample(range(len(new_solution)), 2)
            if new_solution[i] and new_solution[j]:
                swap_idx1 = random.randint(0, len(new_solution[i]) - 1)
                swap_idx2 = random.randint(0, len(new_solution[j]) - 1)
                new_solution[i][swap_idx1], new_solution[j][swap_idx2] = \
                    new_solution[j][swap_idx2], new_solution[i][swap_idx1]
        return new_solution

    def simulated_annealing(self):
        current_solution = self.initial_solution()
        current_distance = self.total_distance(current_solution)
        best_solution = current_solution
        best_distance = current_distance
        temp = self.initial_temp

        no_improvement = 0

        while temp > self.min_temp and no_improvement < self.no_improvement_limit:
            neighbors = [self.random_neighbor(current_solution) for _ in range(10)]
            distances = [self.total_distance(n) for n in neighbors]
            new_solution = neighbors[np.argmin(distances)]
            new_distance = min(distances)

            self.history_best.append(best_distance)
            self.history_avg.append(sum(distances) / len(distances))
            self.history_worst.append(max(distances))

            if new_distance < current_distance or random.random() < math.exp((current_distance - new_distance) / temp):
                current_solution = new_solution
                current_distance = new_distance
                if new_distance < best_distance:
                    best_solution = new_solution
                    best_distance = new_distance
                    no_improvement = 0
                else:
                    no_improvement += 1
            else:
                no_improvement += 1

            temp *= self.cooling_rate

        return best_solution, best_distance

    def plot_convergence(self):
        generations = range(len(self.history_best))
        plt.figure(figsize=(10, 6))
        plt.plot(generations, self.history_best, label='Best', color='blue')
        plt.plot(generations, self.history_avg, label='Average', color='orange', linestyle='--')
        plt.plot(generations, self.history_worst, label='Worst', color='green', linestyle=':')
        plt.fill_between(generations, self.history_best, alpha=0.2, color='blue')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness (Total Distance)')
        plt.title('Fitness Convergence: Simulated Annealing')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# Example usage
file_path = "/Users/shreejoy/Downloads/TASK0/Data/A-n37-k6.vrp"
locations, demands, vehicle_capacity = load_cvrp_data(file_path)

sa = SimulatedAnnealingCVRP(locations, demands, vehicle_capacity)
solution, cost = sa.simulated_annealing()
print("Best total distance:", cost)
sa.plot_convergence()
