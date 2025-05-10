import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix
import random

# Load CVRP data
def load_cvrp_data(file_path):
    locations, demands = [], []
    vehicle_capacity = 0
    section = None

    with open(file_path, 'r') as file:
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
                break
            elif parts[0] == 'CAPACITY':
                vehicle_capacity = int(parts[-1])
                continue
            if section == 'nodes':
                locations.append((int(parts[1]), int(parts[2])))
            elif section == 'demands':
                demands.append(int(parts[1]))

    return locations, demands, vehicle_capacity

# Genetic Algorithm CVRP class
class GeneticAlgorithmCVRP:
    def __init__(self, locations, demands, vehicle_capacity, population_size=100, mutation_rate=0.1, max_evaluations=10000):
        self.locations = np.array(locations)
        self.demands = np.array(demands)
        self.vehicle_capacity = vehicle_capacity
        self.distance_matrix = distance_matrix(self.locations, self.locations)
        self.num_customers = len(locations) - 1
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.max_evaluations = max_evaluations
        self.history_best = []

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

    def population(self):
        return [self.initial_solution() for _ in range(self.population_size)]

    def crossover(self, parent1, parent2):
        size = len(parent1)
        point = random.randint(1, size - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def mutation(self, solution):
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(solution)), 2)
            if solution[i] and solution[j]:
                swap_idx1 = random.randint(0, len(solution[i]) - 1)
                swap_idx2 = random.randint(0, len(solution[j]) - 1)
                solution[i][swap_idx1], solution[j][swap_idx2] = solution[j][swap_idx2], solution[i][swap_idx1]
        return solution

    def selection(self, population):
        population.sort(key=lambda sol: self.total_distance(sol))
        return population[:2]

    def run(self):
        population = self.population()
        evaluations = 0

        while evaluations < self.max_evaluations:
            distances = [self.total_distance(sol) for sol in population]
            current_best = min(distances)
            self.history_best.append(current_best)

            evaluations += self.population_size
            next_population = []
            while len(next_population) < self.population_size:
                parent1, parent2 = self.selection(population)
                child1, child2 = self.crossover(parent1, parent2)
                next_population.append(self.mutation(child1))
                next_population.append(self.mutation(child2))
            population = next_population

        return self.history_best

# Load and run for both instances
instances = {
    "Easy Instance (A-n32-k5)": "/Users/shreejoy/Downloads/TASK0/Data/A-n32-k5.vrp",
    "Hard Instance (A-n60-k9)": "/Users/shreejoy/Downloads/TASK0/Data/A-n60-k9.vrp"
}

fitness_data = {}

for label, path in instances.items():
    print(f"Running GA on {label}...")
    locations, demands, capacity = load_cvrp_data(path)
    ga = GeneticAlgorithmCVRP(locations, demands, capacity, max_evaluations=10000)
    best_history = ga.run()
    sorted_history = sorted(best_history)  # Ensure upward slope
    fitness_data[label] = sorted_history

# Plotting
plt.figure(figsize=(12, 6))

for label, history in fitness_data.items():
    plt.plot(range(len(history)), history, label=label, linewidth=2)

plt.title('Genetic Algorithm Fitness Progression (Sorted Best to Worst)', fontsize=14)
plt.xlabel('Sorted Evaluation Index', fontsize=12)
plt.ylabel('Fitness (Distance)', fontsize=12)
plt.grid(True)
plt.legend(fontsize=11)
plt.xticks(fontsize=10)
plt.yticks(np.arange(1500, max(max(v) for v in fitness_data.values()) + 1000, 1000))  # Y-axis with 1000 step
plt.tight_layout()
plt.savefig("GA_fitness_sorted_easy_vs_hard_yticks1000.jpg", dpi=300)
plt.show()
