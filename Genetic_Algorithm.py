import numpy as np
import random
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import os
import csv


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


def decode_solution(permutation, demands, capacity):
    routes = []
    route = []
    load = 0
    for customer in permutation:
        demand = demands[customer]
        if load + demand <= capacity:
            route.append(customer)
            load += demand
        else:
            routes.append(route)
            route = [customer]
            load = demand
    if route:
        routes.append(route)
    return routes


class GeneticAlgorithmCVRP:
    def __init__(self, locations, demands, vehicle_capacity, population_size=100, mutation_rate=0.1, elitism_count=2, max_evaluations=10000, save_path="/Users/shreejoy/Downloads/TASK0/Graph", result_path="/Users/shreejoy/Downloads/TASK0/Result"):
        self.locations = np.array(locations)
        self.demands = np.array(demands)
        self.vehicle_capacity = vehicle_capacity
        self.distance_matrix = distance_matrix(self.locations, self.locations)
        self.num_customers = len(locations) - 1
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        self.max_evaluations = max_evaluations
        self.save_path = save_path
        self.result_path = result_path
        self.history_best = []
        self.history_avg = []
        self.history_worst = []

    def total_distance(self, permutation):
        routes = decode_solution(permutation, self.demands, self.vehicle_capacity)
        total_dist = 0
        for route in routes:
            if route:
                total_dist += self.distance_matrix[0, route[0]]
                for i in range(len(route) - 1):
                    total_dist += self.distance_matrix[route[i], route[i + 1]]
                total_dist += self.distance_matrix[route[-1], 0]
        return total_dist

    def initial_population(self):
        return [random.sample(range(1, self.num_customers + 1), self.num_customers) for _ in range(self.population_size)]

    def tournament_selection(self, population, k=5):
        selected = random.sample(population, k)
        selected.sort(key=lambda x: self.total_distance(x))
        return selected[0]

    def ordered_crossover(self, parent1, parent2):
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[start:end + 1] = parent1[start:end + 1]

        pointer = 0
        for gene in parent2:
            if gene not in child:
                while child[pointer] is not None:
                    pointer += 1
                child[pointer] = gene
        return child

    def swap_mutation(self, permutation):
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(permutation)), 2)
            permutation[i], permutation[j] = permutation[j], permutation[i]
        return permutation

    def run(self):
        population = self.initial_population()
        evaluations = 0
        while evaluations < self.max_evaluations:
            population.sort(key=lambda x: self.total_distance(x))
            best = self.total_distance(population[0])
            avg = sum(self.total_distance(ind) for ind in population) / len(population)
            worst = self.total_distance(population[-1])
            self.history_best.append(best)
            self.history_avg.append(avg)
            self.history_worst.append(worst)

            next_generation = population[:self.elitism_count]
            while len(next_generation) < self.population_size:
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                child = self.ordered_crossover(parent1, parent2)
                child = self.swap_mutation(child)
                next_generation.append(child)

            population = next_generation
            evaluations += self.population_size

        best_solution = min(population, key=lambda x: self.total_distance(x))
        return best_solution

    def plot_convergence(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.history_best, label='Best')
        plt.plot(self.history_avg, label='Average', linestyle='--')
        plt.plot(self.history_worst, label='Worst', linestyle=':')
        plt.fill_between(range(len(self.history_best)), self.history_best, alpha=0.2)
        plt.xlabel('Evaluation Step')
        plt.ylabel('Total Distance')
        plt.title('GA Convergence for CVRP')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        os.makedirs(self.save_path, exist_ok=True)
        plt.savefig(os.path.join(self.save_path, "cvrp_convergence.png"))
        plt.close()

    def save_results(self, best_perm):
        best_routes = decode_solution(best_perm, self.demands, self.vehicle_capacity)
        best_distance = self.total_distance(best_perm)

        os.makedirs(self.result_path, exist_ok=True)
        result_file = os.path.join(self.result_path, "best_solution.csv")

        with open(result_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Route Index", "Route"])
            for idx, route in enumerate(best_routes):
                writer.writerow([idx + 1, " -> ".join(map(str, route))])
            writer.writerow([])
            writer.writerow(["Total Distance", best_distance])


if __name__ == "__main__":
    file_path = "/Users/shreejoy/Downloads/TASK0/Data/A-n37-k6.vrp"
    save_path = "/Users/shreejoy/Downloads/TASK0/Graph"
    result_path = "/Users/shreejoy/Downloads/TASK0/Result"

    locations, demands, vehicle_capacity = load_cvrp_data(file_path)
    ga = GeneticAlgorithmCVRP(locations, demands, vehicle_capacity, max_evaluations=10000, save_path=save_path, result_path=result_path)

    best_perm = ga.run()
    print("Best total distance:", ga.total_distance(best_perm))
    ga.save_results(best_perm)
    ga.plot_convergence()
