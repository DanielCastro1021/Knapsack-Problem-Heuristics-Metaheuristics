import os
import time
import numpy as np
from tabulate import tabulate


class LocalSearch:
    def __init__(self, capacity, values, weights, time_limit):
        self.capacity, self.values, self.weights, self.time_limit = int(
            capacity), values, weights, time_limit

    def initial_solution(self):
        '''Cria um solução inicial com items mais valiosos'''
        sorted_weights = np.sort(self.weights)

        current_solution_weight = 0
        i = len(self.weights) - 1
        number_of_items = 0

        while current_solution_weight <= self.capacity:
            current_solution_weight += sorted_weights[i]
            i -= 1
            number_of_items += 1

        initial_solution = np.zeros(len(self.values), dtype=int)

        most_valuable_times_indexs = np.argsort(self.values)[-number_of_items:]

        initial_solution[most_valuable_times_indexs] = 1

        return initial_solution

    def evaluate(self, solution):
        np_solution = np.array(solution)
        np_values = np.array(self.values)
        np_weights = np.array(self.weights)

        totalValue = np.dot(np_solution, np_values)
        totalWeight = np.dot(np_solution, np_weights)

        if totalWeight > self.capacity:
            return [-1, -1]

        return [totalValue, totalWeight]

    def neighborhood(self, solution):
        '''Faz um copia da solução e remove os items e coloca os vizinhos deste items'''
        nb = []
        for i in range(0, len(self.values)):
            nb.append(np.copy(solution))
            if nb[i][i] == 1:
                nb[i][i] = 0
            else:
                nb[i][i] = 1
        return nb

    def run(self):
        start_time = time.perf_counter()
        time_limit = start_time+self.time_limit
        solutionsChecked = 0

        current_solution_items = self.initial_solution()
        current_solution = self.evaluate(current_solution_items)

        best_solution_items = np.copy(current_solution_items)
        best_solution = np.copy(current_solution)

        current_time = time.perf_counter()
        done = 0
        while current_time < time_limit and done == 0:
            Neighborhood = self.neighborhood(current_solution_items)

            for s in Neighborhood:
                solutionsChecked += 1

                evaluated_solution = self.evaluate(s)

                if evaluated_solution[0] > best_solution[0]:
                    best_solution_items = np.copy(s)
                    best_solution = np.copy(evaluated_solution)
                if list(best_solution) == list(current_solution):
                    done = 1
                else:
                    current_solution_items = np.copy(best_solution_items)
                    current_solution = np.copy(best_solution)

        time_to_solve = round(time.perf_counter()-start_time, 3)

        return best_solution[0], solutionsChecked, format(time_to_solve, '.3f')


def get_optimum_solution(file):
    with open(file, "r") as f:
        lines = f.readlines()
        return int(lines[0].strip().split()[0])


def get_solution(file, time_limit):
    capacity, values, weights = 0, [], []
    with open(file, "r") as f:
        lines = f.readlines()
        capacity = float(lines[0].strip().split()[1])
        for line in lines[1:]:
            try:
                value, weight = line.strip().split()
                values.append(int(value))
                weights.append(int(weight))
            except:
                pass

    local_search = LocalSearch(
        capacity, values, weights, time_limit)

    return local_search.run()


def main(folder, current_dataset, time_limit):
    files = os.listdir(os.path.join(folder, current_dataset))
    results = []
    for file in files:
        optimum_solution = get_optimum_solution(os.path.join(
            folder, current_dataset+"-optimum", file))

        solution, solutionsChecked, time = get_solution(os.path.join(
            folder, current_dataset, file), time_limit)

        gap = round(((optimum_solution-solution)/optimum_solution)*100, 2)

        results.append([file,
                       optimum_solution, solution, gap, solutionsChecked, time])

    print(tabulate(results, headers=[
          "File Name", "Z*", "Z", "Gap %", "Nº Solutions Checked", f"Time (s)"]))


if __name__ == "__main__":
    folder, datasets = "KP-instances", {
        1: "low-dimensional", 2: "large_scale"}

    current_dataset = datasets.get(2)
    time_limit = 10

    main(folder, current_dataset, time_limit)
