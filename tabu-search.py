import os
import time
import numpy as np
from tabulate import tabulate
import pdb
import logging
from datetime import datetime

import numpy as np


class TabuSearch:
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

    def neighborhood(self, solution, k=1):
        '''Faz um copia da solução e remove os items e coloca os vizinhos deste items'''
        nb = []
        for i in range(0, len(self.values)):
            nb.append(np.copy(solution))
            for j in range(k):
                if (i+j) > 0:
                    a = i+j-len(self.values)
                else:
                    a = i+j

                if nb[i][a] == 1:
                    nb[i][a] = 0
                else:
                    nb[i][a] = 1
        return nb

    def tabu_criteria(self, new_solution, old_solution):
        # find the index where flip occurred
        ind = np.where(new_solution != old_solution)[0]

        if len(ind) > 1:
            print("WARNING found more than one flip")

        return ind[0]

    def aspiration_criteria(self, curr_value=0, val_history=[], neighborhood=[], values=[]):
        # If a neighborhood is provided (which means all solutions are tabu)
        # pick the best solution
        if len(neighborhood) > 0:
            sol = neighborhood[np.nanargmax(values)]
            return sol

        else:
            # if the current solution is higher than the best found so far
            # accept it even if it's tabu
            accept = 0
            try:
                max_val_so_far = np.nanmax(np.array(val_history))
                if curr_value > max_val_so_far:
                    accept = 1
            except:
                pass

            return accept

    def tabu_active(self, sMem, sMemVal, Neighborhood, cVal, valHist):

        # indices of tabu active elements
        active_ind = np.where(sMem > 0)[0]

        # A list to hold the tabu unactive neighbors
        newN = []

        # A loop to check whether each solution is tabu active or not
        for N in Neighborhood:

            # A flag to represent the tabu active status of an element
            not_tabu_active = 1

            for i, a in enumerate(active_ind):
                # If there is even one tabu_active member -
                # do not include this solution and,
                # break out of this loop as soon as you find the first tabu active element
                if (N[a] != sMemVal[a]):
                    # unless aspiration criteria is satisfied
                    exception = self.aspiration_criteria(
                        curr_value=cVal, val_history=valHist)

                    if exception:
                        continue
                    else:
                        not_tabu_active = 0
                        break

            # If there are no tabu active members, add this solution
            if not_tabu_active:
                newN.append(N)

        return np.array(newN)

    # Short term memory

    def short_memory(self, update_ind=0, tenure=3, init=False, mem=0, memValue=0, solution=0):
        """ update_ind: index of the element that was flipped
            solution: this is the candidate solution. we want the flipped value
        """

        # if memory is to be initialized, set every element to 0
        if init:
            mem = np.zeros(len(self.values), dtype=int)
            memValue = np.array([2]*len(self.values))

        else:  # else update memory
            # if memory is to be updated but old memory is not provided, raise error
            if (type(mem) == int) | (type(memValue) == int) | (type(solution) == int):
                print(
                    "Need to provide old memory and solution if you want to update memory \n")
                print("Old memory and solution needs to be an array of size n \n")
                raise TypeError

            # update every active element score
            for i, m in enumerate(mem):
                if m > 0:
                    mem[i] -= 1
                    if mem[i] == 0:
                        memValue[i] = 2

            # update memory with new active element
            mem[update_ind] = tenure
            memValue[update_ind] = solution[update_ind]

        return mem, memValue

    def run(self):
        # Set time counters for execution
        start_time = time.perf_counter()
        time_limit = start_time+self.time_limit

        # Set logger configurations
        now = datetime.now()
        log_fname = "log/ts__"+now.strftime("%d-%m-%y__%H:%M")+".log"
        LOG_FORMAT = "%(message)s"
        logging.basicConfig(filename=log_fname,
                            level=logging.DEBUG,
                            format=LOG_FORMAT)
        logger = logging.getLogger()

        solutionsChecked = 0

        # initial solution and evaluation
        current_solution_items = self.initial_solution()
        current_solution = self.evaluate(current_solution_items)

        # memory initialization
        sMem, sMemVal = self.short_memory(init=True)

        # initial neighborhood
        neighborhood = self.neighborhood(current_solution_items)

        # A counter to count number of iterations
        counter = 0

        # keeping track of all the solutions
        solutions = []
        solutions_values = []
        solutions_weights = []

        current_time = time.perf_counter()
        while current_time < time_limit:
            solutionsChecked += 1

            logger.info(
                "\n \n \n -----------ITERATION {}----------- \n".format(counter))
            logger.info("current solution: {} {} \n".format(
                current_solution, current_solution_items))
            logger.info(
                "length of current-neighborhood: {} \n".format(len(neighborhood)))

            # tabu-unactive subset of neighborhood N
            neighborhood_current = self.tabu_active(
                sMem, sMemVal, neighborhood, current_solution[0], solutions_values)

            logger.info(
                "length of sub-neighborhood: {} \n".format(len(neighborhood_current)))

            # Selecting a candidate
            # if all the solutions are tabu:
            if len(neighborhood_current) == 0:
                all_values = [self.evaluate(s)[0] for s in neighborhood]
                solution_items = self.aspiration_criteria(
                    neighborhood=neighborhood, values=all_values)
                solution = self.evaluate(solution_items)
            else:
                # otherwise -
                # Pick the solution with the best value
                # from non-tabu members even if they are non-improving
                solutions_values = [self.evaluate(
                    s)[0] for s in neighborhood_current]
                solution_items = neighborhood_current[np.nanargmax(
                    solutions_values)]
                solution = self.evaluate(solution_items)

            logger.info("candiddate solution: {} {} \n".format(
                solution, solution_items))

            # Finding where the flip occurred
            tabu_ind = self.tabu_criteria(
                solution_items, current_solution_items)

            logger.info("tabooed element index: {} \n".format(tabu_ind))

            # updating all variables
            sMem, sMemVal = self.short_memory(update_ind=tabu_ind,
                                              mem=sMem,
                                              memValue=sMemVal,
                                              solution=solution_items)
            logger.info(
                "short term memory and value: {} {}".format(sMem, sMemVal))

            current_solution_items = np.copy(solution_items)
            current_solution = np.copy(solution)

            solutions.append(current_solution_items)
            solutions_values.append(current_solution[0])
            solutions_weights.append(current_solution[1])

            logger.info("Solution history {} \n".format(solutions_values))

            neighborhood = self.neighborhood(current_solution_items)

            # stopping criteria
            counter += 1
            logger.info("iteration: {}, current_solution: {} \n".format(
                counter, current_solution[0]))

            current_time = time.perf_counter()

        best_weight = solutions_weights[np.nanargmax(solutions_values)]
        best_solution = np.nanmax(solutions_values)
        best_solution_items = solutions[np.nanargmax(solutions_values)]

        print("\nFinal number of solutions checked: ", solutionsChecked)
        print("Best value found: ", best_solution)
        print("Weight is: ", best_weight)
        print("Total number of items selected: ", np.sum(best_solution_items))
        print("Best solution: ", best_solution_items)

        time_to_solve = round(time.perf_counter()-start_time, 3)
        return best_solution, time_to_solve


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

    tabu_search = TabuSearch(capacity, values, weights, time_limit)

    return tabu_search.run()


def main(folder, current_dataset, time_limit):
    files = os.listdir(os.path.join(folder, current_dataset))
    results = []
    for file in files:

        optimum_solution = get_optimum_solution(os.path.join(
            folder, current_dataset+"-optimum", file))

        solution, time = get_solution(os.path.join(
            folder, current_dataset, file), time_limit)

        gap = round(((optimum_solution-solution)/optimum_solution)*100, 2)

        results.append([file,
                       optimum_solution, solution, gap, time])

    print(tabulate(results, headers=[
          "File Name", "Z*", "Z", "Gap %", "Nº Solutions Checked", f"Time (s)"]))


if __name__ == "__main__":
    folder, datasets = "KP-instances", {
        1: "low-dimensional", 2: "large_scale"}

    current_dataset = datasets.get(1)
    time_limit = 10.00

    main(folder, current_dataset, time_limit)
