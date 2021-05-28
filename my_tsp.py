from charles.charles import Population, Individual
# from charles.search import hill_climb, sim_annealing
from charles.selection import tournament, rank
from charles.mutation import swap_mutation
from charles.crossover import cycle_co, pmx_co
# from random import choices
from copy import deepcopy
import numpy as np
from sklearn.model_selection import ParameterGrid
from pprint import pprint
# from random import shuffle, choice, sample, random
# from operator import  attrgetter
# import csv
# import time
import pandas as pd
from datetime import datetime as DateTime
import tqdm

# Get dataset as user input 
dataset = int(
    input(
        f"Choose a dataset (Enter an interger) -"
        f" \n'1' for Berlin52 or"
        f"\n'2' for eil101 or "
        f"\n'3' for a280\n "
    )
)



# Import the distance matrix for the chosen dataset

if dataset == 1:
    from data.tsp_data_all import distance_matrix_Berlin52 as distance_matrix
    datasetName = "Berlin52"
    print(f"You selected 1: {datasetName}")
    # print(len(distance_matrix))
elif dataset == 2:
    from data.tsp_data_all import distance_matrix_eil101 as distance_matrix
    # print(len(distance_matrix))
    datasetName = "eil101"
    print(f"You selected 2: {datasetName}")
elif dataset == 3:
    from data.tsp_data_all import distance_matrix_a280 as distance_matrix
    # print(len(distance_matrix))
    datasetName = 'a280'
    print(f"You selected 3: {datasetName}")
else: 
    print("invalid input")

# Create param_grid for hyperparameters 
def create_param_grid():
    """A simple function to create a param_grid of 
    hyperparmeter.

    Returns:
        param_grid (list): List of dictionaries of parameter set
    """
    # pop_size_list = np.arange(100,1300,300).tolist()
    gens_list = np.arange(100,1300,300).tolist()
    select_list = ['tournament', 'rank']
    crossover_list = ['cycle_co', 'pmx_co']
    # mutate_list = ['swap_mutation']
    co_p_list = np.arange(0.1,1.1,0.2).round(2).tolist()
    mu_p_list = np.arange(0.1,1.1,0.2).round(2).tolist()
    elitism = [True, False]

    param_set = {
        # 'popsize': pop_size_list,
        'gens' : gens_list,
        'select' : select_list,
        'crossover' : crossover_list,
        # 'mutate' : mutate_list,
        'co_p' : co_p_list,
        'mu_p' : mu_p_list,
        'elitism' : elitism
    }


    param_grid = list(
        ParameterGrid(param_set)
    )

    return param_grid



paramsAndFitnessList = []

def solve_ga_tsp(popsize):
    
    def evaluate(self):
        """A simple objective function to calculate distances
        for the TSP problem.

        Returns:
            int: the total distance of the path
        """

        fitness = 0

        for i in range(len(self.representation)):
            # Calculates full distance, including from last city
            # to first, to terminate the trip
            fitness += distance_matrix[self.representation[i - 1]][self.representation[i]]

        return int(fitness)

    def get_neighbours(self):
        """A neighbourhood function for the TSP problem. Switches
        indexes around in pairs.

        Returns:
            list: a list of individuals
        """
        n = [deepcopy(self.representation) for i in range(len(self.representation) - 1)]

        for count, i in enumerate(n):
            i[count], i[count + 1] = i[count + 1], i[count]

        n = [Individual(i) for i in n]
        return n

    # Monkey patching
    Individual.evaluate = evaluate
    Individual.get_neighbours = get_neighbours

    # Create population from the chosen dataset
    pop = Population(
        size=popsize,
        sol_size=len(distance_matrix[0]),
        valid_set=[i for i in range(len(distance_matrix[0]))],
        replacement=False,
        optim="min",
        print_individual=False
    )
    
    param_grid = create_param_grid()
    
    # Evolve with every parameter set in param_grid
    i=0
    for param_set in tqdm.tqdm(param_grid):
        # if i == 3:
        #     break

        # print(f"\n---Param Set {i+1} of {len(param_grid)}")
        
        pop.evolve(
            gens=param_set['gens'],
            select=globals()[param_set['select']],
            crossover=globals()[param_set['crossover']],
            mutate=swap_mutation,
            co_p=param_set['co_p'],
            mu_p=param_set['mu_p'],
            elitism=param_set['elitism']
        )
        # print(pop.bestIndividual.fitness)
        param_set['fitness'] = pop.bestIndividual.fitness
        param_set['popsize'] = popsize
        paramsAndFitnessList.append(param_set)
        
        i+=1
        # exit()
    # print(paramsAndFitnessList)
    return paramsAndFitnessList

def main():
    # results=[]
    print(f"Started processing all populations at {DateTime.now()}")
    param_grid_results = solve_ga_tsp(100)
    
    # Save the results in a dataframe and export it for visualization
    resultdf = pd.DataFrame(param_grid_results)
    print(resultdf)
    resultdf.to_csv("./tsp_params_fitness_"+datasetName+".csv",index=False)

if __name__ == '__main__':
    main()