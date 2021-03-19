import pygad # https://pypi.org/project/pygad/
import sys
import math
import pandas as pd
import numpy as np


# 適合度函數
def fitness_func(solution, solution_idx):
    # print('solution: [' + ', '.join(str(x) for x in solution) + ']')

    for idx, sol in enumerate(solution): # 第 1 個基因值表示產品 0 的投片量，第 2 個基因值表示產品 1 的投片量...，若 idx = 0, sol = 240
        ms_of_product = ms[idx] # 產品進行加工時，所使用的機台 & 使用順序，如 4, 2, 1, 3
        mc_of_product = mc[idx] # 各機台對於產品的日產量上限，如 350, 250, 210, 470，分別表示機台 4 一天僅能加工 350 個產品 0，機台 2 一天僅能加工 250 個產品 0...

        is_sol_valid = True # 投片量是否可行
        output = sol
        for i, seq in enumerate(ms_of_product): # 若 i = 0, seq = 4
            machine_util = mu[seq - 1][0] # 機台的可用率，如 0.98
            output = math.ceil(output * machine_util) # 產品經過機台加工後的產出量
            ubound = mc_of_product[i] # 機台 for 產品的日產量，如 350
            if output > ubound:
                is_sol_valid = False # 投片量不可行
                break

        if is_sol_valid == False:
            # print('fitness: 0')
            return 0
    
    output_total = 0
    for i, o in enumerate(solution):
        if o > -1:
            mth_output = o * 30
            mth_demand = dm[i][0]
            if mth_output > mth_demand: # 需滿足月投片量限制
                output_total = output_total + mth_output
    
    # print('fitness: ' + str(output_total))
    return output_total





dm_tmp = pd.read_excel(r"D:\Projects\AI\POC\ClassMaterial\20210219_GA_簡介\MPS Example\hw1_dataset.xlsx",sheet_name="Monthly Demand",index_col =[0])
mu_tmp = pd.read_excel(r"D:\Projects\AI\POC\ClassMaterial\20210219_GA_簡介\MPS Example\hw1_dataset.xlsx",sheet_name="Machine Utilization",index_col =[0])
ms_tmp = pd.read_excel(r"D:\Projects\AI\POC\ClassMaterial\20210219_GA_簡介\MPS Example\hw1_dataset.xlsx",sheet_name="Machine Sequence",index_col =[0])
mc_tmp = pd.read_excel(r"D:\Projects\AI\POC\ClassMaterial\20210219_GA_簡介\MPS Example\hw1_dataset.xlsx",sheet_name="Machine Daily Capacity",index_col =[0])

num_p = dm_tmp.shape[0] # product 數
num_m = mu_tmp.shape[0] # machine 數

dm = [list(map(int, dm_tmp.iloc[i])) for i in range(num_p)]
mu = [list(map(float, mu_tmp.iloc[i])) for i in range(num_m)]
ms = [list(map(int, ms_tmp.iloc[i])) for i in range(num_p)]
mc = [list(map(int, mc_tmp.iloc[i])) for i in range(num_p)]

population_size = 20



# 建立 GA 實體
ga_instance = pygad.GA(
                       # 產生初始族群
                       sol_per_pop = population_size, # 族群內的個體數
                       num_genes = num_p, # 個體內的基因數                       
                       gene_type = int,
                       init_range_low = 0,
                       init_range_high = 500,

                       # 選擇 (selection)
                       parent_selection_type = "rws", # 選擇方式
                       keep_parents = 2,

                       # 交配 (crossover)
                       num_parents_mating = population_size, # 取幾個個體進行交配                    
                       crossover_probability = 0.8, # 交配機率
                       crossover_type = "two_points", # 交配方式

                       # 突變 (mutation)
                       mutation_probability = 0.2, # 突變機率
                       mutation_type = "random", # 突變方式
                       mutation_by_replacement = True,
                       random_mutation_min_val = 0,
                       random_mutation_max_val = 500,

                       # 適應度函數
                       fitness_func = fitness_func,

                       num_generations = 200 # 跑幾個世代
                    )
ga_instance.run() # 執行 GA

# ga_instance.plot_result() # 繪製各世代的適應度趨勢



# 取得最佳解
solution, solution_fitness, solution_idx = ga_instance.best_solution()

print("最佳解 (日投片量): {solution}".format(solution = solution))

mth_output = []
for sol in solution:
    mth_output.append(sol * 30)

print("最佳解 (月產量): {solution}".format(solution = mth_output))
print("最佳解的適應度 (各產品月產量加總): {solution_fitness}".format(solution_fitness = solution_fitness))

if ga_instance.best_solution_generation != -1:
    print("最佳解落在第 {best_solution_generation} 個世代".format(best_solution_generation = ga_instance.best_solution_generation))

print("\n")