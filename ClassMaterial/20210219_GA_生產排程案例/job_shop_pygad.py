import pygad # https://pypi.org/project/pygad/
import sys
import math
import pandas as pd
import numpy as np


# 產生初始族群
def create_init_population():
    init_population = []
    for i in range(population_size):
        # 產生亂數
        random_num = list(np.random.permutation(num_gene))
        init_population.append(random_num)

        # 將亂數轉成 job number
        for j in range(num_gene):        
            init_population[i][j] = init_population[i][j] % num_job
    return init_population

# 紀錄修正前後的染色體
def set_solution_map(sol, repaired_sol):
    sol_str = "".join(str(x) for x in sol) # 串接陣列內各元素
    if sol_str in solution_map:
        pass
    else:
        solution_map[sol_str] = repaired_sol

# 校正染色體 (因交配後的染色體中，各 job 出現的次數可能不同)
def repair(solution):
    # print('before repairment')
    # print(solution)
    
    repaired_sol = solution.tolist()
    job_count = {}
    larger,less = [],[]

    for i in range(num_job):
        if i in repaired_sol:
            count = repaired_sol.count(i) # 染色體中，job i 出現的次數
            pos = repaired_sol.index(i) # 染色體中，job i 出現的位置
            job_count[i] = [count,pos]
        else:
            count = 0
            job_count[i] = [count, 0]
        
        if count > num_mc:
            larger.append(i) # 'larger' 紀錄染色體中，出現超過 num_mc 次的 job
        elif count < num_mc: # 'less' 紀錄染色體中，出現少於 num_mc 次的 job
            less.append(i)
            
    # 一一校正 lager 中各 job
    for k in range(len(larger)):
        chg_job = larger[k]
        while job_count[chg_job][0] > num_mc:
            for d in range(len(less)):
                if job_count[less[d]][0] < num_mc:
                    repaired_sol[job_count[chg_job][1]] = less[d]
                    # 更新 job_count
                    job_count[chg_job][1] = repaired_sol.index(chg_job)
                    job_count[chg_job][0] = job_count[chg_job][0] - 1
                    job_count[less[d]][0] = job_count[less[d]][0] + 1
                if job_count[chg_job][0] == num_mc:
                    break     
        
    # print('after repairment')
    # print(repaired_sol)
    set_solution_map(solution, repaired_sol)
    return repaired_sol

# 適合度函數
def fitness_func(solution, solution_idx):
    repaired_sol = repair(solution)
    
    # 計算適合度
    j_keys = [j for j in range(num_job)]
    key_count = { key: 0 for key in j_keys }
    j_count = { key: 0 for key in j_keys }
    m_keys = [j + 1 for j in range(num_mc)]
    m_count = { key: 0 for key in m_keys }
    '''
    j_keys
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    key_count
    {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    j_count
    {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    m_keys
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    m_count
    {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
    '''
    for i in repaired_sol:
        gen_t = int(pt[i][key_count[i]]) # job i 的第 key_count 次 processTime, pt[3][0] processTime=81. pt[4][0]=14==>job4 pt=81, job5 pt=14
        gen_m = int(ms[i][key_count[i]]) # job i 的第 key_count 次 機器 number
        j_count[i] = j_count[i] + gen_t # job i 的 累計 processTime
        m_count[gen_m] = m_count[gen_m] + gen_t # 機器 gen_m 的累計 processTime 
            
        # 把機台累計時間與job累計時間, 取最大時間, 更新機台/job最後時間
        if m_count[gen_m] < j_count[i]:
            m_count[gen_m] = j_count[i]
        elif m_count[gen_m] > j_count[i]:
            j_count[i] = m_count[gen_m]
            
        key_count[i] = key_count[i] + 1

    makespan = max(j_count.values()) # 完工時間 = 取最大的累計 processTime
    # print(makespan)
    return 1 / makespan # 目標: 完工時間最短





pt_tmp = pd.read_excel(r"D:\Project\MyPython\GA\JSP_dataset.xlsx",sheet_name="Processing Time",index_col =[0])
ms_tmp = pd.read_excel(r"D:\Project\MyPython\GA\JSP_dataset.xlsx",sheet_name="Machines Sequence",index_col =[0])

num_mc = pt_tmp.shape[1] # machine 數
num_job = pt_tmp.shape[0] # job 數
num_gene = num_mc * num_job # 染色體內的基因數

pt = [list(map(int, pt_tmp.iloc[i])) for i in range(num_job)]
ms = [list(map(int,ms_tmp.iloc[i])) for i in range(num_job)]


population_size = 30
population_list = create_init_population()

solution_map = {} # key: source solution, value: repaired solution



# 建立 GA 實體
ga_instance = pygad.GA(
                       # 產生初始族群
                       initial_population = population_list,

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
                       random_mutation_max_val = 9,

                       # 適應度函數
                       fitness_func = fitness_func,

                       num_generations = 100 # 跑幾個世代
                    )
ga_instance.run() # 執行 GA

# ga_instance.plot_result() # 繪製各世代的適應度趨勢



# 取得最佳解
solution, solution_fitness, solution_idx = ga_instance.best_solution()

# print("最佳解 : {solution}".format(solution = solution))
print("最佳解 : {solution}".format(solution = solution_map["".join(str(x) for x in solution)]))

print("最佳解的適應度 : {solution_fitness}".format(solution_fitness = solution_fitness))
print("最佳解的適應度倒數 : {solution_fitness}".format(solution_fitness = str(1/solution_fitness)))

if ga_instance.best_solution_generation != -1:
    print("最佳解落在第 {best_solution_generation} 個世代".format(best_solution_generation = ga_instance.best_solution_generation))

print("\n\n\n")