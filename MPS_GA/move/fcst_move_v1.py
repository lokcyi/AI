import sys
import math
import pygad
import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_absolute_error
from product import Product
from op_move_reviser import Op_move_reviser

s_time = time.time() 

# 取得 & 整理資料
raw_mg = pd.read_excel(r'GA/move/forecast_op_move.xlsx', sheet_name = 'mg')
raw_mg.columns = ['mg_id', 'mg_wip', 'mg_ct']
raw_mg['mg_move'] = np.floor(raw_mg['mg_wip'] / raw_mg['mg_ct'])
raw_mg = raw_mg.drop(['mg_wip'], axis = 1)

raw_product_op = pd.read_excel(r'GA/move/forecast_op_move.xlsx', sheet_name = 'product_op')
raw_product_op.columns = ['prod', 'op_seq', 'op_name', 'mg_id', 'wip', 'close_wip']

raw_details = raw_product_op.merge(raw_mg, on='mg_id', suffixes=('', '_right'))

raw_details['close_wip'] = raw_details['close_wip'].fillna(0)
# 後續未用到
# raw_details['op_move_rough'] = np.floor(raw_details['wip'] / raw_details['mg_ct']) # 取整數
# raw_details['op_move'] = raw_details['op_move_rough']
# raw_details['op_ct'] = raw_details['mg_ct']


gene_ubuond = 150 # todo: 定義上限值


op_details = raw_details.sort_values(by = ['prod', 'op_seq'], ascending = True).loc[:, ['prod', 'op_name', 'mg_id', 'wip', 'close_wip']]
op_details['op_move'] = -1 # 基因值
op_details['gene_seq'] = np.arange(op_details.shape[0]) # 從零開始的流水號 = 基因所在位置


# 製作初始族群
population_count = 10
init_populations = []
# 將 mg op_move 加入初始族群並無太大意義
# init_solution = raw_details.sort_values(by = ['prod', 'op_seq'], ascending = True).loc[:, ['op_move_rough']].values.flatten()
# init_populations.append(np.array(init_solution))
for i in range(population_count - 1):    
    init_populations.append(np.random.randint(0, gene_ubuond, op_details.shape[0]))

op_move_reviser = None
products = op_details.groupby('prod').groups.keys()


def compute_fitness():
    moves_1 = []
    moves_2 = []

    # 勿計算 調整後的各站 move 與 從機群總 move 攤到各站的 move 間的 MAE，否則將影響最佳解的取得
    # for prod in products:
    #     rough_op_moves = raw_details.loc[raw_details['prod'] == prod].sort_values(by = ['op_seq'], ascending = True)['op_move_rough'] # 從機群總 move 攤到各站的 move
    #     revised_op_moves = op_move_reviser.get_gene_of(prod) # 調整後的各站 move
    #     moves_1.extend(rough_op_moves)
    #     moves_2.extend(revised_op_moves)
    
    for idx, row in raw_mg.iterrows():
        mg_id = row['mg_id']
        mg_move = row['mg_move'] # 機群的總 move
        total_revised_op_move = op_move_reviser.op_details.loc[op_move_reviser.op_details['mg_id'] == mg_id]['op_move'].sum() # 同機群的各站調整後的總 move
        moves_1.append(mg_move)
        moves_2.append(total_revised_op_move)
    
    return 1 / mean_absolute_error(moves_1, moves_2)


def fitness_func(solution, solution_idx):
    global op_details
    global op_move_reviser

    op_details['op_move'] = solution # 將染色體內的各基因依序填到各產品各站點的 op_move
    op_move_reviser = Op_move_reviser(op_details)
    revised_op_moves = op_move_reviser.revise(raw_mg)    

    solution[:] = revised_op_moves # 將調整後的結果更新回 solution
    fitness = compute_fitness()    
    return fitness


def on_generation(ga_obj):    
    curr_generation = str(ga_obj.generations_completed)
    print("已生成世代 " + curr_generation + " 族群內個體")
    # print(ga_obj.population)


ga_obj = pygad.GA(
                        gene_type = int,
                        initial_population = init_populations,
                        num_generations = 30,
                        keep_parents = (int)(math.floor(population_count * 0.4)),
                        fitness_func = fitness_func,

                        # 選擇
                        parent_selection_type = 'rank',

                        # 交配
                        crossover_type = 'two_points',
                        crossover_probability = 0.6,
                        num_parents_mating = len(init_populations),

                        # 突變
                        mutation_type = 'adaptive',
                        mutation_probability = [0.8, 0.4],
                        # mutation_type = 'random',
                        # mutation_probability = 0.8,                        
                        # mutation_by_replacement = True,
                        # random_mutation_min_val = 0,
                        # random_mutation_max_val = gene_ubuond,

                        on_generation = on_generation
                        )
ga_model_path = 'GA/move/ga_model'
ga_obj.save(filename = ga_model_path)
# ga_obj = pygad.load(filename = ga_model_path)
ga_obj.run()

# ga_obj.plot_result() # 繪製各世代的適應度趨勢

print("\n")
# 取得最佳解 
solution, solution_fitness, solution_idx = ga_obj.best_solution()
print("最佳解 : {solution}".format(solution = solution))
print("最佳解的適應度 : {solution_fitness}".format(solution_fitness = solution_fitness))
if ga_obj.best_solution_generation != -1:
    print("最佳解落在第 {best_solution_generation} 個世代".format(best_solution_generation = ga_obj.best_solution_generation))

print("\n")

op_details['op_move'] = solution
op_move_reviser = Op_move_reviser(op_details)
print('機群總 move: ' + str(raw_mg['mg_move'].sum()))
print('預測機群總 move: ' + str(sum(op_move_reviser.op_details['op_move'].values)))
for prod in products:
    op_moves = op_details.loc[op_details['prod'] == prod]['op_move'].values    
    print('預測產品 ' + prod + ' 總 move: ' + str(sum(op_moves)))

print("\n")
e_time = time.time()
print('time elapsed: ', (e_time - s_time) / 60, 'minute')
