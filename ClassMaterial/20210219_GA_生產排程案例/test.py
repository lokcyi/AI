# 校正染色體 (因交配後的染色體中，各 job 出現的次數可能不同)
def repair(solution):
    # print('before repairment')
    # print(solution)
    
    repaired_sol = solution.tolist()
    job_count = {}
    larger,less = [],[]

    for i in range(num_genes):
        if i in repaired_sol:
            count = repaired_sol.count(i) # 染色體中，job i 出現的次數
            pos = repaired_sol.index(i) # 染色體中，job i 出現的位置
            job_count[i] = [count,pos]
        else:
            count = 0
            job_count[i] = [count, 0]
        
        if count > 1:
            larger.append(i) # 'larger' 紀錄染色體中，出現超過 num_mc 次的 job
        elif count < 1: # 'less' 紀錄染色體中，出現少於 num_mc 次的 job
            less.append(i)
            
    # 一一校正 lager 中各 job
    for k in range(len(larger)):
        chg_job = larger[k]
        while job_count[chg_job][0] > num_genes:
            for d in range(len(less)):
                if job_count[less[d]][0] < num_genes:
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
    print("repaire",solution,repaired_sol)
    return repaired_sol