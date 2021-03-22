import math

class Op_move_reviser:
    def __init__(self, op_details):
        self.op_details = op_details


    def revise(self, raw_mg):
        temp_prods = []
        prev_prod = ''

        # 各站點的 move 須符合所有限制    
        for idx, op in self.op_details.iterrows():
            move_overflow = self.get_excessive_move_of_1(op)
            if move_overflow > 0:
                self.transfer_op_move_of(op, move_overflow, temp_prods)

            if op['prod'] == prev_prod: # 各產品第 N > 1 站
                move_overflow = self.get_excessive_move_of_2(op)
                if move_overflow > 0:
                    self.transfer_op_move_of(op, move_overflow, temp_prods)
            else:
                prev_prod = op['prod']
                    
            if op['prod'] not in temp_prods:
                temp_prods.append(op['prod'])

        for idx, row in raw_mg.iterrows():
            same_mg_rows = self.op_details.loc[self.op_details['mg_id'] == row['mg_id']]
            total = same_mg_rows['wip'].sum() + same_mg_rows['close_wip'].sum() # 同機群的各站的總 wip + close wip
            self.fix_move_exceed_mg_capacity(row, total)
        
        return self.op_details['op_move'].values    

    
    def get_excessive_move_of_1(self, op):
        op_move = self.op_details.loc[op.name, 'op_move']
        return op_move - (op['wip'] + op['close_wip'])


    def transfer_op_move_of(self, op, move_overflow, prods_after):
        # 更新當站的 move
        self.op_details.at[op.name, 'op_move'] = self.op_details.at[op.name, 'op_move'] - move_overflow

        # 移轉多出的 move 給其他產品
        other_prods = self.get_same_op_or_mg_prod(op, prods_after)
        self.receive_move(other_prods, move_overflow)


    # 尋找同站點或同機群的其他產品
    def get_same_op_or_mg_prod(self, op, prods_exclude):
        return self.op_details.loc[((self.op_details['op_name'] == op['op_name']) | (self.op_details['mg_id'] == op['mg_id'])) & 
                                   (self.op_details['prod'] != op['prod']) & 
                                   ~(self.op_details['prod'].isin(prods_exclude))]

    
    def receive_move(self, tos, total_move):
        # 回傳值表示 move 是否已被成功移轉
    
        if len(tos) > 0:
            # 平均分攤 move 至找出的各筆資料
            each_move = math.ceil(total_move / len(tos))
            for idx, to in tos.iterrows():
                self.op_details.at[idx, 'op_move'] = self.op_details.at[idx, 'op_move'] + each_move # 先加上去後再判斷

                # 未超過限制才能移轉
                move_overflow_1 = self.get_excessive_move_of_1(to)
                move_overflow_2 = self.get_excessive_move_of_2(to)
                if move_overflow_1 > 0 or move_overflow_2 > 0:
                    self.op_details.at[idx, 'op_move'] = self.op_details.at[idx, 'op_move'] - each_move
            return True
        else: # 不移轉
            return False

    
    def get_excessive_move_of_2(self, op):                
        # 找出前一站
        prev_gene_idx = self.op_details.iloc[op.gene_seq - 1].name
        pre_op_op_move = self.op_details.loc[prev_gene_idx, 'op_move']
        
        op_move = self.op_details.loc[op.name, 'op_move']        
        return op_move - (op['wip'] + pre_op_op_move) # + xx layer 回來的均值 ???


    def fix_move_exceed_mg_capacity(self, row, total):
        curr_mg_id = row['mg_id']        
        same_mg_rows = self.op_details.loc[self.op_details['mg_id'] == curr_mg_id]

        total_revised_op_move = same_mg_rows['op_move'].sum() # 同機群的各站調整後的總 move

        # 同機群的各站調整後的總 move <= 同機群的各站的總 wip + close wip
        move_overflow = total_revised_op_move - total
        if move_overflow > 0:
            self.reduce_move_down(same_mg_rows, move_overflow)
            
        # 同機群的各站調整後的總 move <= 機群總 move
        curr_mg_move = row['mg_move'] # 機群總 move
        move_overflow = total_revised_op_move - curr_mg_move
        if move_overflow > 0:
            self.reduce_move_down(same_mg_rows, move_overflow)
    

    def reduce_move_down(self, rows, move_overflow):
        # 扣除多出的 move (平均分攤至同機群各站點)
        move_to_subtract = math.ceil(move_overflow / len(rows))
        remain = 0
        for idx, op in rows.iterrows():
            # 更新當站的 move = 當站的 move - 攤至每站的多出的 move
            curr_op_move = self.op_details.at[idx, 'op_move']
            if curr_op_move >= (move_to_subtract + remain):
                self.op_details.at[idx, 'op_move'] = curr_op_move - (move_to_subtract + remain)                    
            else:
                remain = move_to_subtract - curr_op_move
                self.op_details.at[idx, 'op_move'] = 0
