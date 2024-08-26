from collections import defaultdict

def cbba(benefit_dict, max_iterations=100, min_ugvs_per_target=3, max_ugvs_per_target=5):
    # 初始化
    agents = list(set([ugv for target in benefit_dict for ugv in benefit_dict[target]]))
    targets = list(benefit_dict.keys())
    
    agent_assignments = {agent: None for agent in agents}
    agent_benefits = {agent: {} for agent in agents}
    target_assignments = defaultdict(list)

    def calculate_benefits():
        for target, benefits in benefit_dict.items():
            for agent, benefit in benefits.items():
                agent_benefits[agent][target] = benefit

    def select_tasks():
        for agent in agents:
            if agent_benefits[agent]:
                # 按收益从高到低排序
                sorted_benefits = sorted(agent_benefits[agent].items(), key=lambda x: x[1], reverse=True)
                # 选择收益最高的任务
                agent_assignments[agent] = sorted_benefits[0][0]

    def update_assignments():
        # 清空当前目标分配
        for target in targets:
            target_assignments[target] = []

        # 根据每个UGV的选择更新任务分配
        for agent, target in agent_assignments.items():
            if target is not None:
                target_assignments[target].append((agent, agent_benefits[agent][target]))

    def resolve_conflicts():
        for target in targets:
            if len(target_assignments[target]) > max_ugvs_per_target:
                # 找到收益最低的UGV并移除，直到符合最大UGV数量
                sorted_assignments = sorted(target_assignments[target], key=lambda x: x[1])
                while len(target_assignments[target]) > max_ugvs_per_target:
                    ugv_to_remove, _ = sorted_assignments.pop(0)
                    agent_assignments[ugv_to_remove] = None
                    target_assignments[target].remove((ugv_to_remove, agent_benefits[ugv_to_remove][target]))

        for target in targets:
            while len(target_assignments[target]) < min_ugvs_per_target:
                # 找到未分配的UGV并添加到当前目标，直到符合最小UGV数量
                additional_ugvs = sorted(
                    [(agent, agent_benefits[agent][target]) for agent in agents if agent_assignments[agent] is None],
                    key=lambda x: x[1],
                    reverse=True
                )
                if additional_ugvs:
                    ugv_to_add, _ = additional_ugvs.pop(0)
                    agent_assignments[ugv_to_add] = target
                    target_assignments[target].append((ugv_to_add, agent_benefits[ugv_to_add][target]))
                else:
                    break

    def check_consistency():
        for target in targets:
            if len(target_assignments[target]) < min_ugvs_per_target or len(target_assignments[target]) > max_ugvs_per_target:
                return False
        return True

    # 主要循环
    calculate_benefits()
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        select_tasks()
        update_assignments()
        resolve_conflicts()
        if check_consistency():
            break

    # 输出最终任务分配
    final_assignments = []
    for target in targets:
        final_assignments.append([agent for agent, _ in target_assignments[target]])

    return final_assignments
# 示例字典
benefit_dict = {
    "Target_0": {0: 2786.531487222121, 1: 2717.141293354679, 7: 2665.6269656421377, 2: 2758.712848725679, 6: 209.80176109551746, 3: 191.19505225805992, 4: 2546.4535064771144, 8: 211.79407586165013, 5: 188.71857182411463}, 
    "Target_1": {0: 693.8583480919154, 1: 697.2566760652994, 7: 748.83756852169, 2: 681.4751559151266, 6: 3537.702347411496, 3: 3321.0435791638993, 4: 881.2560596238144, 8: 3594.6688509045384, 5: 3258.2662164458225},
    "Target_2": {0: 2404.383627400785, 1: 2559.1189681649075, 7: 2431.784308832652, 2: 2439.4056679026744, 6: 1628.6516195994427, 3: 1835.9578257827973, 4: 2482.683700618737, 8: 1897.0929139596337, 5: 1556.4379882833684}
}

final_assignments = cbba(benefit_dict)
print(final_assignments)
