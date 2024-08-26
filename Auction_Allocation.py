import numpy as np

# 拍卖算法
def auction_algorithm(cost_dict):
    # 初始化
    targets = list(cost_dict.keys())
    ugvs = list(cost_dict[targets[0]].keys())
    n_targets = len(targets)
    n_ugvs = len(ugvs)

    bids = {t: {u: -np.inf for u in ugvs} for t in targets}
    prices = {t: 0 for t in targets}
    assignments = {u: None for u in ugvs}
    target_counts = {t: 0 for t in targets}

    # 拍卖过程
    epsilon = 0.1  # 价格步长
    unassigned_ugvs = list(ugvs)

    while unassigned_ugvs:
        for u in unassigned_ugvs:
            best_target = None
            best_value = -np.inf
            second_best_value = -np.inf

            # 找到最优和次优的任务，并考虑目标分配的约束
            for t in targets:
                if target_counts[t] < 5:  # 确保目标分配不超过5个点
                    value = -cost_dict[t][u] - prices[t]
                    if value > best_value:
                        second_best_value = best_value
                        best_value = value
                        best_target = t
                    elif value > second_best_value:
                        second_best_value = value

            if best_target is None:
                continue

            # 竞拍
            bid_increment = best_value - second_best_value + epsilon
            bids[best_target][u] = bid_increment + prices[best_target]
            prices[best_target] += bid_increment

            # 分配
            current_assignment = assignments[u]
            if current_assignment != best_target:
                assignments[u] = best_target
                if current_assignment is not None:
                    bids[current_assignment][u] = -np.inf
                    target_counts[current_assignment] -= 1
                target_counts[best_target] += 1

        # 检查分配是否满足条件
        unassigned_ugvs = [u for u in ugvs if assignments[u] is None]

    # 确保每个目标至少分配到3个点
    for t in targets:
        while target_counts[t] < 3:
            remaining_ugvs = [u for u in ugvs if assignments[u] is None or target_counts[assignments[u]] > 3]
            if not remaining_ugvs:
                raise ValueError("无法满足每个目标至少分配到3个点的约束条件")
            for u in remaining_ugvs:
                if target_counts[t] >= 3:
                    break
                if assignments[u] is None or target_counts[assignments[u]] > 3:
                    if assignments[u] is not None:
                        target_counts[assignments[u]] -= 1
                    assignments[u] = t
                    target_counts[t] += 1

    # 输出结果
    assignment_result = {t: [] for t in targets}
    for u, t in assignments.items():
        if t is not None:
            assignment_result[t].append(u)

    # 转换为分配列表
    result = [assignment_result[t] for t in targets]

    return result


# 合同网协议分配算法


def contract_net_protocol(cost_dict):
    # 初始化
    targets = list(cost_dict.keys())
    ugvs = list(cost_dict[targets[0]].keys())
    n_targets = len(targets)
    n_ugvs = len(ugvs)

    assignments = {t: [] for t in targets}
    remaining_ugvs = set(ugvs)

    # 确保每个目标至少分配到3个点
    for t in targets:
        bids = [(u, cost_dict[t][u]) for u in remaining_ugvs]
        bids.sort(key=lambda x: x[1])  # 按成本排序
        for u, _ in bids[:3]:  # 选择前3个成本最低的UGV
            assignments[t].append(u)
            remaining_ugvs.remove(u)
        if len(assignments[t]) < 3:
            raise ValueError(f"无法满足目标 {t} 至少分配到3个点的约束条件")

    # 分配剩余的UGV，确保每个目标不超过5个点
    while remaining_ugvs:
        bids = []
        for t in targets:
            for u in remaining_ugvs:
                bids.append((u, t, cost_dict[t][u]))
        bids.sort(key=lambda x: x[2])  # 按成本排序
        for u, t, _ in bids:
            if len(assignments[t]) < 5:
                assignments[t].append(u)
                remaining_ugvs.remove(u)
                break

    # 确保每个目标分配到的点不超过5个
    for t in targets:
        if len(assignments[t]) > 5:
            raise ValueError(f"目标 {t} 分配的点超过5个")

    # 转换为分配列表
    result = [assignments[t] for t in targets]

    return result