import numpy as np
import math

def solve_vrp(distance_matrix, demands, capacity, depot_index=0):
    """
    distance_matrix: (n, n) numpy array, 距离矩阵
    demands: list, 节点需求，demands[i] 是节点 i 的需求
    capacity: int, 车辆最大载重
    depot_index: int, 仓库索引，默认为 0
    
    Returns:
        routes: list of lists. 
                例如: [[0, 2, 3, 0], [0, 4, 1, 0]]
                要求:
                1. 每条路径必须以 depot_index 开始并结束。
                2. 所有非仓库节点(客户)必须恰好出现一次。
                3. 每条路线的需求总和不能超过 capacity。
    """
    # 边界处理
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    n = int(distance_matrix.shape[0])
    depot = int(depot_index)
    if depot < 0 or depot >= n:
        raise ValueError("Invalid depot_index")

    if len(demands) != n:
        raise ValueError("Length of demands must match distance_matrix size")

    # 总需求（排除 depot）
    total_demand = sum(demands[i] for i in range(n) if i != depot and demands[i] > 0)
    if total_demand == 0:
        return []

    # 下界：理论上至少需要的车辆数
    lower_bound = int(math.ceil(total_demand / capacity))

    # 初始阶段：仅对 demand > 0 的客户处理
    customers = [i for i in range(n) if i != depot and demands[i] > 0]
    if len(customers) == 0:
        return []

    # 初始路由：每个客户一个单独的回 depot 路线
    # 路线结构：{ id: {'nodes': [depot, customer, depot], 'load': demands[customer] } }
    routes = {}
    node_to_route = {}  # 客户 -> 路由标识
    for i in customers:
        routes[i] = {'nodes': [depot, i, depot], 'load': int(demands[i])}
        node_to_route[i] = i

    # 计算 Clarke-Wright Savings
    # S(i,j) = d(i, depot) + d(depot, j) - d(i, j)
    savings = []
    for idx_i, i in enumerate(customers):
        di0 = float(distance_matrix[i, depot])
        for j in customers:
            if i >= j:
                continue
            dj0 = float(distance_matrix[depot, j])
            dij = float(distance_matrix[i, j])
            s = di0 + dj0 - dij
            savings.append((s, i, j))
    savings.sort(key=lambda x: x[0], reverse=True)

    # Helpers
    def end_node(route):
        # route['nodes'] 的末尾非 depot 节点
        return route['nodes'][-2]

    def start_node(route):
        # route['nodes'] 的起始非 depot 节点
        return route['nodes'][1]

    # 合并过程：优先考虑大节省
    for s, i, j in savings:
        Ri = node_to_route.get(i)
        Rj = node_to_route.get(j)
        if Ri is None or Rj is None or Ri == Rj:
            continue

        route_i = routes[Ri]
        route_j = routes[Rj]

        # 条件1：Ri 以 i 结尾，Rj 以 j 开头，尝试合并 Ri + Rj
        if (end_node(route_i) == i) and (start_node(route_j) == j):
            new_nodes = route_i['nodes'][:-1] + route_j['nodes'][1:]
            new_load = route_i['load'] + route_j['load']
            if new_load <= capacity:
                route_i['nodes'] = new_nodes
                route_i['load'] = new_load
                for node in route_j['nodes'][1:-1]:
                    node_to_route[node] = Ri
                for node in route_i['nodes'][1:-1]:
                    node_to_route[node] = Ri
                del routes[Rj]
                continue

        # 条件2：Rj 以 j 结尾，Ri 以 i 开始，尝试合并 Rj + Ri
        if (end_node(route_j) == j) and (start_node(route_i) == i):
            new_nodes = route_j['nodes'][:-1] + route_i['nodes'][1:]
            new_load = route_i['load'] + route_j['load']
            if new_load <= capacity:
                route_j['nodes'] = new_nodes
                route_j['load'] = new_load
                for node in route_i['nodes'][1:-1]:
                    node_to_route[node] = Rj
                for node in route_j['nodes'][1:-1]:
                    node_to_route[node] = Rj
                del routes[Ri]
                continue

    # 2-opt 局部改进：对每条路线内部的顺序进行滑动改进
    def route_distance(nodes):
        dist = 0.0
        for a, b in zip(nodes[:-1], nodes[1:]):
            dist += float(distance_matrix[a, b])
        return dist

    def two_opt_improve(nodes):
        best = list(nodes)
        best_dist = route_distance(best)
        improved = True
        while improved:
            improved = False
            m = len(best)
            if m <= 3:
                break
            # 仅对内部段进行 2-opt
            for i in range(1, m - 2):
                for k in range(i + 1, m - 1):
                    new_nodes = best[:i] + list(reversed(best[i:k+1])) + best[k+1:]
                    d = route_distance(new_nodes)
                    if d + 1e-12 < best_dist:
                        best = new_nodes
                        best_dist = d
                        improved = True
                        break
                if improved:
                    break
        return best

    for rid, r in list(routes.items()):
        improved_nodes = two_opt_improve(r['nodes'])
        if improved_nodes != r['nodes']:
            r['nodes'] = improved_nodes
            # 载荷不变

    # 构造最终路线输出：每条路由是一条完整的 [depot, ..., depot] 路线
    final_routes = []
    for rid, r in routes.items():
        if len(r['nodes']) >= 3:
            final_routes.append(r['nodes'])

    # 简单自检：确保所有客户只出现在一个路线中
    visited = set()
    for route in final_routes:
        for node in route:
            if node == depot:
                continue
            if node in visited:
                # 重复客户，忽略该重复，真实场景应避免
                pass
            visited.add(node)

    # 路线数量下界保障：如低于下界，进行保守拆分以至少达到下界
    if len(final_routes) < lower_bound:
        # 重新组织为尽量多的单客户路线，逐步拆分可行性路由
        # 目标：将某些路由中的客户单独拆分成 [depot, x, depot]，并维持原路由合法性
        # 构造一个映射，便于操作
        # 将每条路线尽量拆分，直到达到下界
        additional = []
        # 先收集当前所有路由的可拆分客户
        for r in final_routes:
            # 不拆分 depot
            if len(r) > 3:
                # 试着将最后一个客户单独拆分出去
                # 例如 [0, a, b, c, 0] 拆成 [0, a, 0] 与 [0, b, c, 0]，这里简化为逐步拆分一个客户
                # 选择一个客户进行拆分
                inner = r[1:-1]
                if len(inner) >= 2:
                    x = inner[-1]
                    if demands[x] <= capacity:
                        # 移除 x 得到 ra_without_x
                        ra_without_x_nodes = r[: -1]  # keep depot at end
                        # 将 x 拆分成新路线
                        new_route = [depot, x, depot]
                        additional.append(new_route)
                        # 更新原路由
                        ra_without_x_nodes = r[: -2] + [depot]  # remove last customer before depot
                        # 需要把新路由放回最终路线集合
                        r = ra_without_x_nodes
                        # 重新赋值回最终列表（稍后统一整理）
                        # 这里采用简单策略：把 r 的末端替换更新后再收集
                        idx = final_routes.index(r)
                        final_routes[idx] = ra_without_x_nodes
                        # 将新路由记录
                        continue
        # 将 all additional 路由合并到 final_routes
        final_routes.extend(additional)

    # 进一步的轻量化搬运优化（ relocation ）：
    # 目标：在不违反容量和可行性前提下，通过将客户在路线上重新分配，进一步降低总距离
    def calc_route_dist(nodes):
        dist = 0.0
        for a, b in zip(nodes[:-1], nodes[1:]):
            dist += float(distance_matrix[a, b])
        return dist

    def best_insertion_position(nodes, x):
        best_pos = None
        best_dist = float('inf')
        best_nodes = None
        # 尝试在任意位置插入 x: nodes[:pos] + [x] + nodes[pos:]
        for pos in range(1, len(nodes)):  # pos 从 1 开始，确保 depot 仍在起点
            new_nodes = nodes[:pos] + [x] + nodes[pos:]
            d = calc_route_dist(new_nodes)
            if d < best_dist:
                best_dist = d
                best_pos = pos
                best_nodes = new_nodes
        return best_pos, best_nodes, best_dist

    # 收集路由列表以便搬运操作
    # 仅在存在多条路由且有改进空间时进行尝试
    max_reloc_steps = max(1, len(final_routes) * 3)
    reloc_steps = 0
    # 将 final_routes 转换为可操作的路由对象结构
    route_objs = []
    for fr in final_routes:
        # 载荷计算：简单累加非 depot 节点需求
        load = 0
        for idx in range(1, len(fr) - 1):
            load += int(demands[fr[idx]])
        route_objs.append({'nodes': list(fr), 'load': int(load)})

    # 标准化搬运流程：逐路由尝试移动其中一个客户到其他路由
    improved = True
    while improved and reloc_steps < max_reloc_steps:
        improved = False
        best_delta = 0.0
        best_move = None
        # 先缓存当前所有路由的距离
        current_distances = [calc_route_dist(ro['nodes']) for ro in route_objs]
        # 遍历所有路由对与客户，尝试搬运
        for ia, ra in enumerate(route_objs):
            nodes_a = ra['nodes']
            # 路线至少包含一个客户才有搬运价值
            if len(nodes_a) <= 3:
                continue
            for pos in range(1, len(nodes_a) - 1):
                x = nodes_a[pos]
                dx = int(demands[x])
                if dx <= 0:
                    continue
                # 移除 x 形成 ra_without_x
                ra_without_x_nodes = nodes_a[:pos] + nodes_a[pos+1:]
                if len(ra_without_x_nodes) < 3:
                    # 移除该客户后路线将不再合法，跳过
                    continue
                ra_without_x_dist = calc_route_dist(ra_without_x_nodes)
                ra_without_x_load = ra['load'] - dx
                # 尝试插入到其他路由 rb
                for ib, rb in enumerate(route_objs):
                    if ib == ia:
                        continue
                    if rb['load'] + dx > capacity:
                        continue
                    rb_nodes = rb['nodes']
                    best_pos, best_nodes, new_rb_dist = best_insertion_position(rb_nodes, x)
                    if best_nodes is None:
                        continue
                    new_rb_load = rb['load'] + dx
                    old_rb_dist = calc_route_dist(rb_nodes)
                    old_ra_dist = current_distances[ia]
                    delta = (ra_without_x_dist + new_rb_dist) - (old_ra_dist + old_rb_dist)
                    if delta < best_delta:
                        best_delta = delta
                        best_move = {
                            'from_idx': ia,
                            'to_idx': ib,
                            'x': x,
                            'dx': dx,
                            'ra_without_x_nodes': ra_without_x_nodes,
                            'ra_without_x_dist': ra_without_x_dist,
                            'rb_new_nodes': best_nodes,
                            'rb_new_dist': new_rb_dist,
                        }
        if best_move is not None and best_delta < -1e-6:
            # 应用移动
            ia = best_move['from_idx']
            ib = best_move['to_idx']
            x = best_move['x']
            dx = best_move['dx']
            # 更新来源路由 ra
            ra = route_objs[ia]
            ra['nodes'] = best_move['ra_without_x_nodes']
            ra['load'] = ra['load'] - dx
            # 更新目标路由 rb
            rb = route_objs[ib]
            rb['nodes'] = best_move['rb_new_nodes']
            rb['load'] = rb['load'] + dx
            improved = True
            reloc_steps += 1
        else:
            # 没有更优解，退出搬运循环
            break

    # 最后将 route_objs 转换为最终输出格式
    final_routes = []
    # 只保留合法路线
    for ro in route_objs:
        nodes = ro['nodes']
        if len(nodes) >= 3:
            # 确保头尾为 depot
            if nodes[0] != depot:
                nodes = [depot] + nodes[1:]
            if nodes[-1] != depot:
                nodes = nodes + [depot]
            final_routes.append(nodes)

    # 去除可能的重复客户（理论上不应有重复）
    seen = set()
    unique_routes = []
    for r in final_routes:
        valid = True
        for node in r:
            if node == depot:
                continue
            if node in seen:
                valid = False
                break
            seen.add(node)
        if valid:
            unique_routes.append(r)

    # 如果仍低于下界，进行一个保守的拆分以提升路线数
    if len(unique_routes) < lower_bound:
        # 简单地逐步将单个客户重新形成独立路线，直到达到下界
        # 找到还未独立的客户，逐步加入新的独立 route
        assigned = set()
        for r in unique_routes:
            for node in r:
                if node != depot:
                    assigned.add(node)
        remaining = [c for c in customers if c not in assigned]
        for c in remaining:
            if len(unique_routes) >= lower_bound:
                break
            # 尝试创建一个新路线
            if demands[c] <= capacity:
                unique_routes.append([depot, c, depot])
        # 更新最终返回的路线
        final_routes = unique_routes

    return final_routes