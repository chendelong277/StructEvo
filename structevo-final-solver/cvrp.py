import math
import random
import time
from typing import List, Optional, Tuple

import numpy as np

def _clamp(x: float) -> float:
    if x > 1.0:
        return 1.0
    if x < -1.0:
        return -1.0
    return x

def _pivot_angles(D: np.ndarray, depot: int) -> np.ndarray:
    n = D.shape[0]
    cust = [i for i in range(n) if i != depot]
    angles = np.zeros(n)
    if not cust:
        return angles

    d_dep = np.asarray(D[depot], dtype=float)
    p0 = None
    for i in sorted(cust, key=lambda x: d_dep[x]):
        if d_dep[i] > 1e-12:
            p0 = i
            break

    if p0 is None:
        return angles

    r0 = float(d_dep[p0])
    others = [i for i in cust if i != p0]
    if not others:
        angles[p0] = 0.0
        return angles

    p1 = max(others, key=lambda i: D[p0][i])
    d1 = float(d_dep[p1])
    if d1 <= 1e-12:
        angles[p0] = 0.0
        return angles

    cos_beta = _clamp((r0 * r0 + d1 * d1 - D[p0][p1] * D[p0][p1]) / (2.0 * r0 * d1))
    beta = math.acos(cos_beta)
    x1 = d1 * math.cos(beta)
    y1 = d1 * math.sin(beta)

    for i in cust:
        if i == p0:
            angles[i] = 0.0
            continue
        di = float(d_dep[i])
        if di <= 1e-12:
            angles[i] = 0.0
            continue

        cos_alpha = _clamp((di * di + r0 * r0 - D[i][p0] * D[i][p0]) / (2.0 * di * r0))
        alpha = math.acos(cos_alpha)
        x = di * math.cos(alpha)
        y = di * math.sin(alpha)
        target = D[i][p1] * D[i][p1]
        sq_plus = (x - x1) ** 2 + (y - y1) ** 2
        sq_minus = (x - x1) ** 2 + (-y - y1) ** 2
        if abs(sq_minus - target) < abs(sq_plus - target):
            alpha = -alpha
        angles[i] = alpha

    return angles

def _compute_angles(D: np.ndarray, depot: int) -> np.ndarray:
    n = D.shape[0]
    try:
        d2 = np.asarray(D, dtype=float) ** 2
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ d2 @ J
        w, V = np.linalg.eigh(B)
        idx = np.argsort(w)[::-1]
        coords = np.zeros((n, 2))
        k = 0
        for ind in idx:
            if w[ind] > 1e-10 and k < 2:
                coords[:, k] = math.sqrt(max(float(w[ind]), 0.0)) * V[:, ind]
                k += 1
        if k < 2:
            return _pivot_angles(D, depot)

        rel = coords - coords[depot]
        angles = np.arctan2(rel[:, 1], rel[:, 0])
        angles[depot] = 0.0
        return angles
    except Exception:
        return _pivot_angles(D, depot)

def _route_cost(route: List[int], D: np.ndarray, depot: int) -> float:
    if not route:
        return 0.0
    cost = float(D[depot, route[0]] + D[route[-1], depot])
    for i in range(len(route) - 1):
        cost += float(D[route[i], route[i + 1]])
    return cost

def _total_cost(routes: List[List[int]], D: np.ndarray, depot: int) -> float:
    return float(sum(_route_cost(r, D, depot) for r in routes))

def _loads(routes: List[List[int]], demands: np.ndarray) -> List[int]:
    return [int(sum(demands[c] for c in r)) for r in routes]

def _cluster_cost(cl: List[int], D: np.ndarray, depot: int) -> float:
    return _route_cost(cl, D, depot)

def _sweep_order(
    customers: List[int],
    angles: np.ndarray,
    offset: float,
    reverse: bool,
    rng: random.Random,
    noise: float = 0.0,
) -> List[int]:
    tau = 2.0 * math.pi
    items = []
    for c in customers:
        a = (float(angles[c]) - offset) % tau
        if noise > 0.0:
            a += rng.gauss(0.0, noise)
        items.append((a, rng.random(), c))
    items.sort(key=lambda x: (x[0], x[1]))
    if reverse:
        items.reverse()
    return [c for _, _, c in items]

def _build_sweep_clusters(
    order: List[int], demands: np.ndarray, capacity: int
) -> Optional[List[List[int]]]:
    clusters = []
    cur = []
    rem = capacity
    for c in order:
        d = int(demands[c])
        if d > capacity:
            return None
        if d <= rem:
            cur.append(c)
            rem -= d
        else:
            if cur:
                clusters.append(cur)
            cur = [c]
            rem = capacity - d
    if cur:
        clusters.append(cur)
    return clusters

def _split_order(
    order: List[int], D: np.ndarray, demands: np.ndarray, capacity: int, depot: int
) -> Optional[List[List[int]]]:
    m = len(order)
    if m == 0:
        return []

    INF = 1e100
    dp = [INF] * (m + 1)
    parent = [-1] * (m + 1)
    dp[0] = 0.0

    for i in range(m):
        if dp[i] >= INF / 2:
            continue
        load = 0
        seg_cost = float(D[depot, order[i]])
        for j in range(i, m):
            d = int(demands[order[j]])
            if d > capacity:
                return None
            load += d
            if load > capacity:
                break
            if j > i:
                seg_cost += float(D[order[j - 1], order[j]])
            val = dp[i] + seg_cost + float(D[order[j], depot])
            if val < dp[j + 1] - 1e-9:
                dp[j + 1] = val
                parent[j + 1] = i

    if dp[m] >= INF / 2:
        return None

    clusters = []
    end = m
    while end > 0:
        start = parent[end]
        if start < 0:
            return None
        clusters.append(order[start:end])
        end = start

    clusters.reverse()
    return clusters

def _savings_clusters(
    customers: List[int], D: np.ndarray, demands: np.ndarray, capacity: int, depot: int
) -> List[List[int]]:
    if not customers:
        return []

    clusters = [[c] for c in customers]
    loads = [int(demands[c]) for c in customers]
    active = [True] * len(clusters)
    pos = {c: i for i, c in enumerate(customers)}

    pairs = []
    for a in range(len(customers)):
        i = customers[a]
        for b in range(a + 1, len(customers)):
            j = customers[b]
            s = float(D[depot, i] + D[depot, j] - D[i, j])
            pairs.append((s, i, j))

    pairs.sort(key=lambda x: -x[0])

    for _, i, j in pairs:
        ri = pos[i]
        rj = pos[j]
        if ri == rj or (not active[ri]) or (not active[rj]):
            continue
        if loads[ri] + loads[rj] > capacity:
            continue

        ci = clusters[ri]
        cj = clusters[rj]
        candidates = []

        if ci[-1] == i and cj[0] == j:
            candidates.append(ci + cj)
        if ci[-1] == i and cj[-1] == j:
            candidates.append(ci + cj[::-1])
        if ci[0] == i and cj[0] == j:
            candidates.append(ci[::-1] + cj)
        if ci[0] == i and cj[-1] == j:
            candidates.append(ci[::-1] + cj[::-1])

        if not candidates:
            continue

        merged = min(candidates, key=lambda x: _cluster_cost(x, D, depot))
        for c in clusters[rj]:
            pos[c] = ri
        clusters[ri] = merged
        loads[ri] += loads[rj]
        active[rj] = False

    return [clusters[i] for i in range(len(clusters)) if active[i]]

def _remove_delta(route: List[int], pos: int, D: np.ndarray, depot: int) -> float:
    L = len(route)
    c = route[pos]
    if L == 1:
        return -float(D[depot, c] + D[c, depot])
    if pos == 0:
        nxt = route[1]
        return float(D[depot, nxt] - D[depot, c] - D[c, nxt])
    if pos == L - 1:
        prv = route[-2]
        return float(D[prv, depot] - D[prv, c] - D[c, depot])
    prv = route[pos - 1]
    nxt = route[pos + 1]
    return float(D[prv, nxt] - D[prv, c] - D[c, nxt])

def _insert_delta(route: List[int], pos: int, c: int, D: np.ndarray, depot: int) -> float:
    L = len(route)
    if L == 0:
        return float(D[depot, c] + D[c, depot])
    if pos == 0:
        nxt = route[0]
        return float(D[depot, c] + D[c, nxt] - D[depot, nxt])
    if pos == L:
        prv = route[-1]
        return float(D[prv, c] + D[c, depot] - D[prv, depot])
    prv = route[pos - 1]
    nxt = route[pos]
    return float(D[prv, c] + D[c, nxt] - D[prv, nxt])

def _intra_2opt(route: List[int], D: np.ndarray, depot: int) -> bool:
    L = len(route)
    if L < 3:
        return False

    for i in range(L - 1):
        for j in range(i + 1, L):
            if i == 0 and j == L - 1:
                continue

            old_left = float(D[depot, route[i]] if i == 0 else D[route[i - 1], route[i]])
            old_right = float(D[route[j], depot] if j == L - 1 else D[route[j], route[j + 1]])
            new_left = float(D[depot, route[j]] if i == 0 else D[route[i - 1], route[j]])
            new_right = float(D[route[i], depot] if j == L - 1 else D[route[i], route[j + 1]])

            delta = (new_left + new_right) - (old_left + old_right)
            if delta < -1e-9:
                route[i : j + 1] = reversed(route[i : j + 1])
                return True
    return False

def _best_relocate(
    routes: List[List[int]],
    loads: List[int],
    D: np.ndarray,
    demands: np.ndarray,
    capacity: int,
    depot: int,
) -> bool:
    best_delta = 0.0
    best_move = None

    for a, ra in enumerate(routes):
        if not ra:
            continue
        for p in range(len(ra)):
            c = ra[p]
            if int(demands[c]) > capacity:
                continue
            rem_delta = _remove_delta(ra, p, D, depot)
            for b, rb in enumerate(routes):
                if a == b:
                    continue
                if loads[b] + int(demands[c]) > capacity:
                    continue
                for q in range(len(rb) + 1):
                    delta = rem_delta + _insert_delta(rb, q, c, D, depot)
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_move = (a, p, b, q, c)

    if best_move is None:
        return False

    a, p, b, q, c = best_move
    routes[a].pop(p)
    routes[b].insert(q, c)
    loads[a] -= int(demands[c])
    loads[b] += int(demands[c])
    return True

def _best_swap(
    routes: List[List[int]],
    loads: List[int],
    D: np.ndarray,
    demands: np.ndarray,
    capacity: int,
    depot: int,
) -> bool:
    best_delta = 0.0
    best_move = None

    for a in range(len(routes)):
        ra = routes[a]
        if not ra:
            continue
        for b in range(a + 1, len(routes)):
            rb = routes[b]
            if not rb:
                continue
            cost_ab = _route_cost(ra, D, depot) + _route_cost(rb, D, depot)
            for p in range(len(ra)):
                ca = ra[p]
                for q in range(len(rb)):
                    cb = rb[q]
                    new_load_a = loads[a] - int(demands[ca]) + int(demands[cb])
                    new_load_b = loads[b] - int(demands[cb]) + int(demands[ca])
                    if new_load_a > capacity or new_load_b > capacity:
                        continue

                    na = ra.copy()
                    nb = rb.copy()
                    na[p] = cb
                    nb[q] = ca
                    delta = (_route_cost(na, D, depot) + _route_cost(nb, D, depot)) - cost_ab
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_move = (a, b, p, q, ca, cb)

    if best_move is None:
        return False

    a, b, p, q, ca, cb = best_move
    routes[a][p] = cb
    routes[b][q] = ca
    loads[a] = loads[a] - int(demands[ca]) + int(demands[cb])
    loads[b] = loads[b] - int(demands[cb]) + int(demands[ca])
    return True

def _best_two_opt_star(
    routes: List[List[int]],
    loads: List[int],
    D: np.ndarray,
    demands: np.ndarray,
    capacity: int,
    depot: int,
) -> bool:
    best_delta = 0.0
    best_move = None
    route_costs = [_route_cost(r, D, depot) for r in routes]

    for a in range(len(routes)):
        ra = routes[a]
        if not ra:
            continue
        for b in range(a + 1, len(routes)):
            rb = routes[b]
            if not rb:
                continue

            La = len(ra)
            Lb = len(rb)
            suffix_a = [0] * (La + 1)
            for i in range(La - 1, -1, -1):
                suffix_a[i] = suffix_a[i + 1] + int(demands[ra[i]])
            suffix_b = [0] * (Lb + 1)
            for i in range(Lb - 1, -1, -1):
                suffix_b[i] = suffix_b[i + 1] + int(demands[rb[i]])

            for i in range(-1, La):
                first_a = ra[: i + 1] if i >= 0 else []
                tail_a = ra[i + 1 :]
                tail_a_load = suffix_a[i + 1]

                for j in range(-1, Lb):
                    if i == La - 1 and j == Lb - 1:
                        continue

                    first_b = rb[: j + 1] if j >= 0 else []
                    tail_b = rb[j + 1 :]
                    tail_b_load = suffix_b[j + 1]

                    new_load_a = loads[a] - tail_a_load + tail_b_load
                    new_load_b = loads[b] - tail_b_load + tail_a_load
                    if new_load_a > capacity or new_load_b > capacity:
                        continue

                    nr = first_a + tail_b
                    nb = first_b + tail_a
                    delta = (
                        _route_cost(nr, D, depot)
                        + _route_cost(nb, D, depot)
                        - (route_costs[a] + route_costs[b])
                    )

                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_move = (a, b, first_a, first_b, tail_a, tail_b)

    if best_move is None:
        return False

    a, b, first_a, first_b, tail_a, tail_b = best_move
    routes[a] = first_a + tail_b
    routes[b] = first_b + tail_a
    loads[a] = loads[a] - sum(int(demands[c]) for c in tail_a) + sum(int(demands[c]) for c in tail_b)
    loads[b] = loads[b] - sum(int(demands[c]) for c in tail_b) + sum(int(demands[c]) for c in tail_a)
    return True

def _local_search(
    routes: List[List[int]],
    loads: List[int],
    D: np.ndarray,
    demands: np.ndarray,
    capacity: int,
    depot: int,
) -> Tuple[List[List[int]], List[int]]:
    improved = True
    while improved:
        improved = False

        for r in routes:
            if _intra_2opt(r, D, depot):
                improved = True

        if _best_relocate(routes, loads, D, demands, capacity, depot):
            improved = True

        if _best_swap(routes, loads, D, demands, capacity, depot):
            improved = True

        if _best_two_opt_star(routes, loads, D, demands, capacity, depot):
            improved = True

    return routes, loads

def _perturb(
    routes: List[List[int]],
    loads: List[int],
    D: np.ndarray,
    demands: np.ndarray,
    capacity: int,
    depot: int,
    rng: random.Random,
) -> Tuple[List[List[int]], List[int]]:
    routes = [r.copy() for r in routes]
    loads = loads.copy()
    nroutes = len(routes)
    if nroutes == 0:
        return routes, loads

    steps = 2 + rng.randrange(4)
    for _ in range(steps):
        k = rng.random()
        if k < 0.4:
            a = rng.randrange(nroutes)
            b = rng.randrange(nroutes)
            if a == b or not routes[a]:
                continue
            p = rng.randrange(len(routes[a]))
            c = routes[a][p]
            if loads[b] + int(demands[c]) > capacity:
                continue
            routes[a].pop(p)
            q = rng.randrange(len(routes[b]) + 1)
            routes[b].insert(q, c)
            loads[a] -= int(demands[c])
            loads[b] += int(demands[c])

        elif k < 0.7:
            if nroutes < 2:
                continue
            a, b = rng.sample(range(nroutes), 2)
            if not routes[a] or not routes[b]:
                continue
            p = rng.randrange(len(routes[a]))
            q = rng.randrange(len(routes[b]))
            ca = routes[a][p]
            cb = routes[b][q]
            if (
                loads[a] - int(demands[ca]) + int(demands[cb]) > capacity
                or loads[b] - int(demands[cb]) + int(demands[ca]) > capacity
            ):
                continue
            routes[a][p] = cb
            routes[b][q] = ca
            loads[a] = loads[a] - int(demands[ca]) + int(demands[cb])
            loads[b] = loads[b] - int(demands[cb]) + int(demands[ca])

        else:
            if nroutes < 2:
                continue
            a, b = rng.sample(range(nroutes), 2)
            ra = routes[a]
            rb = routes[b]
            if not ra or not rb:
                continue

            for _attempt in range(6):
                i = rng.randint(-1, len(ra) - 1)
                j = rng.randint(-1, len(rb) - 1)
                if i == len(ra) - 1 and j == len(rb) - 1:
                    continue

                tail_a = ra[i + 1 :]
                tail_b = rb[j + 1 :]
                tail_a_load = sum(int(demands[c]) for c in tail_a)
                tail_b_load = sum(int(demands[c]) for c in tail_b)
                new_load_a = loads[a] - tail_a_load + tail_b_load
                new_load_b = loads[b] - tail_b_load + tail_a_load
                if new_load_a > capacity or new_load_b > capacity:
                    continue

                first_a = ra[: i + 1] if i >= 0 else []
                first_b = rb[: j + 1] if j >= 0 else []
                routes[a] = first_a + tail_b
                routes[b] = first_b + tail_a
                loads[a] = new_load_a
                loads[b] = new_load_b
                break

    return routes, loads

def solve_vrp(
    distance_matrix: np.ndarray,
    demands: List[int],
    capacity: int,
    depot_index: int = 0,
    seed: Optional[int] = None,
) -> List[List[int]]:
    D = np.asarray(distance_matrix, dtype=float)
    demand_arr = np.asarray(demands, dtype=int)
    n = D.shape[0]
    depot = depot_index
    customers = [i for i in range(n) if i != depot]

    if not customers:
        return []

    rng = random.Random(seed)
    angles = _compute_angles(D, depot)
    deadline = time.time() + 60.0

    initial_candidates = []

    for attempt in range(8):
        offset = rng.uniform(0.0, 2.0 * math.pi)
        reverse = rng.random() < 0.5
        noise = rng.uniform(0.0, 0.6) if attempt > 0 else 0.0
        order = _sweep_order(customers, angles, offset, reverse, rng, noise)
        clusters = _split_order(order, D, demand_arr, capacity, depot)
        if clusters is None:
            clusters = _build_sweep_clusters(order, demand_arr, capacity)
        if clusters:
            routes = [list(cl) for cl in clusters]
            initial_candidates.append((routes, _total_cost(routes, D, depot)))

    clusters = _savings_clusters(customers, D, demand_arr, capacity, depot)
    if clusters:
        routes = [list(cl) for cl in clusters]
        initial_candidates.append((routes, _total_cost(routes, D, depot)))

    for _ in range(3):
        order = customers.copy()
        rng.shuffle(order)
        clusters = _split_order(order, D, demand_arr, capacity, depot)
        if clusters is None:
            clusters = _build_sweep_clusters(order, demand_arr, capacity)
        if clusters:
            routes = [list(cl) for cl in clusters]
            initial_candidates.append((routes, _total_cost(routes, D, depot)))

    if not initial_candidates:
        return []

    initial_candidates.sort(key=lambda x: x[1])
    routes = [r.copy() for r in initial_candidates[0][0]]
    loads = _loads(routes, demand_arr)

    routes, loads = _local_search(routes, loads, D, demand_arr, capacity, depot)

    best_routes = [r.copy() for r in routes]
    best_cost = _total_cost(best_routes, D, depot)

    current_routes = [r.copy() for r in best_routes]
    current_loads = loads.copy()
    current_cost = best_cost

    threshold = max(0.03 * best_cost, 1.0)

    while time.time() < deadline:
        cand_routes, cand_loads = _perturb(
            current_routes, current_loads, D, demand_arr, capacity, depot, rng
        )
        cand_routes, cand_loads = _local_search(
            cand_routes, cand_loads, D, demand_arr, capacity, depot
        )
        cand_cost = _total_cost(cand_routes, D, depot)

        if cand_cost < best_cost - 1e-9:
            best_routes = [r.copy() for r in cand_routes]
            best_cost = cand_cost
            current_routes = [r.copy() for r in cand_routes]
            current_loads = cand_loads.copy()
            current_cost = cand_cost
            threshold = max(0.03 * best_cost, 1.0)
        elif cand_cost <= best_cost + threshold or cand_cost <= current_cost:
            current_routes = [r.copy() for r in cand_routes]
            current_loads = cand_loads.copy()
            current_cost = cand_cost
        else:
            if rng.random() < 0.2:
                current_routes = [r.copy() for r in best_routes]
                current_loads = _loads(current_routes, demand_arr)
                current_cost = best_cost

    final_routes = [r for r in best_routes if r]
    return [[depot] + r + [depot] for r in final_routes]
