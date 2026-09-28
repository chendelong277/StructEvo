import math
import random
from typing import List, Tuple

def solve_tsp(coordinates: List[Tuple[float, float]], deadline=None, incumbent_callback=None) -> Tuple[float, List[int]]:

    n = len(coordinates)
    if n <= 1:
        return 0.0, list(range(n))
    if n == 2:
        d = math.dist(coordinates[0], coordinates[1])
        return 2 * d, [0, 1]

    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = coordinates[i]
        for j in range(i + 1, n):
            d = math.hypot(xi - coordinates[j][0], yi - coordinates[j][1])
            dist[i][j] = d
            dist[j][i] = d

    def nearest_neighbor(start: int) -> Tuple[float, List[int]]:
        visited = [False] * n
        route = [start]
        visited[start] = True
        total = 0.0
        current = start
        for _ in range(n - 1):
            best = -1
            best_d = float('inf')
            for j in range(n):
                if not visited[j] and dist[current][j] < best_d:
                    best_d = dist[current][j]
                    best = j
            total += best_d
            route.append(best)
            visited[best] = True
            current = best
        total += dist[current][start]
        return total, route

    best_len = float('inf')
    best_route = None
    num_starts = min(20, n)
    for start in range(num_starts):
        length, route = nearest_neighbor(start)
        if length < best_len:
            best_len = length
            best_route = route

    def greedy_heuristic() -> Tuple[float, List[int]]:
        visited = [False] * n
        route = [0]
        visited[0] = True
        total = 0.0
        current = 0
        for _ in range(n - 1):
            best = -1
            best_d = float('inf')
            for j in range(n):
                if not visited[j] and dist[current][j] < best_d:
                    best_d = dist[current][j]
                    best = j
            total += best_d
            route.append(best)
            visited[best] = True
            current = best
        total += dist[current][0]
        return total, route

    candidate_size = min(max(10, int(math.sqrt(n)) * 2), n - 1)
    candidates = []
    for i in range(n):
        neighbors = sorted(range(n), key=lambda j: dist[i][j] if j != i else float('inf'))
        candidates.append(neighbors[:candidate_size])

    alpha = 1.0 + 0.5 * (1.0 - min(1.0, n / 200.0))
    beta = 3.0 - 1.0 * min(1.0, n / 200.0)
    rho = 0.3 + 0.2 * (1.0 - min(1.0, n / 200.0))
    q0 = 0.85 + 0.1 * min(1.0, n / 200.0)

    tau0 = 1.0 / (n * best_len) if best_len > 0 else 1.0
    tau_max = 1.0 / (0.05 * best_len) if best_len > 0 else 1.0

    tau = [[tau0] * n for _ in range(n)]

    max_iter = min(80, max(10, 1500 // n))
    stagnation = 0
    stagnation_limit = max(3, max_iter // 5)

    def two_opt(route: List[int]) -> Tuple[float, List[int]]:
        length = 0.0
        for k in range(n):
            length += dist[route[k]][route[(k + 1) % n]]

        improved = True
        iteration_count = 0
        max_iterations = n * 2

        while improved and iteration_count < max_iterations:
            improved = False
            iteration_count += 1

            for i in range(n - 2):
                a = route[i]
                b = route[(i + 1) % n]

                check_set = set(candidates[a][:min(10, len(candidates[a]))] +
                               candidates[b][:min(10, len(candidates[b]))])
                for j in range(i + 2, n):
                    if j - i == 1:
                        continue
                    c = route[j]
                    d = route[(j + 1) % n]
                    if c not in check_set and d not in check_set:
                        continue
                    delta = dist[a][c] + dist[b][d] - dist[a][b] - dist[c][d]
                    if delta < -1e-10:
                        route[i + 1:j + 1] = reversed(route[i + 1:j + 1])
                        length += delta
                        improved = True
                        break
                if improved:
                    break
        return length, route

    best_len, best_route = two_opt(best_route[:])
    if incumbent_callback is not None:
        incumbent_callback((best_len, list(best_route)))

    for iteration in range(max_iter):
        if deadline is not None and deadline.expired():
            break

        if stagnation > stagnation_limit // 2:
            rho = min(0.8, rho * 1.05)
        else:
            rho = max(0.1, rho * 0.98)

        num_ants = min(n, max(10, 50 - n // 10))
        tours = []
        lengths = []

        for ant in range(num_ants):

            start = random.randrange(n)
            visited = [False] * n
            route = [start]
            visited[start] = True
            current = start

            for _ in range(n - 1):

                unvisited = [j for j in candidates[current] if not visited[j]]
                if not unvisited:
                    unvisited = [j for j in range(n) if not visited[j]]

                if random.random() < q0:

                    best_j = unvisited[0]
                    best_val = -1.0
                    for j in unvisited:
                        val = (tau[current][j] ** alpha) * ((1.0 / dist[current][j]) ** beta)
                        if val > best_val:
                            best_val = val
                            best_j = j
                    next_city = best_j
                else:

                    total = 0.0
                    probs = []
                    for j in unvisited:
                        p = (tau[current][j] ** alpha) * ((1.0 / dist[current][j]) ** beta)
                        total += p
                        probs.append(p)
                    if total > 0:
                        r = random.random() * total
                        cum = 0.0
                        next_city = unvisited[-1]
                        for idx, j in enumerate(unvisited):
                            cum += probs[idx]
                            if cum >= r:
                                next_city = j
                                break
                    else:
                        next_city = unvisited[0]

                route.append(next_city)
                visited[next_city] = True
                current = next_city

            length = 0.0
            for k in range(n):
                length += dist[route[k]][route[(k + 1) % n]]

            length, route = two_opt(route)
            tours.append(route)
            lengths.append(length)

            if length < best_len:
                best_len = length
                best_route = route[:]
                stagnation = 0
                if incumbent_callback is not None:
                    incumbent_callback((best_len, list(best_route)))

        for i in range(n):
            for j in range(n):
                if i != j:
                    tau[i][j] *= (1.0 - rho)
                    if tau[i][j] < tau0 * 0.01:
                        tau[i][j] = tau0 * 0.01

        k = max(1, num_ants // 3)
        sorted_indices = sorted(range(len(lengths)), key=lambda x: lengths[x])
        for idx in sorted_indices[:k]:
            route = tours[idx]
            length = lengths[idx]
            deposit = 1.0 / length
            for kk in range(n):
                i = route[kk]
                j = route[(kk + 1) % n]
                tau[i][j] += deposit
                if tau[i][j] > tau_max:
                    tau[i][j] = tau_max

        deposit_best = 1.0 / best_len
        for kk in range(n):
            i = best_route[kk]
            j = best_route[(kk + 1) % n]
            tau[i][j] += deposit_best * n
            if tau[i][j] > tau_max:
                tau[i][j] = tau_max

        stagnation += 1
        if stagnation > stagnation_limit:

            for i in range(n):
                for j in range(n):
                    if i != j:
                        tau[i][j] = tau0
            stagnation = 0

            start = random.randrange(n)
            _, temp_route = nearest_neighbor(start)
            temp_len, temp_route = two_opt(temp_route)
            if temp_len < best_len:
                best_len = temp_len
                best_route = temp_route
                if incumbent_callback is not None:
                    incumbent_callback((best_len, list(best_route)))

    best_len, best_route = two_opt(best_route[:])

    greedy_len, greedy_route = greedy_heuristic()
    greedy_len, greedy_route = two_opt(greedy_route)
    if greedy_len < best_len:
        best_len = greedy_len
        best_route = greedy_route

    return best_len, best_route
