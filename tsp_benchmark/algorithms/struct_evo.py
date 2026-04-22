import math
import os
import random


def _resolve_max_evals(default: int = 10000) -> int:
    raw_value = os.environ.get("STRUCTEVO_MAX_EVALS")
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return max(1, value)


MAX_EVALS = _resolve_max_evals()
def euclidean_dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def build_distance_matrix(coordinates):
    n = len(coordinates)
    dist = [[0.0]*n for _ in range(n)]
    for i in range(n):
        xi, yi = coordinates[i]
        for j in range(i+1, n):
            d = euclidean_dist(coordinates[i], coordinates[j])
            dist[i][j] = d
            dist[j][i] = d
    return dist

def nearest_neighbor_tour(start, dist):
    n = len(dist)
    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)
    current = start
    while unvisited:
        nxt = min(unvisited, key=lambda x: dist[current][x])
        tour.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return tour

class Evaluator:
    def __init__(self, dist, max_evals=MAX_EVALS):
        self.dist = dist
        self.n = len(dist)
        self.evals = 0
        self.max_evals = max_evals

    def can_eval(self):
        return self.evals < self.max_evals

    def tour_cost(self, tour):
        if not self.can_eval():
            raise StopIteration("Evaluation budget exhausted")
        total = 0.0
        n = len(tour)
        if n == 0:
            self.evals += 1
            return 0.0
        for i in range(n-1):
            total += self.dist[tour[i]][tour[i+1]]
        total += self.dist[tour[-1]][tour[0]]
        self.evals += 1
        return total

def order_crossover(p1, p2):
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [-1]*n
    child[a:b+1] = p1[a:b+1]
    pos = (b+1) % n
    p2_idx = (b+1) % n
    while -1 in child:
        if p2[p2_idx] not in child:
            child[pos] = p2[p2_idx]
            pos = (pos+1)%n
        p2_idx = (p2_idx+1)%n
    return child

def mutate_inversion(tour):
    n = len(tour)
    a, b = sorted(random.sample(range(n), 2))
    tour[a:b+1] = list(reversed(tour[a:b+1]))

def mutate_swap(tour):
    n = len(tour)
    i, j = random.sample(range(n), 2)
    tour[i], tour[j] = tour[j], tour[i]

def mutate_insertion(tour):
    n = len(tour)
    i, j = random.sample(range(n), 2)
    if i < j:
        x = tour.pop(j)
        tour.insert(i+1, x)
    else:
        x = tour.pop(i)
        tour.insert(j+1, x)

def two_opt_local_search(tour, dist, evaluator=None, max_iterations=1000):
    n = len(tour)
    if n <= 2:
        cost = 0.0
        for i in range(n-1):
            cost += dist[tour[i]][tour[i+1]]
        if n > 0:
            cost += dist[tour[-1]][tour[0]]
        return tour, cost

    cost = 0.0
    for i in range(n-1):
        cost += dist[tour[i]][tour[i+1]]
    cost += dist[tour[-1]][tour[0]]

    improved = True
    iters = 0
    while improved and iters < max_iterations:
        improved = False
        iters += 1
        for i in range(0, n-1):
            a = tour[i]
            b = tour[i+1]
            j_start = i+2
            j_end = n if i > 0 else n-1
            for j in range(j_start, j_end):
                c = tour[j]
                d = tour[(j+1)%n]
                delta = dist[a][c] + dist[b][d] - dist[a][b] - dist[c][d]
                if delta < -1e-12:
                    tour[i+1:j+1] = list(reversed(tour[i+1:j+1]))
                    cost += delta
                    improved = True
                    break
            if improved:
                break
        if evaluator is not None and not evaluator.can_eval():
            break
    return tour, cost

def tournament_select(population, k=3, bias=0.9):
    if not population:
        return None
    if random.random() > bias:
        return random.choice(population)
    selected = random.sample(population, min(k, len(population)))
    selected.sort(key=lambda ind: ind['cost'])
    return selected[0]

def ensure_valid_tour(tour, n):
    if len(tour) != n:
        return list(range(n))
    used = [False]*n
    res = []
    for x in tour:
        if isinstance(x, int) and 0 <= x < n and not used[x]:
            res.append(x)
            used[x] = True
    for i in range(n):
        if not used[i]:
            res.append(i)
    return res

def population_diversity(population, sample_pairs=100):
    m = len(population)
    if m < 2:
        return 0.0
    n = len(population[0]['tour'])
    pairs = []
    max_pairs = min(sample_pairs, m*(m-1)//2)
    if m*(m-1)//2 <= max_pairs:
        for i in range(m):
            for j in range(i+1, m):
                pairs.append((i,j))
    else:
        seen = set()
        while len(pairs) < max_pairs:
            i, j = random.sample(range(m), 2)
            if i > j:
                i, j = j, i
            if (i,j) in seen:
                continue
            seen.add((i,j))
            pairs.append((i,j))
    total_mismatch = 0.0
    for i,j in pairs:
        t1 = population[i]['tour']
        t2 = population[j]['tour']
        mismatches = sum(1 for k in range(n) if t1[k] != t2[k])
        total_mismatch += mismatches / n
    return total_mismatch / len(pairs) if pairs else 0.0

def solve_tsp(coordinates):
    n = len(coordinates)
    if n == 0:
        return 0.0, []
    if n == 1:
        return 0.0, [0]

    BASE_POP = 100
    ELITE_RATE = 0.08
    BASE_CROSSOVER_RATE = 0.9
    BASE_MUTATION_RATE = 0.05
    BASE_MEMETIC_RATE = 0.30
    TOURNAMENT_K = 3
    NO_IMPROVE_LIMIT = 50
    MAX_GENERATIONS = 1000

    dist = build_distance_matrix(coordinates)
    evaluator = Evaluator(dist, max_evals=MAX_EVALS)

    est_init_eval = min(200, max(20, n))
    pop_upper = max(10, min(BASE_POP, evaluator.max_evals // max(5, est_init_eval//10)))
    POP_SIZE = min(pop_upper, max(10, int(2 * math.sqrt(n))))
    POP_SIZE = max(8, POP_SIZE)

    population = []
    nn_count = max(1, POP_SIZE // 4)
    starts = list(range(n))
    random.shuffle(starts)
    for i in range(nn_count):
        if not evaluator.can_eval():
            break
        start = starts[i % n]
        tour = nearest_neighbor_tour(start, dist)
        try:
            cost = evaluator.tour_cost(tour)
        except StopIteration:
            break
        population.append({'tour': tour, 'cost': cost})

    while len(population) < POP_SIZE and evaluator.can_eval():
        tour = list(range(n))
        random.shuffle(tour)
        try:
            cost = evaluator.tour_cost(tour)
        except StopIteration:
            break
        population.append({'tour': tour, 'cost': cost})

    if not population:
        return 0.0, []

    POP_SIZE = len(population)
    elite_count = max(1, int(round(ELITE_RATE * POP_SIZE)))

    population.sort(key=lambda ind: ind['cost'])
    best = {'tour': population[0]['tour'][:], 'cost': population[0]['cost']}
    best_no_improve = 0
    generation = 0

    try:
        while evaluator.can_eval() and generation < MAX_GENERATIONS:
            generation += 1

            div = population_diversity(population)

            stagnation_factor = min(1.0, best_no_improve / max(1, NO_IMPROVE_LIMIT//3))

            mutation_rate = min(0.6, BASE_MUTATION_RATE * (1.0 + 2.2*stagnation_factor + (1.0 - div)))
            memetic_rate = min(0.85, BASE_MEMETIC_RATE * (1.0 + 1.2*div - 0.6*stagnation_factor))
            crossover_rate = BASE_CROSSOVER_RATE

            population.sort(key=lambda ind: ind['cost'])
            new_population = []
            elites = []
            for i in range(elite_count):
                p = population[i]
                elites.append({'tour': p['tour'][:], 'cost': p['cost']})
            for elite in elites:
                if evaluator.can_eval() and random.random() < 0.75:
                    try:
                        base_cost = evaluator.tour_cost(elite['tour'])
                    except StopIteration:
                        pass
                    else:
                        improved_tour, improved_cost = two_opt_local_search(elite['tour'][:], dist, evaluator=evaluator, max_iterations=500)
                        elite['tour'] = improved_tour
                        elite['cost'] = improved_cost
                new_population.append({'tour': elite['tour'][:], 'cost': elite['cost']})

            while len(new_population) < POP_SIZE and evaluator.can_eval():
                p1 = tournament_select(population, k=TOURNAMENT_K, bias=0.92)
                p2 = tournament_select(population, k=TOURNAMENT_K, bias=0.92)
                if p1 is None or p2 is None:
                    break
                parent1 = p1['tour'][:]
                parent2 = p2['tour'][:]

                if random.random() < crossover_rate:
                    child_tour = order_crossover(parent1, parent2)
                else:
                    child_tour = parent1[:]

                if random.random() < mutation_rate:
                    op = random.choices(
                        [mutate_inversion, mutate_swap, mutate_insertion],
                        weights=[0.5 if div < 0.2 else 0.3, 0.3, 0.2],
                        k=1
                    )[0]
                    op(child_tour)

                child_tour = ensure_valid_tour(child_tour, n)

                apply_memetic = (random.random() < memetic_rate)
                if apply_memetic and evaluator.can_eval():
                    try:
                        base_cost = evaluator.tour_cost(child_tour)
                    except StopIteration:
                        raise
                    parent_quality = min(p1['cost'], p2['cost'])
                    if parent_quality < population[len(population)//2]['cost']:
                        max_it = 300
                    else:
                        max_it = 120
                    improved_tour, improved_cost = two_opt_local_search(child_tour[:], dist, evaluator=evaluator, max_iterations=max_it)
                    child_tour = improved_tour
                    child_cost = improved_cost
                else:
                    try:
                        child_cost = evaluator.tour_cost(child_tour)
                    except StopIteration:
                        raise

                new_population.append({'tour': child_tour[:], 'cost': child_cost})

            div = population_diversity(new_population)
            if (best_no_improve > NO_IMPROVE_LIMIT//2 or div < 0.12) and evaluator.can_eval():
                immigrants = max(1, POP_SIZE // 8)
                for _ in range(immigrants):
                    if not evaluator.can_eval():
                        break
                    if random.random() < 0.65:
                        t = nearest_neighbor_tour(random.randrange(n), dist)
                    else:
                        t = list(range(n))
                        random.shuffle(t)
                    try:
                        c = evaluator.tour_cost(t)
                    except StopIteration:
                        break
                    new_population.sort(key=lambda ind: ind['cost'])
                    worst_idx = -1
                    if c < new_population[worst_idx]['cost'] or random.random() < 0.4:
                        new_population[worst_idx] = {'tour': t[:], 'cost': c}

            population = [{'tour': ind['tour'][:], 'cost': ind['cost']} for ind in new_population]

            population.sort(key=lambda ind: ind['cost'])
            if population[0]['cost'] + 1e-12 < best['cost']:
                best = {'tour': population[0]['tour'][:], 'cost': population[0]['cost']}
                best_no_improve = 0
            else:
                best_no_improve += 1

            if best_no_improve >= NO_IMPROVE_LIMIT and evaluator.can_eval():
                replace_cnt = max(1, POP_SIZE // 5)
                for _ in range(replace_cnt):
                    if not evaluator.can_eval():
                        break
                    if random.random() < 0.5:
                        t = list(range(n))
                        random.shuffle(t)
                    else:
                        t = nearest_neighbor_tour(random.randrange(n), dist)
                    try:
                        c = evaluator.tour_cost(t)
                    except StopIteration:
                        break
                    population.sort(key=lambda ind: ind['cost'])
                    population[-1] = {'tour': t[:], 'cost': c}
                best_no_improve = 0

    except StopIteration:
        pass

    try:
        if evaluator.can_eval() and best.get('tour') is not None:
            final_cost = evaluator.tour_cost(best['tour'])
            best['cost'] = final_cost
    except StopIteration:
        pass

    return best['tour']
