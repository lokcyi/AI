import random as rd
import copy
from matplotlib import pyplot as plt

#染色體:城市
class Location:
    def __init__(self, name, x, y):
        self.name = name
        #座標
        self.loc = (x, y)
    #計算二個城市之間的距離
    def distance_between(self, location2):
        assert isinstance(location2, Location)
        return ((self.loc[0] - location2.loc[0]) ** 2 + (self.loc[1] - location2.loc[1]) ** 2) ** (1 / 2)


def create_locations():
    locations = []
    xs = [8, 50, 18, 35, 90, 40, 84, 74, 34, 40, 60, 74]
    ys = [3, 62, 0, 25, 89, 71, 7, 29, 45, 65, 69, 47]
    cities = ['Z', 'P', 'A', 'K', 'O', 'Y', 'N', 'X', 'G', 'Q', 'S', 'J']
    for x, y, name in zip(xs, ys, cities):
        locations.append(Location(name, x, y))
    return locations, xs, ys, cities

#個體:路徑
class Route:
    def __init__(self, path):
        # path is a list of Location obj
        #path: list, Location集合
        self.path = path
        #length: 路徑的長度
        self.length = self._set_length()

    def _set_length(self):
        total_length = 0
        path_copy = self.path[:]
        from_here = path_copy.pop(0)
        init_node = copy.deepcopy(from_here)
        while path_copy:
            to_there = path_copy.pop(0)
            total_length += to_there.distance_between(from_here)
            from_here = copy.deepcopy(to_there)
        total_length += from_here.distance_between(init_node)
        return total_length


class GeneticAlgo:
    #初始化重要參數
    def __init__(self, locs, level=10, populations=100, variant=3, mutate_percent=0.1, elite_save_percent=0.1):
        #城市集合
        self.locs = locs
        #進化次數，目前設定10
        self.level = level
        #群體大小:組成群體的個體數量
        self.variant = variant
        #群體的數量
        self.populations = populations
        #mutate_percent:突變比率
        self.mutates = int(populations * mutate_percent)
        #最短的幾％路徑被視為精英路徑（
        self.elite = int(populations * elite_save_percent)
    #負責從self.locs(要走的城市)當中隨機生成可行的路徑（每個都要走到），然後回傳這個路徑回來
    def _find_path(self):
        # locs is a list containing all the Location obj
        locs_copy = self.locs[:]
        path = []
        while locs_copy:
            to_there = locs_copy.pop(locs_copy.index(rd.choice(locs_copy)))
            path.append(to_there)
        return path
    #Step 2.建立第一代的路徑群體
    def _init_routes(self):
        routes = []
        for _ in range(self.populations):
            path = self._find_path()
            routes.append(Route(path))
        return routes
    #Step3. 產生下一代
    def _get_next_route(self, routes):
        #排序，依路徑長度排序，愈短的愈前面
        routes.sort(key=lambda x: x.length, reverse=False)
        #取得前面幾%的菁英路徑
        elites = routes[:self.elite][:]
        #用菁英路徑來繁衍子代
        crossovers = self._crossover(elites)
        return crossovers[:] + elites
    #取得繁衍後的子代
    def _crossover(self, elites):
        # Route is a class type
        normal_breeds = []
        mutate_ones = []
        for _ in range(self.populations - self.mutates):
            #繁衍
            father, mother = rd.choices(elites[:4], k=2)
            index_start = rd.randrange(0, len(father.path) - self.variant - 1)
            # list of Location obj
            father_gene = father.path[index_start: index_start + self.variant]
            father_gene_names = [loc.name for loc in father_gene]
            mother_gene = [gene for gene in mother.path if gene.name not in father_gene_names]
            mother_gene_cut = rd.randrange(1, len(mother_gene))
            # create new route path
            next_route_path = mother_gene[:mother_gene_cut] + father_gene + mother_gene[mother_gene_cut:]
            next_route = Route(next_route_path)
            # add Route obj to normal_breeds
            normal_breeds.append(next_route)
            #突變
            # for mutate purpose
            copy_father = copy.deepcopy(father)
            idx = range(len(copy_father.path))
            gene1, gene2 = rd.sample(idx, 2)
            #隨機調換父路徑當中的兩座城市作為突變
            copy_father.path[gene1], copy_father.path[gene2] = copy_father.path[gene2], copy_father.path[gene1]
            mutate_ones.append(copy_father)
        mutate_breeds = rd.choices(mutate_ones, k=self.mutates)
        return normal_breeds + mutate_breeds
    #Step4. 進化
    def evolution(self):
        routes = self._init_routes()
        for _ in range(self.level):
            routes = self._get_next_route(routes)
        routes.sort(key=lambda x: x.length)
        return routes[0].path, routes[0].length


if __name__ == '__main__':
    my_locs, xs, ys, cities = create_locations()
    my_algo = GeneticAlgo(my_locs, level=40, populations=150, variant=2, mutate_percent=0.02, elite_save_percent=0.15)
    best_route, best_route_length = my_algo.evolution()
    best_route.append(best_route[0])
    print([loc.name for loc in best_route], best_route_length)
    print([(loc.loc[0], loc.loc[1]) for loc in best_route], best_route_length)

    fig, ax = plt.subplots()
    ax.plot([loc.loc[0] for loc in best_route], [loc.loc[1] for loc in best_route], 'red', linestyle='-', marker='')
    ax.scatter(xs, ys)
    for i, txt in enumerate(cities):
        ax.annotate(txt, (xs[i], ys[i]))
    plt.show()