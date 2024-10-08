class Graph:
    def __init__(self, data):
        self.data = data

    @classmethod
    def from_nodes_and_edges(cls, nodes, edges):
        data = {"nodes": nodes, "edges": edges}
        return cls(data)  # 使用 cls 创建并返回类的一个实例


# 使用普通的构造函数
g1 = Graph({"nodes": [1, 2, 3], "edges": [(1, 2), (2, 3)]})

# 使用类方法构造实例
g2 = Graph.from_nodes_and_edges([1, 2, 3], [(1, 2), (2, 3)])

print(g1.data)  # {'nodes': [1, 2, 3], 'edges': [(1, 2), (2, 3)]}
print(g2.data)  # {'nodes': [1, 2, 3], 'edges': [(1, 2), (2, 3)]}
