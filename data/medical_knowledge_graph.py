import networkx as nx

class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()

    def add_data(self, entities):
        for entity, entity_type in entities:
            self.graph.add_node(entity, type=entity_type)

    def add_relationship(self, node1, node2, relation):
        self.graph.add_edge(node1, node2, relation=relation)

    def visualize(self):
        # Optionally visualize the graph
        nx.draw(self.graph, with_labels=True)
