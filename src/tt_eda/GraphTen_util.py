import copy 
from collections import deque

def remove_key_from_nest_dic(node_dic,node_to_remove):
        node_to_remove_edges=node_dic[node_to_remove.node_id]
        del node_dic[node_to_remove.node_id]
        for node in node_to_remove_edges:
            node_dic[node].remove(node_to_remove.node_id)
            if not node_dic[node] : del node_dic[node]

def swap_list_vals(ls1,idx1,ls2,idx2):
      for i, j in zip(idx1, idx2):
            ls2[j] = ls1[i]

def flatten(nested_list):
    for item in nested_list:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

def compute_all_shortest_paths(netw):
        graph = copy.deepcopy(netw.edge_net)
        all_paths = {}
        for start in graph:
            paths = {start: [start]}
            queue = deque([start])
            while queue:
                current = queue.popleft()                
                for neighbor in graph.get(current, []):
                    if neighbor not in paths:
                        paths[neighbor] = paths[current] + [neighbor]
                        queue.append(neighbor)                        
            all_paths[start] = paths   
        return all_paths