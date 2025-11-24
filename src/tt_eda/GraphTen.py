import numpy as np
import copy
from opt_einsum import parser
import GraphTen_util as g_util

 
class TNode:
    def __init__(self, rank=2, node_id=None, leg_dims=[], random=False):
        self.rank = rank
        self.leg_dims = leg_dims
        self.random = random
        self.node_id = node_id
        self.edge_nodes = []
        self.open_legs = [i for i in range(rank)]
        self.edged_legs = []
        self._tensor = None
        self.leg_indcs = [i for i in range(rank)]

    def __str__(self):
        return f"TNode(node_id={self.node_id}, rank={self.rank}, leg_dim={self.leg_dims})"

    def __repr__(self):
        return f"TNode(node_id={self.node_id}, rank={self.rank}, leg_dim={self.leg_dims})"

    @property
    def tensor(self):
        return self._tensor
    
    @tensor.setter
    def tensor(self, new_tensor):
        if new_tensor is not None and not isinstance(new_tensor, np.ndarray):
            raise TypeError("Tensor must be of NumPy array type")
        
        self._tensor = new_tensor
        if new_tensor is not None:
            self.rank = new_tensor.ndim
            self.leg_dims = list(new_tensor.shape)
            self.leg_indcs = list(range(self.rank)) 

    def remove_edge(self, edge_node):
        self.edge_nodes = [edge for edge in self.edge_nodes if edge is not edge_node]
     
  

class TNetwork:
    def __init__(self,nodes_ls=None):
        self.nodes_ls = nodes_ls if nodes_ls is not None else []
        self.nodes_dic = {}
        self.edge_net = {}
        self.einsum_dic = {}

    def is_edged(self,node1,node2):
        return node2.node_id in self.edge_net[node1.node_id]
   
    def add_to_nodes_dic(self):

        nested_list = self.nodes_ls
        flattened_list = list(g_util.flatten(nested_list))
        for node in flattened_list:
            self.nodes_dic.update({'{}'.format(node.node_id): node})

    def add_to_edge_net(self,node,node2):
        
        if node.node_id in self.edge_net:
            self.edge_net[node.node_id].append(node2.node_id)
        else:
            self.edge_net.setdefault('{}'.format(node.node_id), [])
            self.edge_net[node.node_id].append(node2.node_id)

    def remove_node_from_edge_net(self,node_to_remove):
        node_to_remove_edges=self.edge_net[node_to_remove.node_id]
        del self.edge_net[node_to_remove.node_id]
        for node in node_to_remove_edges:
            self.edge_net[node].remove(node_to_remove.node_id)
            if not self.edge_net[node] : del self.edge_net[node]

    def show_edged_legs(self,node1,node2):
        leg_curr_idx = self.edge_net[node1.node_id].index(node2.node_id)
        leg_curr_node = node1.edged_legs[leg_curr_idx]
        return leg_curr_node

    def check_leg_dims(self,p_node, p_leg, c_node, c_leg):
        p_leg_dims = [p_node.leg_dims[i] for i in p_leg]
        c_leg_dims = [c_node.leg_dims[i] for i in c_leg]

        return p_leg_dims==c_leg_dims

    def form_edge(self, p_node, p_leg, c_node, c_leg):
    
        assert self.check_leg_dims(p_node, p_leg, c_node, c_leg),"Non-equal leg dimensions!"
        assert set(p_leg).issubset(p_node.open_legs), f"Open legs {p_node.open_legs} of the first node ({p_node.node_id}) does not have the leg {p_leg} to be edged with {c_leg} of {c_node.node_id} "
        assert set(c_leg).issubset(c_node.open_legs), f"Open legs {c_node.open_legs} of the first node ({c_node.node_id}) does not have the leg {c_leg} to be edged with {p_leg} of {p_node.node_id}"
    
        p_node.edge_nodes.append(c_node)
        c_node.edge_nodes.append(p_node)

        c_node.open_legs= [i for i in c_node.open_legs if i not in c_leg ]
        c_node.edged_legs.append(c_leg)
        
        p_node.open_legs= [i for i in p_node.open_legs if i not in p_leg ]
        p_node.edged_legs.append(p_leg)
        
        self.add_to_edge_net(p_node,c_node)
        self.add_to_edge_net(c_node,p_node)
        
    def add_nodes(self, node):
        self.nodes_ls.append(node)
    
    def gen_ctr_str_static(self):

        nested_list = self.nodes_ls
        flattened_list = list(g_util.flatten(nested_list))
        netw_idx = {}
     
        for idx,node in enumerate(flattened_list):
            if not netw_idx:
                index_str = "".join([parser.get_symbol(i) for i in range(node.rank)])
                netw_idx.update({'{}'.format(node.node_id):index_str })
            else:
                accum_idx_str = "".join([netw_idx[flattened_list[i].node_id] for i in range(idx)])
                index_str = "".join(parser.gen_unused_symbols(accum_idx_str, node.rank))
                netw_idx.update({'{}'.format(node.node_id):index_str })


        cpy_edge_net = copy.deepcopy(self.edge_net)
        for node in self.nodes_dic:
            if node in cpy_edge_net:                    # IF EDGE NODES have the same edge twice with different legs (contraction over 2 legs) since dictionary is updated after one letter assignments it is deleted and no chance fpr takig
                                                                    #care of the second leg generalzie or tidy up the edge net dic
                for edge_node in cpy_edge_net[node]:
                    
                    leg1_idx = self.edge_net[node].index(edge_node)
                    leg2_idx = self.edge_net[edge_node].index(node)
                    leg_1 = self.nodes_dic[node].edged_legs[leg1_idx]
                    leg_2 = self.nodes_dic[edge_node].edged_legs[leg2_idx]
                   
                    assert self.check_leg_dims(self.nodes_dic[node], leg_1,self.nodes_dic[edge_node], leg_2), "Non-equal leg dimensions!"
                    
                    node_str=list(netw_idx[node])
                    edge_node_str = list(netw_idx[edge_node])
                    g_util.swap_list_vals(node_str,leg_1,edge_node_str,leg_2)
                    netw_idx[node] = "".join(node_str)
                    netw_idx[edge_node] = "".join(edge_node_str)
            
                g_util.remove_key_from_nest_dic(cpy_edge_net,self.nodes_dic[node])

        return netw_idx
     
    

def test_graph_network():
    M1=TNode(4, 'M1', [4,4,4,4], False)
    M2=TNode(3, 'M2', [4,4,4], False)
    M3=TNode(2, 'M3', [4,4], False)
    TN = TNetwork() 
    TN.add_nodes([M1,M2,M3])
    TN.add_to_nodes_dic()
    TN.form_edge(p_node=M1,p_leg=[1],c_node=M2,c_leg=[0])
    TN.form_edge(p_node=M2,p_leg=[1],c_node=M3,c_leg=[0])
    print(TN.edge_net)
  

if __name__ == "__main__":
    test_graph_network()
    
    
    