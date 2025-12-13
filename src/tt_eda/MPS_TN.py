from .GraphTen import TNetwork,TNode
from typing import List
import numpy as np
import opt_einsum as oe
import math
import numpy as np
from typing import List
Array = np.ndarray

class MPS(TNetwork):
    
    def __init__(self,length: int, num_phys_leg: List[int] | int = 1, nodes_ls:List[TNode] | None = None):
        super().__init__()  
        self.length = length
        self.num_phys_leg = num_phys_leg
        self.nodes_ls = nodes_ls
        self.form_netw()
    
    def __repr__(self):
        return f"MPS with (length={self.length}, {self.num_phys_leg} physical leg per node, with {self.nodes_ls})"

    def form_edge(self, p_node: TNode, p_leg: List[int],
                     c_node: TNode, c_leg: List[int]) -> None:
        # if len(p_node.edge_nodes) >= 2: For quantum circuits cannot satisfy!!!
        #     raise ValueError("Cannot add more than two child")
        assert self.check_leg_dims(p_node, p_leg, c_node, c_leg), "Non-equal leg dimensions!"
        p_node.edge_nodes.append(c_node)
        c_node.edge_nodes.append(p_node)

        c_node.open_legs= [i for i in c_node.open_legs if i not in c_leg ]
        c_node.edged_legs.append(c_leg)
        
        p_node.open_legs= [i for i in p_node.open_legs if i not in p_leg ]
        p_node.edged_legs.append(p_leg)

    def form_netw(self):
        if self.nodes_ls[0].leg_dims[0] != 1 and self.nodes_ls[self.length-1].leg_dims[2] != 1:
            raise ValueError("Edge nodes do not have valid dimensions!")
        for i in range(self.length-1):
            if isinstance(self.num_phys_leg,int):
                self.form_edge(self.nodes_ls[i],[self.num_phys_leg+1],self.nodes_ls[i+1],[0])
            else:
                self.form_edge(self.nodes_ls[i],[self.num_phys_leg[i]+1],self.nodes_ls[i+1],[0])




    def braket(self):
        if isinstance(self.num_phys_leg,int):
            L = np.tensordot(self.nodes_ls[0].tensor,self.nodes_ls[0].tensor.conj(),([0,1],[0,1]))
            for i in range(1,self.length):
                L = np.tensordot(L,self.nodes_ls[i].tensor.conj(),([1],[0]))
                L = np.tensordot(L,self.nodes_ls[i].tensor,([0,1],[0,1]))
        else:
            raise NotImplementedError()

        return L

    def frob_norm(self)->float:
        return np.sqrt(self.braket())

    def make_L_canon(self):
        '''Gauging towards the leftmost site of the MPS, i.e., making it left-canonical where the leftmost tensor is the orthogonality center'''
        
        for i in range(self.length-1):
            A = self.nodes_ls[i].tensor
            D_left, d, D_right = A.shape
            A_mat = A.reshape(D_left * d, D_right)
            Q, R = np.linalg.qr(A_mat, mode='reduced')
            r = Q.shape[1]
            self.nodes_ls[i].tensor = Q.reshape(D_left, d, r)
            # rc.is_T_QR(A,Q.reshape(D_left, d, r),R,2)
            self.nodes_ls[i+1].tensor = np.tensordot(R, self.nodes_ls[i+1].tensor, axes=[1, 0])

        # Normalizing the rightmost tensor R (orthogonality center), so that \bra(R|R)=1 the rest (everything to the left of the orthogonality center) is isometric anyway so identities until right end
        R = self.nodes_ls[self.length-1].tensor 
        R /= np.linalg.norm(R)
        self.nodes_ls[self.length-1].tensor = R

    def make_R_canon(self):
        '''Gauging towards the rightmost site of the MPS, i.e., making it right-canonical where the rightmost tensor is the orthogonality center'''
        
        for i in reversed(range(1,self.length)):
            A = self.nodes_ls[i].tensor
            D_left, d, D_right = A.shape
            A_mat = A.reshape(D_left , d*D_right)
            Q, R = np.linalg.qr(A_mat.conj().T, mode='reduced')
            Q = Q.conj().T
            R = R.conj().T
            r = Q.shape[0]
            self.nodes_ls[i].tensor = Q.reshape(r, d, D_right)
            self.nodes_ls[i-1].tensor = np.tensordot(self.nodes_ls[i-1].tensor,R, axes=[2, 0])

        # Normalizing the leftmost tensor (orthogonality center), so that \bra(L|L)=1 the rest (everything to the right of the orthogonality center) is isometric anyway so identities until left end
        R = self.nodes_ls[0].tensor 
        R /= np.linalg.norm(R)
        self.nodes_ls[0].tensor = R
        
    def make_mix_canon(self,orth_cen_idx):
        
        for i in range(orth_cen_idx):
            A = self.nodes_ls[i].tensor
            D_left, d, D_right = A.shape
            A_mat = A.reshape(D_left * d, D_right)
            Q, R = np.linalg.qr(A_mat, mode='reduced')
            r = Q.shape[1]
            self.nodes_ls[i].tensor = Q.reshape(D_left, d, r)
            self.nodes_ls[i+1].tensor = np.tensordot(R, self.nodes_ls[i+1].tensor, axes=[1, 0])
        
        for i in reversed(range(orth_cen_idx+1,self.length)):
            A = self.nodes_ls[i].tensor
            D_left, d, D_right = A.shape
            A_mat = A.reshape(D_left , d*D_right)
            Q, R = np.linalg.qr(A_mat.conj().T, mode='reduced')
            Q = Q.conj().T
            R = R.conj().T
            r = Q.shape[0]
            self.nodes_ls[i].tensor = Q.reshape(r, d, D_right)
            self.nodes_ls[i-1].tensor = np.tensordot(self.nodes_ls[i-1].tensor,R, axes=[2, 0])

        M = self.nodes_ls[orth_cen_idx].tensor
        M /= np.linalg.norm(M)
        self.nodes_ls[orth_cen_idx].tensor = M

    def upd_orth_cen(self,curr_cen,next_cen):
        eps = 1e-12
        assert curr_cen >= 0 and next_cen >=0

        if curr_cen < next_cen:
            for i in range(curr_cen,next_cen):
                A = self.nodes_ls[i].tensor
                D_left, d, D_right = A.shape
                A_mat = A.reshape(D_left * d, D_right)
                Q, R = np.linalg.qr(A_mat, mode='reduced')
                r = Q.shape[1]
                self.nodes_ls[i].tensor = Q.reshape(D_left, d, r)
                self.nodes_ls[i+1].tensor = np.tensordot(R, self.nodes_ls[i+1].tensor, axes=[1, 0])
            
            R = self.nodes_ls[next_cen].tensor 
            R /=(np.linalg.norm(R)+eps)
            self.nodes_ls[next_cen].tensor = R
        else:
            for i in reversed(range(next_cen+1,curr_cen+1)):
                A = self.nodes_ls[i].tensor
                D_left, d, D_right = A.shape
                A_mat = A.reshape(D_left , d*D_right)
                Q, R = np.linalg.qr(A_mat.conj().T, mode='reduced')
                Q = Q.conj().T
                R = R.conj().T
                r = Q.shape[0]
                self.nodes_ls[i].tensor = Q.reshape(r, d, D_right)
                self.nodes_ls[i-1].tensor = np.tensordot(self.nodes_ls[i-1].tensor,R, axes=[2, 0])

            M = self.nodes_ls[next_cen].tensor
            M /= (np.linalg.norm(M) + eps)
            self.nodes_ls[next_cen].tensor = M
            # rc.is_Q_isom(self.nodes_ls[0].tensor,0)

    def apply_SVD_trunc(self,tensor: np.ndarray, reshape_dims: tuple, trunc_err: float = 1E-6) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Performing SVD on a reshaped tensor and truncating
        singular values according to a truncation error tolerance.
        sum(S_kept**2) / sum(S_total**2) >= 1 - trunc_err**2.
        """
        # Maybe put more try except for SVD failure, for some reason this happens sometimes during two-site gradient updates merge split ..., and as of now I dont know the root cause, most likely over/underflow issues
        
        if not isinstance(reshape_dims, tuple) or len(reshape_dims) != 2:
            raise ValueError("reshape_dims must be a tuple of length 2 (rows, cols).")
        if np.prod(reshape_dims) != tensor.size:
            raise ValueError(f"Invalid reshape_dims {reshape_dims}. Product {np.prod(reshape_dims)} "
                            f"does not match tensor size {tensor.size}.")
        if not (0.0 <= trunc_err <= 1.0):
            raise ValueError("trunc_err must be between 0.0 and 1.0.")

        T_mat = tensor.reshape(*reshape_dims)
        try:
            U, S, Vh = np.linalg.svd(T_mat, full_matrices=False)
        except np.linalg.LinAlgError as e:
            print(f"SVD computation failed: {e}")
            raise

        S_sq = S**2
        total_S_sq = np.sum(S_sq)

        if total_S_sq < 1e-28:
            print("Warning: Tensor norm is near zero. Keeping minimal dimension (1).")
            chi_new = 1
            if len(S) == 0:
                print("Warning: SVD resulted in zero singular values.")
                rows, cols = reshape_dims
                min_dim = min(rows, cols)
                U_trunc = U[:, :1] if U.shape[1] > 0 else np.zeros((rows, 1))
                S_trunc = np.zeros(1)
                Vh_trunc = Vh[:1, :] if Vh.shape[0] > 0 else np.zeros((1, cols))
                return U_trunc, S_trunc, Vh_trunc, chi_new

        else:
            cumulative_S_sq_norm = np.cumsum(S_sq) / total_S_sq
            target_cumulative_sum = 1.0 - trunc_err**2
            chi_new = np.searchsorted(cumulative_S_sq_norm, target_cumulative_sum, side='left') + 1
            chi_new = max(1, chi_new)
            chi_new = min(chi_new, len(S))

        U_trunc = U[:, :chi_new]
        S_trunc = S[:chi_new]
        Vh_trunc = Vh[:chi_new, :]

        return U_trunc, S_trunc, Vh_trunc, chi_new




    def calc_basis_amplitude(self,config:List[int]):
        ''' c_s1s2s3,...,sN = <s1s2s3,...,sN|\psi>
            Computing the probabilities globally, if initially the MPS is not normalized (especially for 1-Norm), then 
            the resulting value would be the unnormalized probability. This function would work just fine for 2-Norm pdf which is
            in some canonical form and normalized. 
        '''
        fix_tens = []
        val = 0
        for idx,node in enumerate(self.nodes_ls):
            fix_tens.append(node.tensor[:,config[idx],:].copy())
        
        val = fix_tens[0]
        for i in range(self.length-1):
            val = oe.contract('ab,bc->ac',val,fix_tens[i+1])
            # val = np.tensordot(, axes=([1],[0]))   
        return val
    
    def calc_1Norm_prob_w_condits(self, config: List[int]) -> float:
        ''' 
        The chain rule of probability states the following: 
        p(x_1,x_2,x_3,...,x_N)=p(x_1)p(x_2|x_1)p(x_3|x_1 x_2) .., p(x_N|x_1 x_2 x_3 ... x_{N-1})
        Iteratively computing each term on the RHS allows us to compute the joint probability distribution sequentially. 
        With this, we get to normalize the pdf sequentially and in the end we obtain normalized probabilities without having the
        need to compute some kind of partition function sum_(all configs)P. 

        This is the core function in the sampling, log likelihood computations. Here 1-Norm definition of prob. is used unlike 
        the Born rule
        '''
        R_envs = self.R_envs_gen()
        L = np.ones((1,1))
        cond_probs=[]
        for i in range(self.length):
            if i == self.length-1:
                M = self.nodes_ls[i].tensor
                R = R_envs[i]
                P = oe.contract('ab,bcd,de->ace',L,M,R)
                P = P/np.sum(P)
                P = list(P.flatten())
                cond_probs.append(P[config[i]])
            else:
                M = self.nodes_ls[i].tensor
                R = R_envs[i]
                P = oe.contract('ab,bcd,de->ace',L,M,R)
                P = P/np.sum(P)
                P = list(P.flatten())
                cond_probs.append(P[config[i]])
                M_idx = M[:,config[i],:]
                L = oe.contract('ab,bd->ad',L,M_idx)
                L = L/np.linalg.norm(L)
               
        return np.prod(cond_probs)
    
    def calc_1Norm_conditional_probs(self, config: List[int]) -> List:
        ''' 
        The chain rule of probability states the following: 
        p(x_1,x_2,x_3,...,x_N)=p(x_1)p(x_2|x_1)p(x_3|x_1 x_2) .., p(x_N|x_1 x_2 x_3 ... x_{N-1})
        Iteratively computing each term on the RHS allows us to compute the joint probability distribution sequentially. 
        With this, we get to normalize the pdf sequentially and in the end we obtain normalized probabilities without having the
        need to compute some kind of partition function sum_(all configs)P. 

        This is the core function in the sampling, log likelihood computations. Here 1-Norm definition of prob. is used unlike 
        the Born rule

        Return: List of conditional probs, their product might cause overflow, for example 10^-100 rounds to 0 in floating-point arithmetic
        We will take log of these eventually anyway, then we can take sums of logs.
        '''
        R_envs = self.R_envs_gen()
        L = np.ones((1,1))
        cond_probs=[]
        for i in range(self.length):
            if i == self.length-1:
                M = self.nodes_ls[i].tensor
                R = R_envs[i]
                P = oe.contract('ab,bcd,de->ace',L,M,R)
                P = P/np.sum(P)
                P = list(P.flatten())
                cond_probs.append(P[config[i]])
            else:
                M = self.nodes_ls[i].tensor
                R = R_envs[i]
                P = oe.contract('ab,bcd,de->ace',L,M,R)
                P = P/np.sum(P)
                P = list(P.flatten())
                cond_probs.append(P[config[i]])
                M_idx = M[:,config[i],:]
                L = oe.contract('ab,bd->ad',L,M_idx)
                L = L/np.linalg.norm(L)
               
        return cond_probs

    def calc_2Norm_prob_w_condits(self, config: List[int]) -> float:
        ''' 
        The chain rule of probability states the following: 
        p(x_1,x_2,x_3,...,x_N)=p(x_1)p(x_2|x_1)p(x_3|x_1 x_2) .., p(x_N|x_1 x_2 x_3 ... x_{N-1})
        '''
        cond_probs = self.calc_2Norm_conditional_probs(config)

        return np.prod(cond_probs)

    def calc_2Norm_conditional_probs(self, config: List[int]) -> List:
        ''' 
        The chain rule of probability states the following: 
        p(x_1,x_2,x_3,...,x_N)=p(x_1)p(x_2|x_1)p(x_3|x_1 x_2) .., p(x_N|x_1 x_2 x_3 ... x_{N-1})
        Iteratively computing each term on the RHS allows us to compute the joint probability distribution sequentially. 
        With this, we get to normalize the pdf sequentially and in the end we obtain normalized probabilities without having the
        need to compute some kind of partition function sum_(all configs)P. 

        This is the core function in the sampling, log likelihood computations. Here 2-Norm definition of prob. is used unlike 
        the 1-Norm

        Return: List of conditional probs, their product might cause overflow, for example 10^-100 rounds to 0 in floating-point arithmetic
        We will take log of these eventually anyway, then we can take sums of logs.
        '''

        # self.make_L_canon()
        cond_probs=[]
        v = np.ones((1,))    
        for i, A in enumerate(self.nodes_ls):
            A = A.tensor 
            D_left, d, D_right = A.shape
            probs = np.zeros(d)
            for k in range(d):
                slice_k = A[:, k, :]         
                vec = v @ slice_k            
                probs[k] = np.dot(vec, vec.conj())
            probs /= probs.sum()
            cond_probs.append(probs[config[i]])
            k = config[i]
            v = (v @ A[:, k, :])
            v = v / np.linalg.norm(v)

        return cond_probs

    def seq_calc_partit_func_1Norm(self):
        ''' Computing the 1Norm of PDF in MPS format.
            First sequntially compute the right env. of each site until the first site. Then contract it with the first site MPS
            to have the total partition function Z (norm fac.) calculated 
            This is not sequential normalization!
        '''

        R_env = [None]*self.length   
        for i in reversed(range(self.length)):
            if i == self.length-1:
                Id = np.ones((1,1))
                R_env[i] = Id
            elif i == self.length-2:
                R_env[i] = np.sum(self.nodes_ls[i+1].tensor,axis=1)
            else:
                M_sum = np.sum(self.nodes_ls[i+1].tensor,axis=1)
                R_env[i] = oe.contract('ab,bd->ad',M_sum,R_env[i+1])
                 
        A1 = self.nodes_ls[0].tensor.copy()
        A1 = np.sum(A1,axis=1)
        norm = A1 @ R_env[0]
        return norm

    def calc_1Norm_prob_global(self,config: List[int],normalized: bool = False):
        
        '''A'''

        # Fist sequentially computing the normalization factor of the probability tensor in MPS/TT format if the MPS is not normalized.
        if not normalized:
            Z = self.seq_calc_partit_func_1Norm()

        fix_tens = []
        val = 0
        for idx,node in enumerate(self.nodes_ls):
            fix_tens.append(node.tensor[:,config[idx],:].copy())
        
        val = fix_tens[0]
        for i in range(self.length-1):
            val = oe.contract('ab,bc->ac',val,fix_tens[i+1])

        return val/Z




   
    def R_envs_gen(self)-> List[np.ndarray]:
        '''R environment tensors for computing conditionals, this differs from the config envs. Here we sum over physical indices to the right 
        of the target tensor for tracing out the remaining rand.vars, to compute the conditional probabilities.
        '''
        eps = 1e-12
        R_env = [None]*self.length   
        for i in reversed(range(self.length)):
            if i == self.length-1:
                Id = np.ones((1,1))
                R_env[i] = Id
            elif i == self.length-2:
                R_env[i] = np.sum(self.nodes_ls[i+1].tensor,axis=1)
                norm = np.linalg.norm(R_env[i])
                R_env[i] /= (norm + eps)
            else:
                M_sum = np.sum(self.nodes_ls[i+1].tensor,axis=1)
                R_env[i] = oe.contract('ab,bd->ad',M_sum,R_env[i+1])
                norm = np.linalg.norm(R_env[i])
                R_env[i] /= (norm + eps)
        
        return R_env
    
    def R_envs_config_gen(self,config: List[int])-> List[np.ndarray]:
        '''For a specific multi index configuration, yields a list of R environment tensors'''
        epsilon = 1e-12
        R_env = [None]*self.length   
        for i in reversed(range(self.length)):
            if i == self.length-1:
                Id = np.ones((1,1))
                R_env[i] = Id
            elif i == self.length-2:
                R_env[i] = self.nodes_ls[i+1].tensor[:,config[i+1],:]
                norm = np.linalg.norm(R_env[i])
                R_env[i] = R_env[i]/(norm + epsilon)   
            else:
                M_sum = self.nodes_ls[i+1].tensor[:,config[i+1],:]
                R_env[i] = oe.contract('ab,bd->ad',M_sum,R_env[i+1])
                norm = np.linalg.norm(R_env[i])
                R_env[i] = R_env[i]/(norm + epsilon)
        
        return R_env
    
    def L_envs_gen(self)-> List[np.ndarray]:
        '''Left environment tensors for computing conditionals, this differs from the config envs. Here we sum over physical indices to the left 
        of the target tensor to trace out the remaining rand.vars, to compute the conditional probabilities.
        '''
        eps = 1e-12
        L_env = [None]*self.length   
        for i in range(self.length):
            if i == 0:
                Id = np.ones((1,1))
                L_env[i] = Id
            elif i == 1:
                L_env[i] = np.sum(self.nodes_ls[i-1].tensor,axis=1)
                norm = np.linalg.norm(L_env[i])
                L_env[i] /= (norm + eps)
            else:
                M_sum = np.sum(self.nodes_ls[i-1].tensor,axis=1)
                L_env[i] = oe.contract('ab,bd->ad',L_env[i-1],M_sum)
                norm = np.linalg.norm(L_env[i])
                L_env[i] /= (norm + eps)
        
        return L_env

    def L_envs_config_gen(self,config: List[int])-> List[np.ndarray]:
        '''Generates the left environment tensors for each site.'''
        eps = 1e-12
        L_env = [None]*self.length   
        for i in range(self.length):
            if i == 0:
                Id = np.ones((1,1))
                L_env[i] = Id
            else:
                M_prev = self.nodes_ls[i-1].tensor[:,config[i-1],:]
                L_env[i] = oe.contract('ab,bd->ad',L_env[i-1],M_prev)
                norm = np.linalg.norm(L_env[i])
                L_env[i] /= (norm + eps)
        
        return L_env

    def grad_over_site_samples(self,L_env_sites: List[np.ndarray],R_env_sites: List[np.ndarray])-> np.ndarray:
        """No SVD, or merging sites into rank-4 tensors. Just single site"""
        # We really dont need this actually see the grads inside opt funcs#
        grad_all = []
        for L_env, R_env in zip(L_env_sites,R_env_sites):
            grad = np.outer(L_env,R_env)
            grad_all.append(grad)
        return np.array(grad_all)




    def one_site_batch_grad_default(self, samples: np.ndarray, steps_per_site: int = 1, num_sweeps: int = 1, learning_rate: float = 0.01)-> None:

        ## I am mutating internal state, returning nothing, this stuff will piss off JAX so badly
        """
        Direct MPS one-site gradient-ascent with batch-averaged gradients,
        multiple inner updates per site, and proper environment handling.
        """
        N, L = samples.shape
        grad_clip = 0.5  
        R_envs = [self.R_envs_config_gen(s) for s in samples]
        L_envs = []
        for _ in range(N):
            env = [None] * L
            env[0] = np.ones((1, 1))
            L_envs.append(env)

        for sweep in range(num_sweeps):
            # Left-to-right 
            for site in range(L):
                T = self.nodes_ls[site].tensor 
                for _ in range(steps_per_site):
                    grad = np.zeros_like(T)
                    for idx, sample in enumerate(samples):
                        phys = sample[site]
                        Lm = L_envs[idx][site]   
                        Rm = R_envs[idx][site]
                        G = np.einsum('a,b->ab', Lm.ravel(), Rm.ravel()) 
                        G_norm = np.linalg.norm(G)
                        if G_norm > grad_clip:
                            G = (grad_clip / G_norm) * G
                        grad[:, phys, :] += G

                    grad /= N
                    T += learning_rate * grad
                    T /= np.linalg.norm(T)
                self.nodes_ls[site].tensor = T

                if site < L - 1:
                    for idx, sample in enumerate(samples):
                        phys = sample[site]
                        L_prev = L_envs[idx][site]
                        L_envs[idx][site+1] = L_prev @ T[:, phys, :]
                        L_envs[idx][site+1] /= np.linalg.norm(L_envs[idx][site+1])

            # Right-to-left 
            for site in reversed(range(L)):
                T = self.nodes_ls[site].tensor
                for _ in range(steps_per_site):
                    grad = np.zeros_like(T)
                    for idx, sample in enumerate(samples):
                        phys = sample[site]
                        Lm = L_envs[idx][site]   
                        Rm = R_envs[idx][site]
                        G = np.einsum('a,b->ab', Lm.ravel(), Rm.ravel()) 
                        G_norm = np.linalg.norm(G)
                        if G_norm > grad_clip:
                            G = (grad_clip / G_norm) * G
                        grad[:, phys, :] += G

                    grad /= N
                    T += learning_rate * grad
                    T /= np.linalg.norm(T)
                self.nodes_ls[site].tensor = T

                if site > 0:
                    for idx, sample in enumerate(samples):
                        phys = sample[site]
                        R_prev = R_envs[idx][site]
                        R_envs[idx][site-1] = T[:, phys, :] @ R_prev
                        R_envs[idx][site-1] /= np.linalg.norm(R_envs[idx][site-1])        

    def one_site_batch_grad_weighted_default(self, samples: np.ndarray, steps_per_site: int = 1, num_sweeps: int = 1, learning_rate: float = 0.01, weights: np.ndarray = None):
        """
        Direct MPS one-site gradient-ascent with batch-averaged gradients,
        multiple inner updates per site, and proper environment handling.
        """

        if weights is None or len(weights) != len(samples) or any(w < 0 for w in weights):
            raise ValueError("Proper weights must be provided: non-negative and match number of samples.")
        else:
            weights = np.array(weights, dtype=float)
            weights /= np.sum(weights)  

        N, L = samples.shape
        grad_clip = 0.5  
        R_envs = [self.R_envs_config_gen(s) for s in samples]
        L_envs = []
        for _ in range(N):
            env = [None] * L
            env[0] = np.ones((1, 1))
            L_envs.append(env)

        for sweep in range(num_sweeps):
            # Left-to-right 
            for site in range(L):
                T = self.nodes_ls[site].tensor 
                for _ in range(steps_per_site):
                    grad = np.zeros_like(T)
                    for idx, sample in enumerate(samples):
                        phys = sample[site]
                        Lm = L_envs[idx][site]   
                        Rm = R_envs[idx][site]
                        G = np.einsum('a,b->ab', Lm.ravel(), Rm.ravel()) 
                        G_norm = np.linalg.norm(G)
                        if G_norm > grad_clip:
                            G = (grad_clip / G_norm) * G
                        grad[:, phys, :] += weights[idx] * G

                    grad /= N
                    T += learning_rate * grad
                    T /= np.linalg.norm(T)
                self.nodes_ls[site].tensor = T

                if site < L - 1:
                    for idx, sample in enumerate(samples):
                        phys = sample[site]
                        L_prev = L_envs[idx][site]
                        L_envs[idx][site+1] = L_prev @ T[:, phys, :]
                        L_envs[idx][site+1] /= np.linalg.norm(L_envs[idx][site+1])

            # Right-to-left 
            for site in reversed(range(L)):
                T = self.nodes_ls[site].tensor
                for _ in range(steps_per_site):
                    grad = np.zeros_like(T)
                    for idx, sample in enumerate(samples):
                        phys = sample[site]
                        Lm = L_envs[idx][site]   
                        Rm = R_envs[idx][site]
                        G = np.einsum('a,b->ab', Lm.ravel(), Rm.ravel()) 
                        G_norm = np.linalg.norm(G)
                        if G_norm > grad_clip:
                            G = (grad_clip / G_norm) * G
                        grad[:, phys, :] += weights[idx] * G

                    grad /= N
                    T += learning_rate * grad
                    T /= np.linalg.norm(T)
                self.nodes_ls[site].tensor = T

                if site > 0:
                    for idx, sample in enumerate(samples):
                        phys = sample[site]
                        R_prev = R_envs[idx][site]
                        R_envs[idx][site-1] = T[:, phys, :] @ R_prev
                        R_envs[idx][site-1] /= np.linalg.norm(R_envs[idx][site-1])
        
        return 0

    def one_site_batch_grad_env_norm_weights_bcktrck(
        self, samples, steps_per_site=1, num_sweeps=1, learning_rate=0.01,
        weights=None, 
        grad_clip=5.0,          
        acc_clip=None,                         
        norm_scale=True,                       
        tr_eps=0.05,                           
        backtrack_factor=0.5,                  
        max_backtracks=5,                      
        eps=1e-12):


        # By far the most experimental coordinate descent optimizer for MPS. I have applied bunch of tricks I found either in literature on DL/ML stuff or online blogs etc. 
        # Nothing mathematically rigorous here, not even justified as to why use, seems to work, gotta learn the math at some point.
        # I will explain them down below mostly for myself to remember later on. The user does not have to know any of this. Most is default to False because you gotta play with them depending on the case
        ###
        ### 1. Gradient Calculation (The Loop):
        ###    Computes grad(Tensor) using the frozen L/R environments.
        ###    TRICK: Per-Sample Clipping (grad_clip): Caps contributions from outlier
        ###      samples to prevent them from destabilizing the batch average.
        ###    TRICK: Global Clipping (acc_clip): Caps the total batch gradient norm
        ###    to prevent "exploding gradients" (common in deep tensor chains).
        ###
        ### 2. Optimization Step (The Update):
        ###    TRICK: Norm Scaling (norm_scale): Ignores gradient magnitude. Uses
        ###      only direction, forcing a fixed step size (aka Manifold Optimization style).
        ###    TRICK: Backtracking / 'Trust Region (while loop stuff)':
        ###      Since L/R environments are only valid for the OLD tensor, changing T
        ###      too much invalidates them. We check the relative change (tr_eps).
        ###      If T changes too much, we reject the step, shrink it (backtrack_factor),
        ###      and try again. This enforces a sort of "Trust Region."
        ###
        ### 3. Projection:
        ###     T /= norm(T): Renormalizes the tensor to prevent numerical explosion/vanishing.
        ###      This does NOT enforce Left/Right canonical form (isometry).
        ###      It simply constrains the tensor magnitude to the unit sphere for stability.
        ### USEFUL READS I FOUND : 1) https://www.ruder.io/optimizing-gradient-descent/ 
        ###                        2) Optimization Algorithms on Matrix Manifolds: https://press.princeton.edu/absil?srsltid=AfmBOorz3pPUv66vMVQjsAmQEGes9QyHG0Ou8gHwuxiqf3fsm03jNv4U     


        N, L = samples.shape
        if weights is None:
            w = np.ones(N, dtype=float) / N
        else:
            w = np.asarray(weights, dtype=float)
            w_sum = w.sum()
            if w_sum <= 0:
                w = np.ones(N, dtype=float) / N
            else:
                w /= w_sum

        R_envs = [self.R_envs_config_gen(s) for s in samples]

        L_envs = []
        for _ in range(N):
            env = [None] * L
            env[0] = np.ones((1, 1))
            L_envs.append(env)

        for sweep in range(num_sweeps):
            for site in range(L):
                T = self.nodes_ls[site].tensor
                for _ in range(steps_per_site):
                    grad = np.zeros_like(T)

                    for idx, sample in enumerate(samples):
                        phys = sample[site]
                        Lm = L_envs[idx][site]
                        Rm = R_envs[idx][site]
                        G = self.grad_over_site_samples([Lm], [Rm])[0]

                        G_norm = np.linalg.norm(G) + eps
                        if G_norm > grad_clip:
                            G *= (grad_clip / G_norm)

                        grad[:, phys, :] += w[idx] * G

                    gnorm = np.linalg.norm(grad) + eps
                    if acc_clip is not None and gnorm > acc_clip:        
                        grad *= (acc_clip / gnorm)

                    step_lr = learning_rate
                    if norm_scale:
                        grad *= (learning_rate / gnorm)                   
                        step_lr = 1.0                                     
              
                    T_old = T
                    step = step_lr * grad
                    bt = 0
                    while True:
                        T_try = T_old + step
                        T_try /= (np.linalg.norm(T_try) + eps)
                        if tr_eps is None:
                            T = T_try
                            break
                        rel_change = np.linalg.norm(T_try - T_old) / (np.linalg.norm(T_old) + eps)
                        if rel_change <= tr_eps or bt >= max_backtracks:
                            T = T_try
                            break
                        step *= backtrack_factor
                        bt += 1
                    self.nodes_ls[site].tensor = T

                if site < L - 1:
                    for idx, sample in enumerate(samples):
                        phys = sample[site]
                        L_prev = L_envs[idx][site]
                        L_next = L_prev @ T[:, phys, :]
                        L_envs[idx][site+1] = L_next / (np.linalg.norm(L_next) + eps)

            for site in reversed(range(L)):
                T = self.nodes_ls[site].tensor
                for _ in range(steps_per_site):
                    grad = np.zeros_like(T)
                    for idx, sample in enumerate(samples):
                        phys = sample[site]
                        Lm = L_envs[idx][site]
                        Rm = R_envs[idx][site]
                        G = self.grad_over_site_samples([Lm], [Rm])[0]

                        G_norm = np.linalg.norm(G) + eps
                        if G_norm > grad_clip:
                            G *= (grad_clip / G_norm)

                        grad[:, phys, :] += w[idx] * G

                    gnorm = np.linalg.norm(grad) + eps
                    if acc_clip is not None and gnorm > acc_clip:
                        grad *= (acc_clip / gnorm)

                    step_lr = learning_rate
                    if norm_scale:
                        grad *= (learning_rate / gnorm)
                        step_lr = 1.0

                    T_old = T
                    step = step_lr * grad
                    bt = 0
                    while True:
                        T_try = T_old + step
                        T_try /= (np.linalg.norm(T_try) + eps)

                        if tr_eps is None:
                            T = T_try
                            break

                        rel_change = np.linalg.norm(T_try - T_old) / (np.linalg.norm(T_old) + eps)
                        if rel_change <= tr_eps or bt >= max_backtracks:
                            T = T_try
                            break

                        step *= backtrack_factor
                        bt += 1

                    self.nodes_ls[site].tensor = T

                if site > 0:
                    for idx, sample in enumerate(samples):
                        phys = sample[site]
                        R_prev = R_envs[idx][site]
                        R_next = T[:, phys, :] @ R_prev
                        R_envs[idx][site-1] = R_next / (np.linalg.norm(R_next) + eps)

        return 0





    def two_site_batch_grad_born(self, samples: np.ndarray, grads_per_bond: int = 1, num_sweeps: int = 1, learning_rate: float = 0.01) -> None:
        """
        Two-site gradient-ascent using Born probabilities, DMRG-style sweep with multiple inner descent steps per bond.
        """
        n_samples = samples.shape[0]
        grad_clip = 5.0
        eps = 1E-12
        delta = 1E-6
        R_envs_samples = [self.R_envs_config_gen(s) for s in samples]

        L_envs_samples = []

        for _ in range(n_samples):
            L_env = [np.ones((1,1))] + [None] * (self.length - 1)
            L_envs_samples.append(L_env)

        for _ in range(num_sweeps):
            for k in range(self.length - 1):
                for _ in range(grads_per_bond):
                    A_k = self.nodes_ls[k].tensor
                    A_kp1 = self.nodes_ls[k+1].tensor
                    M = np.tensordot(A_k, A_kp1, axes=([2],[0]))  

                    grad_total = np.zeros_like(M)
                    for idx, sample in enumerate(samples):
                        s_k, s_kp1 = sample[k], sample[k+1]
                        L = L_envs_samples[idx][k]
                        R = R_envs_samples[idx][k+1]
                        psi_n = float(L @ M[:, s_k, s_kp1, :] @ R)
                        dpsi_dM = np.einsum('a,b->ab', L.ravel(), R.ravel())
                        grad_total[:, s_k, s_kp1, :] += (2.0 / psi_n) * dpsi_dM
                    grad_total /= n_samples
                    grad_total -= 2 * M

                    G_norm = np.linalg.norm(grad_total) + eps
                    if G_norm > grad_clip:
                        grad_total *= (grad_clip / G_norm)
                   
                    M += learning_rate * grad_total
                    M /= np.linalg.norm(M)

                D_l, d, _, D_r = M.shape
                M_mat = M.reshape(D_l*d, d*D_r)
                U, S, Vh, chi = self.apply_SVD_trunc(
                    M_mat, (D_l*d, d*D_r), trunc_err=delta)
                U = U.reshape(D_l, d, chi)
                Vh = Vh.reshape(chi, d, D_r)
                S_mat = np.diag(S)

                self.nodes_ls[k].tensor = U
                self.nodes_ls[k+1].tensor = np.tensordot(S_mat, Vh, axes=([1],[0]))
                self.nodes_ls[k+1].tensor /= np.linalg.norm(self.nodes_ls[k+1].tensor)

                
                for idx, sample in enumerate(samples):
                    s_k = sample[k]
                    L_prev = L_envs_samples[idx][k]
                    A = self.nodes_ls[k].tensor[:, s_k, :]
                    L_envs_samples[idx][k+1] = L_prev @ A
                    norm = np.linalg.norm(L_envs_samples[idx][k+1])
                    L_envs_samples[idx][k+1] /= (norm + eps)

            for k in reversed(range(1, self.length)):
                for _ in range(grads_per_bond):
                    A_km1 = self.nodes_ls[k-1].tensor
                    A_k = self.nodes_ls[k].tensor
                    M = np.tensordot(A_km1, A_k, axes=([2],[0]))

                    grad_total = np.zeros_like(M)
                    for idx, sample in enumerate(samples):
                        s_km1, s_k = sample[k-1], sample[k]
                        L = L_envs_samples[idx][k-1]
                        R = R_envs_samples[idx][k]
                        psi_n = float(L @ M[:, s_km1, s_k, :] @ R)
                        dpsi_dM = np.einsum('a,b->ab', L.ravel(), R.ravel())
                        grad_total[:, s_km1, s_k, :] += (2.0 / psi_n) * dpsi_dM
                    grad_total /= n_samples
                    grad_total -= 2 * M

                    G_norm = np.linalg.norm(grad_total) + eps
                    if G_norm > grad_clip:
                        grad_total *= (grad_clip / G_norm)

                    M += learning_rate * grad_total
                    M /= np.linalg.norm(M)

                D_l, d, _, D_r = M.shape
                M_mat = M.reshape(D_l*d, d*D_r)
                U, S, Vh, chi = self.apply_SVD_trunc(
                    M_mat, (D_l*d, d*D_r), trunc_err=delta)
                U = U.reshape(D_l, d, chi)
                Vh = Vh.reshape(chi, d, D_r)
                S_mat = np.diag(S)

                self.nodes_ls[k].tensor = Vh
                self.nodes_ls[k-1].tensor = np.tensordot(U, S_mat, axes=([2],[0]))
                self.nodes_ls[k-1].tensor /= np.linalg.norm(self.nodes_ls[k-1].tensor)

           
                for idx, sample in enumerate(samples):
                    s_k = sample[k]
                    R_prev = R_envs_samples[idx][k]
                    A = self.nodes_ls[k].tensor[:, s_k, :]
                    R_envs_samples[idx][k-1] = A @ R_prev
                    norm = np.linalg.norm(R_envs_samples[idx][k-1]) 
                    R_envs_samples[idx][k-1] = R_envs_samples[idx][k-1]/(norm + eps)

        return 0

    def one_site_batch_grad_born(self, samples: np.ndarray, steps_per_site: int = 1, num_sweeps: int = 1, learning_rate: float = 0.01) -> None:
        """
        One-site gradient-ascent using Born probabilities, DMRG-style sweep (coordinate ascent aka alternating optimization).
        It differs from Stokes and Terilla stuff in that the updated tensor is directly the gradient itself, as in NOT A_new ---> A_old - grad BUT A_new ---> grad 
        """
        N, L = samples.shape 
        eps = 1e-12
        self.make_R_canon() 

        R_envs = [self.R_envs_config_gen(s) for s in samples]
        L_envs = [[None]*L for _ in range(N)]
        for idx in range(N):
            L_envs[idx][0] = np.ones((1,1))

        # Updated sites during a single sweep: (1 -> 2 ... > L-1) + (L -> L-1 ... -> 2)
        # l-to-r sweep ends up with a left-canonical mps
        # r-to-l sweep ends up with a right-canonical mps
        for _ in range(num_sweeps):
            for site in range(L-1):   
                T = self.nodes_ls[site].tensor
                grad = np.zeros_like(T)
                for idx, sample in enumerate(samples):    
                    # Born grad start
                    s_k = sample[site]
                    Lk = L_envs[idx][site]
                    Rk = R_envs[idx][site]
                    psi_n = float(Lk @ T[:, s_k, :] @ Rk)
                    dpsi_dM = np.einsum('a,b->ab', Lk.ravel(), Rk.ravel())

                #     chi[:, phys, :] += weights[idx] * np.outer(np.ravel(Lm), np.ravel(Rm))
                # chi /= (np.linalg.norm(chi) + eps)
                # T = chi/(np.linalg.norm(chi)+eps)
                # self.nodes_ls[site].tensor = T
                # So therefore, dpsi_dM part is the optimal tensor for L2 minimization

                    grad[:, s_k, :] += (2.0 / psi_n) * dpsi_dM
                
                grad /= N
                grad -= 2 * T/(np.linalg.norm(T))
                    # Born grad end

                T += learning_rate * grad
                self.nodes_ls[site].tensor = T
                self.upd_orth_cen(curr_cen=site, next_cen=site + 1)
                env_up = self.nodes_ls[site].tensor
                for idx, sample in enumerate(samples):
                    phys = sample[site]
                    L_prev = L_envs[idx][site]
                    L_next = L_prev @ env_up[:, phys, :]
                    L_envs[idx][site + 1] = L_next
                    norm = np.linalg.norm(L_envs[idx][site+1]) 
                    L_envs[idx][site+1] /= (norm + eps)

        
            # right-to-left
            for site in reversed(range(1,L)):        
                T = self.nodes_ls[site].tensor
                grad = np.zeros_like(T)
                
                for idx, sample in enumerate(samples):
                    # Born grad start
                    s_k = sample[site]
                    Lk = L_envs[idx][site]
                    Rk = R_envs[idx][site]
                    psi_n = float(Lk @ T[:, s_k, :] @ Rk)
                    dpsi_dM = np.einsum('a,b->ab', Lk.ravel(), Rk.ravel())
                    grad[:, s_k, :] += (2.0 / psi_n) * dpsi_dM
                grad /= N
                grad -= 2 * T/(np.linalg.norm(T))
                    # Born grad end

                T += learning_rate * grad
                self.nodes_ls[site].tensor = T
                self.upd_orth_cen(curr_cen=site, next_cen=site - 1)
                env_up = self.nodes_ls[site].tensor
                for idx, sample in enumerate(samples):
                    phys = sample[site]
                    R_prev = R_envs[idx][site]
                    R_next = env_up[:, phys, :] @ R_prev
                    R_envs[idx][site - 1] = R_next
                    norm = np.linalg.norm(R_envs[idx][site - 1])
                    R_envs[idx][site - 1] /= (norm + eps)

        return  0

    def one_site_tsgd_born(self, samples: np.ndarray, num_sweeps: int = 1, learning_rate: float = 0.01) -> None:
        """
        One-site Tangent-space Gradient Optimization (TSGO) as in Sun et al.
        Performs gradient steps on the MPS tensor at each site, respecting the unit-norm constraint by rotating in tangent space.
        """
        N, L = samples.shape 
        eps = 1e-12
        self.make_R_canon() 

        R_envs = [self.R_envs_config_gen(s) for s in samples]
        L_envs = [[None]*L for _ in range(N)]
        for idx in range(N):
            L_envs[idx][0] = np.ones((1,1))

        # Updated sites during a single sweep: (1 -> 2 ... > L-1) + (L -> L-1 ... -> 2)
        # l-to-r sweep ends up with a left-canonical mps
        # r-to-l sweep ends up with a right-canonical mps
        for _ in range(num_sweeps):
            for site in range(L-1):   
                T = self.nodes_ls[site].tensor
                grad = np.zeros_like(T)
                
                for idx, sample in enumerate(samples):    
                    # Born grad start
                    s_k = sample[site]
                    Lk = L_envs[idx][site]
                    Rk = R_envs[idx][site]
                    psi_n = float(Lk @ T[:, s_k, :] @ Rk)
                    dpsi_dM = np.einsum('a,b->ab', Lk.ravel(), Rk.ravel())
                    grad[:, s_k, :] += (2.0 / psi_n) * dpsi_dM
                grad /= N
                grad -= 2 * T/(np.linalg.norm(T))
                    # Born grad end

                # tangent space projection
                inner = np.tensordot(T, grad, axes=([0,1,2],[0,1,2]))
                G_perp = grad - inner * T
                G_norm = np.linalg.norm(G_perp)
                print('l to r',G_norm)
                print(G_norm)
                G_dir = G_perp / G_norm
                theta = math.atan(learning_rate)
                # theta = math.pi/36
                # rotating on unit-sphere
                T = math.cos(theta)*T + math.sin(theta)*G_dir
                self.nodes_ls[site].tensor = T
                
                self.upd_orth_cen(curr_cen=site, next_cen=site + 1)
                env_up = self.nodes_ls[site].tensor
                for idx, sample in enumerate(samples):
                    phys = sample[site]
                    L_prev = L_envs[idx][site]
                    L_next = L_prev @ env_up[:, phys, :]
                    L_envs[idx][site + 1] = L_next
                    # norm = np.linalg.norm(L_envs[idx][site+1]) 
                    # L_envs[idx][site+1] /= (norm + eps)

        
            # right-to-left
            for site in reversed(range(1,L)):        
                T = self.nodes_ls[site].tensor
                grad = np.zeros_like(T)
                
                for idx, sample in enumerate(samples):
                    # Born grad start
                    s_k = sample[site]
                    Lk = L_envs[idx][site]
                    Rk = R_envs[idx][site]
                    psi_n = float(Lk @ T[:, s_k, :] @ Rk)
                    dpsi_dM = np.einsum('a,b->ab', Lk.ravel(), Rk.ravel())
                    grad[:, s_k, :] += (2.0 / psi_n) * dpsi_dM
                grad /= N
                grad -= 2 * T/(np.linalg.norm(T))
                    # Born grad end
                
                inner = np.tensordot(T, grad, axes=([0,1,2],[0,1,2]))
                G_perp = grad - inner * T
                G_norm = np.linalg.norm(G_perp)
                print('r to l',G_norm)
                G_dir = G_perp / G_norm
                theta = math.atan(learning_rate)
                # theta = math.pi/36
                T = math.cos(theta)*T + math.sin(theta)*G_dir
                self.nodes_ls[site].tensor = T

                self.upd_orth_cen(curr_cen=site, next_cen=site - 1)
                env_up = self.nodes_ls[site].tensor
                for idx, sample in enumerate(samples):
                    phys = sample[site]
                    R_prev = R_envs[idx][site]
                    R_next = env_up[:, phys, :] @ R_prev
                    R_envs[idx][site - 1] = R_next
                    # norm = np.linalg.norm(R_envs[idx][site - 1])
                    # R_envs[idx][site - 1] /= (norm + eps)


  
 

    def log_likelihood_1norm(self, config: List[int]) -> float:
        """Calculate the log likelihood using 2-norm probability for a given configuration.
        """
        eps = 1e-12

        if not isinstance(config, (list, np.ndarray)):
            raise TypeError(f"Config must be a list or array, got {type(config)}")
            
        if len(config) != self.length:
            raise ValueError(f"Config length {len(config)} doesn't match MPS length {self.length}")

        phys_dims = [node.leg_dims[1] for node in self.nodes_ls]
        for i, (idx, dim) in enumerate(zip(config, phys_dims)):
            if not isinstance(idx, (int, np.integer)):
                raise TypeError(f"Config index {i} must be integer, got {type(idx)}")
            if idx < 0 or idx >= dim:
                raise ValueError(f"Config index {i}={idx} out of bounds [0,{dim-1}]")

        try:
            cond_probs = self.calc_1Norm_conditional_probs(config)
            if any(p <= 0 for p in cond_probs):
                print(f"Warning: Zero or negative probability encountered for config {config}")
                return -np.inf
                
            logprob = np.sum(np.log(cond_probs))  
            if isinstance(logprob, complex):
                if abs(logprob.imag) > 1e-14:
                    raise ValueError(f"Log probability is complex: {logprob}")
                logprob = logprob.real
                
            if np.isnan(logprob):
                raise ValueError("Log probability is NaN")
            if np.isinf(logprob) and logprob > 0:
                raise ValueError("Log probability is +inf")
                
            return float(logprob)
    
        except ValueError as e:
            print(f"Value error calculating log likelihood for config {config}: {e}")
            return -np.inf
        except Exception as e:
            print(f"Unexpected error calculating log likelihood for config {config}: {e}")
            return -np.inf

    def log_likelihood_1norm_samples(self, samples: List[List[int]]) -> np.ndarray:
        return np.apply_along_axis(self.log_likelihood_1norm, axis=1, arr=samples)

    def log_likelihood_2norm(self, config: List[int]) -> float:
        """Calculate the log likelihood using 2-norm probability for a given configuration.
        """
        eps = 1e-12

        if not isinstance(config, (list, np.ndarray)):
            raise TypeError(f"Config must be a list or array, got {type(config)}")
            
        if len(config) != self.length:
            raise ValueError(f"Config length {len(config)} doesn't match MPS length {self.length}")

        phys_dims = [node.leg_dims[1] for node in self.nodes_ls]
        for i, (idx, dim) in enumerate(zip(config, phys_dims)):
            if not isinstance(idx, (int, np.integer)):
                raise TypeError(f"Config index {i} must be integer, got {type(idx)}")
            if idx < 0 or idx >= dim:
                raise ValueError(f"Config index {i}={idx} out of bounds [0,{dim-1}]")

        try:
            cond_probs = self.calc_2Norm_conditional_probs(config)
            if any(p <= 0 for p in cond_probs):
                print(cond_probs)
                print(f"Warning: Zero or negative probability encountered for config {config}")
                return -np.inf
                
            logprob = np.sum(np.log(cond_probs))  
            if isinstance(logprob, complex):
                if abs(logprob.imag) > 1e-14:
                    raise ValueError(f"Log probability is complex: {logprob}")
                logprob = logprob.real
                
            if np.isnan(logprob):
                raise ValueError("Log probability is NaN")
            if np.isinf(logprob) and logprob > 0:
                raise ValueError("Log probability is +inf")
                
            return float(logprob)
    
        except ValueError as e:
            print(f"Value error calculating log likelihood for config {config}: {e}")
            return -np.inf
        except Exception as e:
            print(f"Unexpected error calculating log likelihood for config {config}: {e}")
            return -np.inf

    def log_likelihood_2norm_samples(self, samples: List[List[int]]) -> np.ndarray:
       
        return np.apply_along_axis(self.log_likelihood_2norm, axis=1, arr=samples)


    def seq_sample_from(self,num_samples: int = 1,seed: int = 5)-> np.ndarray:
        ''' 
        https://doi.org/10.1103/PhysRevB.85.165146: Perfect sampling with unitary tensor networks
        https://tensornetwork.org/mps/algorithms/sampling/
        Instead of using global random number generator (RNG) we use employ explicit state management for reproducibility. With this, for a given seed
        each random operation consumes a key and returns a new key, ensuring reproducibility. Basically, sequentially calling the same function with the
        same seed in the main function yields the same sampling result. This is also useful to compare the sampling results of different sampling algorithms.
        '''
        sampled_multi_indcs = []
        R_envs = self.R_envs_gen()
        rng = np.random.RandomState(seed)

        for _ in range(num_samples):
            sampled_indcs= []
            L = np.ones((1,1))
            for i in range(self.length):
                if i == self.length-1:
                    M = self.nodes_ls[i].tensor
                    R = R_envs[i]
                    # P = oe.contract('abc,cd->abd',M,R)
                    P = oe.contract('ab,bcd,de->ace',L,M,R)
                    # print(P)
                    # P = P/np.sum(P)
                    # assert np.all(P >= 0), "For the sampling probabilities must be non-negative" rng
                    # P = P.flatten()
                    P = np.abs(P)

                    P = P / np.sum(P)
                    if not np.all(P >= 0):
                        print("Assertion failed: For the sampling probabilities must be non-negative")
                        print("P:", P)
                        raise AssertionError("For the sampling probabilities must be non-negative")
                    P = P.flatten()

                    # n = np.random.choice(M.shape[1], p=P) 
                    n = rng.choice(M.shape[1], p=P)    #for reproducibility

                    sampled_indcs.append(n)
                else:
                    M = self.nodes_ls[i].tensor
                    # M_nxt = self.nodes_ls[i+1].tensor
                    R = R_envs[i]
                    # P = oe.contract('abc,cd->abd',M,R)
                    P = oe.contract('ab,bcd,de->ace',L,M,R)
                    # print(P)

                    # P = P/np.sum(P)
                    # print(P)
                    # assert np.all(P >= 0), "For the sampling probabilities must be non-negative"
                    P = np.abs(P)

                    P = P / np.sum(P)
                    if not np.all(P >= 0):
                        print("Assertion failed: For the sampling probabilities must be non-negative")
                        raise AssertionError("For the sampling probabilities must be non-negative")
                    P = P.flatten()

                    # P = P.flatten()
                    # n = np.random.choice(M.shape[1], p=P)
                    n = rng.choice(M.shape[1], p=P)  #for reproducibility

                    sampled_indcs.append(n)
                    M_idx = M[:,n,:]
                    L = oe.contract('ab,bd->ad',L,M_idx)
                    L = L/np.linalg.norm(L)
                    # M_nxt = oe.contract('ac,cde->ade',M,M_nxt)
                    # M_nxt=M_nxt/np.sqrt(P[n])
                    # n = np.random.choice(M.shape[1], p=P)
                    # self.nodes_ls[i+1].tensor = M_nxt
           
            sampled_multi_indcs.append(sampled_indcs)
        
        return np.array(sampled_multi_indcs)

    def seq_sample_dolgov(self,num_samples: int = 1,seed: int = 5) -> np.ndarray:
        # Dolgov: https://arxiv.org/abs/1810.01212 for cont.us though stil lthe same for dsicrete the logic 
        samples = []
        rng = np.random.RandomState(seed)
        R_envs = self.R_envs_gen()
        for _ in range(num_samples):
            Q = np.ones((1, 1))  
            sample = []
            for k in range(self.length):
                core = self.nodes_ls[k].tensor
                Z = R_envs[k]
                probs = []
                for i in range(core.shape[1]):
                    prob = Q @ core[:, i, :] @ Z  
                    probs.append(prob.item())
                probs = np.abs(probs) / np.sum(np.abs(probs))  
                i_k = rng.choice(core.shape[1], p=probs)  
                sample.append(i_k)
                Q = Q @ core[:, i_k, :]  
                Q /= np.linalg.norm(Q)  
            samples.append(sample)
        return np.array(samples)

    def born_sample(self, num_samples: int = 1, seed: int = 5) -> np.ndarray:
        rng = np.random.RandomState(seed)
        samples = []
        for _ in range(num_samples):
            v = np.ones((1,))    
            s = []
            for i, A in enumerate(self.nodes_ls):
                A = A.tensor 
                D_left, d, D_right = A.shape
                probs = np.zeros(d)
                for k in range(d):
                    slice_k = A[:, k, :]         
                    vec = v @ slice_k            
                    probs[k] = np.dot(vec, vec.conj())
                probs /= probs.sum()
                k = rng.choice(d, p=probs)
                s.append(k)
                v = (v @ A[:, k, :])
                v = v / np.linalg.norm(v)
            samples.append(s)
        return np.array(samples)


  


    def gaussian_perturbation(self, sigma: float = 0.01) -> None:
        for node in self.nodes_ls:
            perturbation = np.random.normal(0, sigma, node.tensor.shape)
            node.tensor += perturbation
            node.tensor = np.maximum(node.tensor, 0)  
            node.tensor /= np.linalg.norm(node.tensor)  

    def uniform_perturbation(self, sigma: float = 0.01) -> None:
        for node in self.nodes_ls:
            perturbation = np.random.uniform(-sigma, sigma, node.tensor.shape)
            node.tensor += perturbation
            node.tensor = np.maximum(node.tensor, 0)
            norm = np.linalg.norm(node.tensor)
            if norm > 0:
                node.tensor /= norm
            else:
                node.tensor.fill(1.0 / np.sqrt(node.tensor.size))

    def mix_with_uniform(self, alpha: float = 0.05) -> None:
        for node in self.nodes_ls:
            uniform = np.ones_like(node.tensor)
            uniform /= np.linalg.norm(uniform)
            node.tensor = (1 - alpha) * node.tensor + alpha * uniform
            node.tensor /= np.linalg.norm(node.tensor)






    @classmethod
    def uniform_Born_pdf_MPS(cls, length: int, num_phys_leg: List[int] | int = 1, const_chi: int = 1, phys_leg_dim: List[int] | int = 2) -> 'MPS':
        '''Creates an MPS representing a uniform probability distribution (w.r.t Born rule) over all configurations.
        P(x) = 1/d^N for all x in {0,...,d-1}^N
        where d is the physical leg dimension and N the number of sites (length).
        This is done by setting each slice of each tensor to 1/sqrt(d) and ensuring proper bond dimensions.
        '''
        nodes = []

        if isinstance(phys_leg_dim, int):
            phys_dims = [phys_leg_dim] * length
        else:
            phys_dims = phys_leg_dim
      
        for i in range(length):
            if const_chi == 1:
                left_dim, right_dim = 1, 1
            else:
                if i == 0:
                    left_dim, right_dim = 1, const_chi
                elif i == length - 1:
                    left_dim, right_dim = const_chi, 1
                else:
                    left_dim, right_dim = const_chi, const_chi
            
            phys_dim = phys_dims[i]
            node = TNode(rank=3, node_id=f'M{i}', leg_dims=[left_dim, phys_dim, right_dim])
            
            if const_chi == 1:
                node.tensor = np.zeros((left_dim, phys_dim, right_dim))
                node.tensor[0, :, 0] = 1.0 / np.sqrt(phys_dim)
            else:
                node.tensor = np.zeros((left_dim, phys_dim, right_dim))
                if i == 0:
                    node.tensor[0, :, 0] = 1/np.sqrt(phys_dim)
                elif i == length - 1:
                    node.tensor[0, :, 0] = 1/np.sqrt(phys_dim)
                else:
                    for m in range(const_chi):
                        node.tensor[m, :, m] = 1.0 / np.sqrt(phys_dim)
            
            nodes.append(node)
        
        return cls(length, num_phys_leg, nodes)

    @classmethod
    def from_uniform(cls, length: int, num_phys_leg: List[int] | int = 1, const_chi: int = 1, phys_leg_dim: List[int] | int = 2, seed: int = 30) -> 'MPS':
        """Factory method to create an MPS with tensors with random elements from the uniform distribution on the interval (0, 1). 
        This is mostly to initialize the probability density function in MPS format
        Warning: The resulting density is not necessarily a uniform distribution. This is just for initialization with non-negative entries.
        """
        # np.random.seed(seed)
        rng = np.random.RandomState(seed)
        nodes=[]
        if isinstance(num_phys_leg,int):
            for i in range(length):
                if i == 0:
                    leg_dims = [1, phys_leg_dim, const_chi]
                elif i == length - 1:
                    leg_dims = [const_chi, phys_leg_dim, 1]
                else:
                    leg_dims = [const_chi, phys_leg_dim, const_chi]
                node = TNode(rank=3,node_id='M{}'.format(i),leg_dims=leg_dims)
                node.tensor =  np.random.uniform(0, 1, size=node.leg_dims) #np.random.uniform(0, 1, size=node.leg_dims)
                # node.tensor = rng.rand(*node.leg_dims)
                # node.tensor = node.tensor/np.linalg.norm(node.tensor)
                nodes.append(node)
        else:
            raise NotImplementedError

        return cls(length, num_phys_leg, nodes)
        
    @classmethod
    def uniform_pdf_MPS_pad_zero(cls, length: int, num_phys_leg: int | list[int] = 1, const_chi: int = 1, phys_leg_dim: int | list[int] = 2) -> 'MPS':
        '''Creates an MPS representing a uniform probability distribution (w.r.t one-norm definition) over all configurations via padding with zeros
        P(x) = 1/d^N for all x in {0,...,d-1}^N
        where d is the physical leg dimension and N the number of sites (length).
        This is done by setting the first slice of each tensor to 1/d and all other entries to zero.
        Note: This is different from the uniform_Born_pdf_MPS which represents a uniform distribution w.r.t the Born rule.
        '''
        nodes = []

        if isinstance(phys_leg_dim, int):
            phys_dims = [phys_leg_dim] * length
        else:
            phys_dims = phys_leg_dim

        if isinstance(num_phys_leg, int):
            num_phys_legs = [num_phys_leg] * length
        else:
            num_phys_legs = num_phys_leg
        
        for i in range(length):
            if const_chi == 1:
                left_dim, right_dim = 1, 1
            else:
                if i == 0:
                    left_dim, right_dim = 1, const_chi
                elif i == length - 1:
                    left_dim, right_dim = const_chi, 1
                else:
                    left_dim, right_dim = const_chi, const_chi
            
            phys_dim = phys_dims[i]
            node = TNode(rank=3, node_id=f'M{i}', leg_dims=[left_dim, phys_dim, right_dim])
            node.tensor = np.zeros((left_dim, phys_dim, right_dim))
            node.tensor[0, :, 0] = 1.0 / phys_dim  
            nodes.append(node)

        return cls(length, num_phys_legs, nodes)

    @classmethod
    def uniform_pdf_MPS(cls, length: int, num_phys_leg: List[int] | int = 1, const_chi: int = 1, phys_leg_dim: List[int] | int = 2) -> 'MPS':
        '''Creates an MPS representing a uniform probability distribution (w.r.t one-norm definition) over all configurations
        P(x) = 1/d^N for all x in {0,...,d-1}^N
        where d is the physical leg dimension and N the number of sites (length).
        This is done by setting the diagonal entries of each tensor to 1/d and all other entries to zero.
        Note: This is different from the uniform_Born_pdf_MPS which represents a uniform distribution w.r.t the Born rule.
        This differs from uniform_pdf_MPS_pad_zero where only the first slice is set to 1/d and all other entries to zero.
        '''
        nodes = []

        if isinstance(phys_leg_dim, int):
            phys_dims = [phys_leg_dim] * length
        else:
            phys_dims = phys_leg_dim
        
        for i in range(length):
            if const_chi == 1:
                left_dim, right_dim = 1, 1
            else:
                if i == 0:
                    left_dim, right_dim = 1, const_chi
                elif i == length - 1:
                    left_dim, right_dim = const_chi, 1
                else:
                    left_dim, right_dim = const_chi, const_chi
            
            phys_dim = phys_dims[i]
            node = TNode(rank=3, node_id=f'M{i}', leg_dims=[left_dim, phys_dim, right_dim])
            
            if const_chi == 1:
                node.tensor = np.zeros((left_dim, phys_dim, right_dim))
                node.tensor[0, :, 0] = 1.0 / phys_dim
            else:
                node.tensor = np.zeros((left_dim, phys_dim, right_dim))
                if i == 0:
                    node.tensor[0, :, 0] = 1/phys_dim
                elif i == length - 1:
                    node.tensor[0, :, 0] = 1/phys_dim
                else:
                    for m in range(const_chi):
                        node.tensor[m, :, m] = 1.0 / phys_dim
            
            nodes.append(node)
        
        return cls(length, num_phys_leg, nodes)
    
    @classmethod
    def GHZ_pdf_MPS_1Norm(cls, length: int, num_phys_leg: List[int] | int = 1, const_chi: int = 1, phys_leg_dim: List[int] | int = 2) -> 'MPS':
        '''
        Creates an MPS representing the GHZ state probability distribution 
        for 1-Norm definition. Padding with zeros to enlarge the bond dimension
        '''
        nodes = []

        if isinstance(phys_leg_dim, int):
            phys_dims = [phys_leg_dim] * length
        else:
            phys_dims = phys_leg_dim
        
        for i in range(length):
            if i == 0:
                left_dim, right_dim = 1, const_chi
            elif i == length - 1:
                left_dim, right_dim = const_chi, 1
            else:
                left_dim, right_dim = const_chi, const_chi
            
            phys_dim = phys_dims[i]
            node = TNode(rank=3, node_id=f'M{i}', leg_dims=[left_dim, phys_dim, right_dim])
                        
            node.tensor = np.zeros((left_dim, phys_dim, right_dim))
            if i == 0:
                node.tensor[0, 0, 0] = 0.5 
                node.tensor[0, 1, const_chi-1] = 0.5
            elif i == length - 1:
                node.tensor[0, 0, 0] = 1
                node.tensor[const_chi-1, 1, 0] = 1
            else:
                node.tensor[0, 0, 0] = 1
                node.tensor[const_chi-1, 1, const_chi-1] = 1
                
            nodes.append(node)
        
        return cls(length, num_phys_leg, nodes)


