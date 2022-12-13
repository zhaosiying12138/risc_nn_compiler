import numpy as np

M = 64; N = 64; K = 64; Mc = 8; Kc = 8; Nr = 2 

def riscnn_fill_zero(C):
    C[:] = 0

def riscnn_mat_muladd(C, A, B):
    C[:] += A @ B

def riscnn_load(local_mem, global_mem):
    local_mem[:] = global_mem[:]

def riscnn_store(local_mem, global_mem):
    global_mem[:] = local_mem[:]

def riscnn_flow_to(local_mem, shared_mem):
    shared_mem[:] = local_mem[:]

def riscnn_flow_from(local_mem, shared_mem):
# Empty Implementation in real hardware since PE flow directly to local memory
    local_mem[:] = shared_mem[:]

def riscnn_set_ldst_base(i, j):
    pass

class ExeBlock(object):
    def __init__(self, A_global: np.ndarray, B_global: np.ndarray, C_global: np.ndarray) -> None:
        self.predicate = 0
        self.A_global = A_global
        self.B_global = B_global
        self.C_global = C_global

class ExeBlockA(ExeBlock):
    def __init__(self, block_id, env: ExeBlock, B_shared: np.ndarray) -> None:
        super().__init__(env.A_global, env.B_global, env.C_global)
        self.block_id = block_id
        self.B_shared = B_shared
    # Local Memory Declaration
        self.C_local = np.empty((Mc, Nr), dtype="float32")
        self.A_local = np.empty((Mc, Kc), dtype="float32")
        self.B_local = np.empty((Kc, Nr), dtype="float32")

    # i & j are Iterators(or placeholder)
    def run(self, i: int, j: int):
    # Set LD_BASE & ST_BASE according to i & j
        riscnn_set_ldst_base(i, j)
    # Set Sparse Vector
        self.predicate = (j == 0)
    # Load Stage
        if (self.predicate):
            riscnn_load(self.A_local[:], self.A_global[self.block_id * Mc : self.block_id * Mc + Mc, i * Kc : i * Kc + Kc])
        riscnn_load(self.B_local[:], self.B_global[i * Kc : i * Kc + Kc, j * Nr : j * Nr + Nr])
        riscnn_load(self.C_local[:], self.C_global[self.block_id * Mc : self.block_id * Mc + Mc, j * Nr : j * Nr + Nr])
    # Cal Stage
        riscnn_mat_muladd(self.C_local, self.A_local, self.B_local)
    # Flow Stage
        riscnn_flow_to(self.B_local, self.B_shared)
    # Store Stage
        riscnn_store(self.C_local[:,:], self.C_global[self.block_id * Mc : self.block_id * Mc + Mc, j * Nr : j * Nr + Nr])

class ExeBlockB(ExeBlock):
    def __init__(self, block_id, env: ExeBlock, B_shared: np.ndarray) -> None:
        super().__init__(env.A_global, env.B_global, env.C_global)
        self.block_id = block_id
        self.B_shared = B_shared
    # Local Memory Declaration
        self.C_local = np.empty((Mc, Nr), dtype="float32")
        self.A_local = np.empty((Mc, Kc), dtype="float32")
        self.B_local = np.empty((Kc, Nr), dtype="float32")

    def run(self, i: int, j: int):
    # Set Sparse Vector
        self.predicate = (j == 0)
    # Load Stage
        if (self.predicate):
            riscnn_load(self.A_local[:], self.A_global[self.block_id * Mc : self.block_id * Mc + Mc, i * Kc : i * Kc + Kc])
        riscnn_flow_from(self.B_local[:], self.B_shared[:])
        riscnn_load(self.C_local[:], self.C_global[self.block_id * Mc : self.block_id * Mc + Mc, j * Nr : j * Nr + Nr])
    # Cal Stage
        riscnn_mat_muladd(self.C_local, self.A_local, self.B_local)
    # Store Stage
        riscnn_store(self.C_local[:,:], self.C_global[self.block_id * Mc : self.block_id * Mc + Mc, j * Nr : j * Nr + Nr])

def compute_graph():
    dtype = "float32"
    a_np = np.random.rand(M, K).astype(dtype)
    b_np = np.random.rand(K, N).astype(dtype)
    c_tmm = a_np @ b_np
    c_np = np.empty((M, N), dtype="float32")

    # Compute Graph
    env = ExeBlock(a_np, b_np, c_np)
    b_shared = np.empty((Kc, Nr), dtype= "float32")
    ebA = ExeBlockA(0, env, b_shared)
    ebBs = []
    for block_id in range(1, M // Mc):
        ebBs.append(ExeBlockB(block_id, env, b_shared))
    for i_k in range(K // Kc):
        for i_n in range(N // Nr):
            ebA.run(i_k, i_n)
            for ebB in ebBs:
                ebB.run(i_k, i_n)
    np.testing.assert_allclose(c_np, c_tmm, rtol=1e-5)
    print("Success!")

compute_graph()