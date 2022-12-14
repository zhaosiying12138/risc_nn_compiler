import numpy as np

M = 128; N = 64; K = 64; Mc = 8; Kc = 8; Nr = 2 

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

def riscnn_if(cond, fn1, args1):
    if (cond):
        fn1(*args1);

def riscnn_if_else(cond, fn1, args1, fn2, args2):
    if (cond):
        fn1(*args1);
    else:
        fn2(*args2);

class ExeBlock(object):
    def __init__(self, A_global: np.ndarray, B_global: np.ndarray, C_global: np.ndarray) -> None:
        self.A_global = A_global
        self.B_global = B_global
        self.C_global = C_global
        self.succ = []

    def connect(self, succ_exb):
        self.succ.append(succ_exb)

    def callnext(self, i, j):
        for exb in self.succ:
            exb.run(i, j)

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
    # Set LD_BASE & ST_BASE according to iterator i & j
        riscnn_set_ldst_base(i, j)
    # Load Stage       
        riscnn_if(j == 0, riscnn_load,
            [self.A_local[:], self.A_global[self.block_id * Mc : self.block_id * Mc + Mc, i * Kc : i * Kc + Kc]])
        riscnn_load(self.B_local[:], self.B_global[i * Kc : i * Kc + Kc, j * Nr : j * Nr + Nr])
        riscnn_if_else((i == 0), riscnn_fill_zero, [self.C_local],
            riscnn_load, [self.C_local[:], self.C_global[self.block_id * Mc : self.block_id * Mc + Mc, j * Nr : j * Nr + Nr]])
    # Cal Stage
        riscnn_mat_muladd(self.C_local, self.A_local, self.B_local)
    # Flow Stage
        riscnn_flow_to(self.B_local, self.B_shared)
    # Store Stage
        riscnn_store(self.C_local[:,:], self.C_global[self.block_id * Mc : self.block_id * Mc + Mc, j * Nr : j * Nr + Nr])

        self.callnext(i, j)

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
    # Set LD_BASE & ST_BASE according to iterator i & j
        riscnn_set_ldst_base(i, j)
    # Load Stage
        riscnn_if(j == 0, riscnn_load,
            [self.A_local[:], self.A_global[self.block_id * Mc : self.block_id * Mc + Mc, i * Kc : i * Kc + Kc]])
        riscnn_flow_from(self.B_local[:], self.B_shared[:])
        riscnn_if_else(i == 0, riscnn_fill_zero, [self.C_local],
            riscnn_load, [self.C_local[:], self.C_global[self.block_id * Mc : self.block_id * Mc + Mc, j * Nr : j * Nr + Nr]])
    # Cal Stage
        riscnn_mat_muladd(self.C_local, self.A_local, self.B_local)
    # Store Stage
        riscnn_store(self.C_local[:,:], self.C_global[self.block_id * Mc : self.block_id * Mc + Mc, j * Nr : j * Nr + Nr])

        self.callnext(i, j)

def risc_nn_sim():
    dtype = "float32"
    a_global = np.random.rand(M, K).astype(dtype)
    b_global = np.random.rand(K, N).astype(dtype)
    # Intentionally make output matrix random to test if risc-nn HW initialize it correctly.
    c_global = np.random.rand(M, N).astype(dtype) # c_global = np.empty((M, N), dtype)
    c_ref = a_global @ b_global
    
    # Compute Graph
    env = ExeBlock(a_global, b_global, c_global)
    b_shared = np.empty((Kc, Nr), dtype)
    ebA = ExeBlockA(0, env, b_shared)
    ebBs = []
    for block_id in range(1, M // Mc):
        ebBs.append(ExeBlockB(block_id, env, b_shared))
    for ebB in ebBs:
        ebA.connect(ebB)
    env.connect(ebA)

    for i_k in range(K // Kc):
        for i_n in range(N // Nr):
            env.callnext(i_k, i_n)

    np.testing.assert_allclose(c_global, a_global @ b_global, rtol=1e-5)
    print("Success!")

__main__ = risc_nn_sim()