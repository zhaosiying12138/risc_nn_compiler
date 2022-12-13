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

class ExeBlockA(object):
    def __init__(self, block_id) -> None:
        self.block_id = block_id
    # Local Memory Declaration
        self.C_local = np.empty((Mc, Nr), dtype="float32")
        self.A_local = np.empty((Mc, Kc), dtype="float32")
        self.B_local = np.empty((Kc, Nr), dtype="float32")

    def run(self, A_global: np.ndarray, B_global: np.ndarray, C_global: np.ndarray,
                    B_shared: np.ndarray, i: int, j: int):
    # Load Stage
        if (j == 0):
            riscnn_load(self.A_local[:], A_global[self.block_id * Mc : self.block_id * Mc + Mc, i * Kc : i * Kc + Kc])
        riscnn_load(self.B_local[:], B_global[i * Kc : i * Kc + Kc, j * Nr : j * Nr + Nr])
        riscnn_load(self.C_local[:], C_global[self.block_id * Mc : self.block_id * Mc + Mc, j * Nr : j * Nr + Nr])
    # Cal Stage
        riscnn_mat_muladd(self.C_local, self.A_local, self.B_local)
    # Flow Stage
        riscnn_flow_to(self.B_local, B_shared)
    # Store Stage
        riscnn_store(self.C_local[:,:], C_global[self.block_id * Mc : self.block_id * Mc + Mc, j * Nr : j * Nr + Nr])

class ExeBlockB(object):
    def __init__(self, block_id) -> None:
        self.block_id = block_id
    # Local Memory Declaration
        self.C_local = np.empty((Mc, Nr), dtype="float32")
        self.A_local = np.empty((Mc, Kc), dtype="float32")
        self.B_local = np.empty((Kc, Nr), dtype="float32")

    def run(self, A_global: np.ndarray, B_global: np.ndarray, C_global: np.ndarray,
                    B_shared: np.ndarray, i: int, j: int):
        # Load Stage
        if (j == 0):
            riscnn_load(self.A_local[:], A_global[self.block_id * Mc : self.block_id * Mc + Mc, i * Kc : i * Kc + Kc])
        riscnn_flow_from(self.B_local[:], B_shared[:])
        riscnn_load(self.C_local[:], C_global[self.block_id * Mc : self.block_id * Mc + Mc, j * Nr : j * Nr + Nr])
        # Cal Stage
        riscnn_mat_muladd(self.C_local, self.A_local, self.B_local)
        # Store Stage
        riscnn_store(self.C_local[:,:], C_global[self.block_id * Mc : self.block_id * Mc + Mc, j * Nr : j * Nr + Nr])

def virtual_centric_leader():
    dtype = "float32"
    a_np = np.random.rand(M, K).astype(dtype)
    b_np = np.random.rand(K, N).astype(dtype)
    c_tmm = a_np @ b_np
    c_np = np.empty((M, N), dtype="float32")

    # Compute Graph
    b_shared = np.empty((Kc, Nr), dtype= "float32")
    ebA = ExeBlockA(0)
    ebBs = []
    for block_id in range(1, M // Mc):
        ebBs.append(ExeBlockB(block_id))
    for i_k in range(K // Kc):
        for i_n in range(N // Nr):
            ebA.run(a_np, b_np, c_np, b_shared, i_k, i_n)
            for ebB in ebBs:
                ebB.run(a_np, b_np, c_np, b_shared, i_k, i_n)
    np.testing.assert_allclose(c_np, c_tmm, rtol=1e-5)
    print("Success!")

virtual_centric_leader()