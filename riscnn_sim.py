import numpy as np

#M = 128; N = 64; K = 64; Mc = 8; Kc = 8; Nr = 2
M = 16; N = 4; K = 16; Mc = 8; Kc = 8; Nr = 2
MODEL_COMPILE = False

def riscnn_prim_func(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@riscnn_prim_func()
def riscnn_fill_zero(C, get_local_baseaddr, handle):
    if (not MODEL_COMPILE):
        C[:] = 0
    else:
        print("// riscnn_fill_zero", file=handle)
        [m, n] = C.shape
        for i1 in range(n):
            for i0 in range(m):
                C_local_addr = get_local_baseaddr("C_local", i0, i1, m)
                print(f"MOV 0x0, [{C_local_addr}]", file=handle)

@riscnn_prim_func()
def riscnn_mat_muladd(C: np.ndarray, A, B, get_local_baseaddr, handle):
    if (not MODEL_COMPILE):
        C[:] += A @ B
    else:
        print("// riscnn_mat_muladd", file=handle)
        [m, k] = A.shape
        [_, n] = B.shape
        # C(i0, i1) += A(i0, i2) * B(i2, i1)     
        for i0 in range(m):
            for i1 in range(n):
                for(i2) in range(k):
                    A_local_addr = get_local_baseaddr("A_local", i0, i2, m)
                    B_local_addr = get_local_baseaddr("B_local", i2, i1, k)
                    C_local_addr = get_local_baseaddr("C_local", i0, i1, m)
                    print(f"MADD [{C_local_addr}], [{A_local_addr}], [{B_local_addr}]", file=handle)

@riscnn_prim_func()
def riscnn_load(local_mem, global_mem, load_type, global_lda, get_local_baseaddr, handle):
    if (not MODEL_COMPILE):
        local_mem[:] = global_mem[:]
    else:
        print("// riscnn_load", file=handle)
        [m, n] = local_mem.shape
        for i0 in range(n):
            for i1 in range(m):
                global_offset = (i1 + i0 * global_lda) * 4
                if (load_type == "A"):
                    A_local_addr = get_local_baseaddr("A_local", i1, i0, m)
                    print(f"LOAD_A [{A_local_addr}], " + "{" + f"{global_offset}" + "}", file=handle)
                elif (load_type == "B"):
                    B_local_addr = get_local_baseaddr("B_local", i1, i0, m)
                    print(f"LOAD_B [{B_local_addr}], " + "{" + f"{global_offset}" + "}", file=handle)
                elif (load_type == "C"):
                    C_local_addr = get_local_baseaddr("C_local", i1, i0, m)
                    print(f"LOAD_C [{C_local_addr}], " + "{" + f"{global_offset}" + "}", file=handle)
                else:
                    assert(False)

@riscnn_prim_func()
def riscnn_store(local_mem, global_mem, global_lda, get_local_baseaddr, handle):
    if (not MODEL_COMPILE):
        global_mem[:] = local_mem[:]
    else:
        print("// riscnn_store", file=handle)
        [m, n] = local_mem.shape
        for i0 in range(n):
            for i1 in range(m):
                global_offset = (i1 + i0 * global_lda) * 4
                C_local_addr = get_local_baseaddr("C_local", i1, i0, m)
                print(f"STORE [{C_local_addr}], " + "{" + f"{global_offset}" + "}", file=handle)

def riscnn_flow_to(local_mem, shared_mem):
    shared_mem[:] = local_mem[:]

def riscnn_flow_from(local_mem, shared_mem):
# Empty Implementation in real hardware since PE flow directly to local memory
    local_mem[:] = shared_mem[:]

@riscnn_prim_func()
def riscnn_set_ldst_base(block_id, ld_baseaddr_a, ld_baseaddr_b, st_baseaddr):
    if (not MODEL_COMPILE):
        pass
    else:
        print(f"block_id {block_id}: {ld_baseaddr_a}, {ld_baseaddr_b}, {st_baseaddr}")
    pass

@riscnn_prim_func()
def riscnn_if(cond, fn1, args1, handle):
    if (not MODEL_COMPILE):
        if (cond):
            fn1(*args1);
    else:
        print("// Sparse Vector Set", file=handle)
        fn1(*args1);
        print("// Sparse Vector UnSet", file=handle)

@riscnn_prim_func()
def riscnn_if_else(cond, fn1, args1, fn2, args2, handle):
    if (not MODEL_COMPILE):
        if (cond):
            fn1(*args1);
        else:
            fn2(*args2);
    else:
        print("// Sparse Vector Set", file=handle)
        fn1(*args1);
        print("// Sparse Vector UnSet", file=handle)
        print("// Sparse Vector Set", file=handle)
        fn2(*args2);
        print("// Sparse Vector UnSet", file=handle)

class ExeBlock(object):
    def __init__(self, A_global: np.ndarray, B_global: np.ndarray, C_global: np.ndarray) -> None:
        self.global_mem_size = 0
        self.global_mem_symbol = {}
        self.local_mem_size = 0
        self.local_mem_symbol = {}
        self.A_global = A_global
        self.B_global = B_global
        self.C_global = C_global
        self.declare_global_mem(self.A_global, "A_global")
        self.declare_global_mem(self.B_global, "B_global")
        self.declare_global_mem(self.C_global, "C_global")
        self.succ = []

    def declare_global_mem(self, mem, name):
        self.global_mem_symbol[name] = self.global_mem_size
        self.global_mem_size += mem.size * 4    # 4 is the size of float32

    def get_global_baseaddr(self, name, row_start, col_start, lda) -> int:
        #print(f"{name}: ({row_start}, {col_start}), lda={lda}")
        return self.global_mem_symbol[name] + (row_start + col_start * lda) * 4

    def declare_local_mem(self, mem, name):
        self.local_mem_symbol[name] = self.local_mem_size
        if (None is not mem):
            self.local_mem_size += mem.size * 4

    def get_local_baseaddr(self, name, row_start, col_start, lda) -> int:
        #print(f"{name}: ({row_start}, {col_start}), lda={lda}")
        return self.local_mem_symbol[name] + (row_start + col_start * lda) * 4

    def connect(self, succ_exb):
        self.succ.append(succ_exb)

    def callnext(self, i, j):
        for exb in self.succ:
            exb.run(i, j)

class ExeBlockA(ExeBlock):
    static_f = None
    def __init__(self, block_id, env: ExeBlock, B_shared: np.ndarray) -> None:
        super().__init__(env.A_global, env.B_global, env.C_global)
        self.block_id = block_id
        self.B_shared = B_shared
        print(self.global_mem_symbol)
        if (None is ExeBlockA.static_f):
            ExeBlockA.static_f = open("ExeBlockA_ASM.txt", "w")
            print("// ExeBlockA ASM\n", file=ExeBlockA.static_f)

    # Local Memory Declaration
        self.A_local = np.empty((Mc, Kc), dtype="float32")
        self.B_local = np.empty((Kc, Nr), dtype="float32")
        self.C_local = np.empty((Mc, Nr), dtype="float32")
        self.declare_local_mem(self.A_local, "A_local")
        self.declare_local_mem(self.B_local, "B_local")
        self.declare_local_mem(self.C_local, "C_local")
        self.declare_local_mem(None, "Tmp_local")
        print("ExeBlock_A: " + str(self.local_mem_symbol))

    # i & j are Iterators(or placeholder)
    def run(self, i: int, j: int):
    # Set LD_BASE & ST_BASE according to iterator i & j
        riscnn_set_ldst_base(self.block_id, self.get_global_baseaddr("A_global", self.block_id * Mc, i * Kc, M),
                                self.get_global_baseaddr("B_global", i * Kc, j * Nr, K),
                                self.get_global_baseaddr("C_global", self.block_id * Mc, j * Nr, M))
    # Load Stage
        riscnn_if(j == 0, riscnn_load,
            [self.A_local[:], self.A_global[self.block_id * Mc : self.block_id * Mc + Mc, i * Kc : i * Kc + Kc], "A", M, self.get_local_baseaddr, ExeBlockA.static_f],
            ExeBlockA.static_f)
        riscnn_load(self.B_local[:], self.B_global[i * Kc : i * Kc + Kc, j * Nr : j * Nr + Nr], "B", K, self.get_local_baseaddr, ExeBlockA.static_f)
        riscnn_if_else((i == 0), riscnn_fill_zero, [self.C_local, self.get_local_baseaddr, ExeBlockA.static_f],
            riscnn_load,
            [self.C_local[:], self.C_global[self.block_id * Mc : self.block_id * Mc + Mc, j * Nr : j * Nr + Nr], "C", M, self.get_local_baseaddr, ExeBlockA.static_f],
            ExeBlockA.static_f)
    # Cal Stage
        riscnn_mat_muladd(self.C_local, self.A_local, self.B_local, self.get_local_baseaddr, ExeBlockA.static_f)
    # Flow Stage
        riscnn_flow_to(self.B_local, self.B_shared)
    # Store Stage
        riscnn_store(self.C_local[:,:], self.C_global[self.block_id * Mc : self.block_id * Mc + Mc, j * Nr : j * Nr + Nr], M,
                        self.get_local_baseaddr, ExeBlockA.static_f)

        if (not MODEL_COMPILE):
            self.callnext(i, j)

class ExeBlockB(ExeBlock):
    static_f = None
    def __init__(self, block_id, env: ExeBlock, B_shared: np.ndarray) -> None:
        super().__init__(env.A_global, env.B_global, env.C_global)
        self.block_id = block_id
        self.B_shared = B_shared
        if (None is ExeBlockB.static_f):
            ExeBlockB.static_f = open("ExeBlockB_ASM.txt", "w")
            print("// ExeBlockB ASM\n", file=ExeBlockB.static_f)

    # Local Memory Declaration
        self.C_local = np.empty((Mc, Nr), dtype="float32")
        self.A_local = np.empty((Mc, Kc), dtype="float32")
        self.B_local = np.empty((Kc, Nr), dtype="float32")
        self.declare_local_mem(self.C_local, "C_local")
        self.declare_local_mem(self.A_local, "A_local")
        self.declare_local_mem(self.B_local, "B_local")
        self.declare_local_mem(None, "Tmp_local")
        print("ExeBlock_B: " + str(self.local_mem_symbol))

    def run(self, i: int, j: int):
    # Set LD_BASE & ST_BASE according to iterator i & j
        riscnn_set_ldst_base(self.block_id,
                                self.get_global_baseaddr("A_global", self.block_id * Mc, i * Kc, M),
                                self.get_global_baseaddr("B_global", i * Kc, j * Nr, K),
                                self.get_global_baseaddr("C_global", self.block_id * Mc, j * Nr, M))
    # Load Stage
        riscnn_if(j == 0, riscnn_load,
            [self.A_local[:], self.A_global[self.block_id * Mc : self.block_id * Mc + Mc, i * Kc : i * Kc + Kc], "A", M, self.get_local_baseaddr, ExeBlockB.static_f],
            ExeBlockB.static_f)
        riscnn_flow_from(self.B_local[:], self.B_shared[:])
        riscnn_if_else(i == 0, riscnn_fill_zero, [self.C_local, self.get_local_baseaddr, ExeBlockB.static_f],
            riscnn_load,
            [self.C_local[:], self.C_global[self.block_id * Mc : self.block_id * Mc + Mc, j * Nr : j * Nr + Nr], "C", M, self.get_local_baseaddr, ExeBlockB.static_f],
            ExeBlockB.static_f)
    # Cal Stage
        riscnn_mat_muladd(self.C_local, self.A_local, self.B_local, self.get_local_baseaddr, ExeBlockB.static_f)
    # Store Stage
        riscnn_store(self.C_local[:,:], self.C_global[self.block_id * Mc : self.block_id * Mc + Mc, j * Nr : j * Nr + Nr],
                        M, self.get_local_baseaddr, ExeBlockB.static_f)

        if (not MODEL_COMPILE):
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

    global MODEL_COMPILE
    MODEL_COMPILE = True

    ebA.run(0, 0)
    ebB.run(0, 0)

    print("Success!")

__main__ = risc_nn_sim()