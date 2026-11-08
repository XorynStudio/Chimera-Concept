# CHIMERA-1 SERIES — TARGET MICROARCHITECTURE & RESEARCH SPECIFICATION
**Document Number:** XRN-CHM1-ARCH-230  
**Revision:** 2.3 (128 KB 3-Macro Liquid SRAM, 30-Seed Monte Carlo & Discrete-Event Cache/DMA Revision)  
**Architect:** Jack Record — Xoryn Studio  
**Target Process Assumptions:** Inverted 3D Stack — Top Compute Tile: 2nm-class CFET (`0.74V` Vdd GPU / `0.92V` Vdd CPU Boost) with Backside Power Delivery (BSPD); Bottom Base Die: 4nm-class FinFET (L3 + NPU + I/O); N4P FinFET (`0.82V`) evaluated as current-node baseline  
**Hardware Prototype Slice:** IHP SG13CMOS5L 130nm Gate-Level ASIC Slice (`tt_um_xorynstudios_chimera` — DRC-Clean GDSII, `484` Standard Cells, `20.66%` Tile Utilisation)

---

## 1. Architectural Overview & The 128 KB 3-Macro Liquid SRAM

The **CHIMERA-1 Series** (`CHM1-C8+1`, `CHM1-C16+1`, `CHM1-C32+1`) is a research and target processor microarchitecture that merges 4-wide superscalar `RV64GC` CPU execution and SIMT `VEC4` GPU compute into a single homogeneous silicon fabric.

To avoid both (a) the L2 write-port saturation of write-through caches (e.g., AMD Bulldozer) and (b) the state-destruction or multi-hundred-cycle dirty-flush penalties of a naive shared L1 macro, each **CHIMERA-1 Elastic Core** contains **exact power-of-two `128 KB` of on-core 6T SRAM (`32 KB + 32 KB + 64 KB`)** partitioned across **three dedicated physical macros**, connected to the cluster L2 via a **`512-bit/cycle` (`64 B/cycle`) bidirectional port**:

| Physical Macro | Size & Physical Geometry | Role in `FUSE-4` (CPU Mode) | Role in `HYBRID-2C8G` (2 CPU + 8 GPU) | Role in `WAVE-16` (GPU Mode) | Transition & Drain Arithmetic (`512-bit/cycle` L2 Port) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Macro 0 (Pinned CPU L1D)** | **`32 KB`** (8-Way Set-Associative 6T `1RW`, `3-cycle` load-to-use) | **`32 KB` Write-Back CPU L1D** (Shared by 4 threads; matches Zen L1D capacity) | **`32 KB` Write-Back CPU L1D** (Dedicated to CPU Threads 0–1) | **`32 KB` Pinned CPU L1D** (Remains asleep/warm; **never overwritten by GPU**) | **0 cycles** flush on `CPU -> GPU`; **`1 cycle`** non-blocking CPU resume on `GPU -> CPU` (CPU stack, globals, and scalar state stay intact). |
| **Macro 1 (Tex L1 + Tile ROP)** | **`32 KB`** (`2 × 16 KB` Sub-Macros: `1A` = 4-Way Cache, `1B` = Ping-Pong Buffer) | **1A (`16 KB`):** Clean L1 Data Way (`3c` hit)<br>**1B (`16 KB`):** Clean Secondary L1 Way (`Dirty = 0`) | **1A (`16 KB`):** Shared L1 Tex/Data (`3c`)<br>**1B (`16 KB`):** `2 × 8 KB` Z/Color Tile Buffer | **1A (`16 KB`):** 2D Morton L1 Texture Cache (`3c` hit)<br>**1B (`16 KB`):** Double-Buffered `2 × 8 KB` (`16×16` px) Z/Color Tile Buffer | Single `8 KB` (`65,536-bit`) tile drains to L2 in **`128 cycles`** (`65,536 ÷ 512`); full `16 KB` (`131,072-bit`) double-buffer drains in **`256 cycles`**. |
| **Macro 2 (Elastic Victim / VGPR)** | **`64 KB`** (**8 × `8 KB` `1RW` 6T Sub-Banks**; each built from 4 × `2 KB` `[64×256b]` standard-aspect macros) | **`64 KB` Clean-Only L1 Victim Cache** (`4-cycle` hit; hardware `Dirty = 0` invariant) | **Sub-Banks 0–3 (`32 KB`):** CPU Clean Victim Cache<br>**Sub-Banks 4–7 (`32 KB`):** `128 VEC4` GPU Registers | **`64 KB` Dynamic Wavefront Register File (`256 VEC4` VGPRs)** across **1 to 16 Resident Wavefronts** | **`1 cycle`** mux flip on `WAVE.SPAWN` (clean victim tags invalidated in 1 clock); **`29 cycles`** measured warm resume + L2 DMA contention on return to CPU. |

### 1.1 Solving 6T `1RW` Sub-Bank Conflicts, Wordline Aspect Ratio & 7-Bit Register Encoding
1. **Standard-Aspect-Ratio Sub-Bank Macros (`4 × 2 KB` per Sub-Bank):** A monolithic `32-word × 2048-bit` SRAM array is too shallow and wide for standard SRAM compilers at `2.50 GHz`. Instead, each of the **8 independent `8 KB` `1RW` sub-banks** (`Sub-Bank 0..7`) is physically synthesized as **4 parallel `2 KB` (`64 words × 256-bit`) standard-compiler 6T macros** striped across the 16 `VEC4` lanes (`4-cycle` hit latency in CPU victim mode; `1-cycle` pipelined throughput via a 2-stage Operand Collector in GPU mode).
2. **7-Bit Register Fields (`0..127` VGPRs/Wave) & Organic Sub-Bank Addressing:**
   * The 32-bit ISA instruction word allocates **7-bit register indices** (`[23:17] Vd`, `[16:10] Va`, `[9:3] Vb`, plus a 3-bit reuse/accumulator selector `[2:0] Vc`), addressing up to **128 `VEC4` registers per wavefront**.
   * All instructions index Macro 2 using the single hardware formula: `Bank_ID = (Reg_Idx + Wave_ID × 3) mod 8`.
   * Rather than synthetic modulo-4 register classes, each wavefront slot combines **EX-to-EX Result Forwarding** (bypassing SRAM reads when a producer instruction completes its 4-cycle ALU pipeline on the current clock) with a **2-Entry Operand Reuse Latch** (`[Vd, Va]`). Across our 25-instruction compiled PBR shader/compute DAG in [sim/chimera_sim.py](file:///c:/Users/Jack/Documents/antigravity/epic-nobel/sim/chimera_sim.py), EX forwarding and the 2-entry reuse latch satisfy **`45.9% – 47.4%`** of source operand reads without touching the SRAM bitlines, keeping organic 8-bank `1RW` port conflicts to **`0.76% – 3.66%`**.

---

## 2. Multi-Seed Behavioural Latency-Hiding & Bandwidth-Contention Model (`sim/chimera_sim.py`)

To evaluate how a 64 KB VGPR file (`1 to 16 resident wavefronts`) hides both a **4-cycle FMA pipeline latency** (with full Read-After-Write scoreboarding) and external DRAM latency/contention across `33 total cores` running at `2.50 GHz` (`576 GB/s` GDDR6 = `17.45 GB/s/core` = **`6.98 Bytes/cycle/core`** sustainable off-chip share), [sim/chimera_sim.py](file:///c:/Users/Jack/Documents/antigravity/epic-nobel/sim/chimera_sim.py) executes a 25-instruction compiled shader/compute dependency graph (`20%` memory loads, `80%` ALU/FMA/Dot-Product math) across **30 independent RNG seeds**:

### 2.1 Study 1A: 30-Seed Monte Carlo Sweep (`Default 72/18/7/3` Cache Hit Mix, `4-Cycle` ALU Latency)
* **Note on Issue vs. True ALU Utilisation:** `Issue Util %` counts all issued instructions (`ALU + MEM`), whereas **`True ALU Util %`** counts strictly math instructions (`VFMA`, `VMUL`, `VADD`, `VDOT`, `BARY.EVAL` = `80%` of the instruction stream) executing on the 16-lane `VEC4` FMA pipes.
* Because of the **4-cycle FMA pipeline latency** and true RAW dependency chains, a single wavefront (`1W`) achieves only **`15.3% ± 4.2%` True ALU Utilisation** (`19.1% ± 5.2%` Issue Util).

| Resident Waves (Threads) | VGPRs / Wave (`64 KB`) | Issue Util % (`Mean ± SD`) | **True ALU Util % (`Mean ± SD`)** | `1RW` Bank Conflict % | EX-Fwd / Reuse Bypass % |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1 Wave (`16 Threads`)** | `128 VEC4` (`32 KB`) | `19.1 % ± 5.2 %` | **`15.3 % ± 4.2 %`** | `0.76 %` | `46.5 %` |
| **2 Waves (`32 Threads`)** | `128 VEC4` (`32 KB`) | `33.5 % ± 7.4 %` | **`26.8 % ± 6.0 %`** | `1.34 %` | `46.5 %` |
| **4 Waves (`64 Threads`)** | `64 VEC4` (`16 KB`) | `56.7 % ± 8.1 %` | **`45.4 % ± 6.5 %`** | `2.27 %` | `46.3 %` |
| **8 Waves (`128 Threads`)** | `32 VEC4` (`8 KB`) | `80.7 % ± 6.7 %` | **`64.5 % ± 5.3 %`** | `3.23 %` | `45.9 %` |
| **12 Waves (`192 Threads`)** | `16 VEC4` (`4 KB`) | `88.3 % ± 3.6 %` | **`70.6 % ± 2.9 %`** | `3.53 %` | `47.4 %` |
| **16 Waves (`256 Threads`)** | `16 VEC4` (`4 KB`) | `91.6 % ± 2.5 %` | **`73.3 % ± 2.0 %`** (of `80%` ceiling) | `3.66 %` | `47.4 %` |

### 2.2 Study 1B: True-ALU Sensitivity Across Cache Hit-Rate Regimes (`30 Seeds`, `256 B` per DRAM Miss)
At `6.98 Bytes/cycle/core` (`576 GB/s ÷ 33 cores ÷ 2.50 GHz`) and `256 Bytes` per uncoalesced `VEC4` DRAM miss, external GDDR6 bandwidth saturates once DRAM misses exceed `~13.6%` of memory loads:

| Hit-Rate Mix (`L1 / L2 / L3 / DRAM`) | **No Bandwidth Cap (`4W / 8W / 16W` True ALU %)** | **With `7.0 B/cycle/core` Cap (`4W / 8W / 16W` True ALU %)** | Primary Bottleneck |
| :--- | :--- | :--- | :--- |
| **Default Cached (`72 / 18 / 7 / 3`)** | `44.1 %` / `66.3 %` / `73.9 %` | **`42.1 %` / `62.6 %` / `72.8 %`** | Latency-bound at 4W; approaches `80%` ALU mix ceiling at 16W |
| **Moderate Mixed (`50 / 20 / 15 / 15`)** | `23.3 %` / `42.8 %` / `66.4 %` | **`21.0 %` / `38.5 %` / `55.6 %`** | Transitions to GDDR6 bandwidth contention above 8 waves |
| **Streaming Heavy (`30 / 20 / 20 / 30`)** | `16.0 %` / `30.9 %` / `53.9 %` | **`13.7 %` / `24.9 %` / `32.1 %`** | **Hard GDDR6 Roofline Bandwidth Wall (`32.1%` True ALU @ 16W)** |

---

## 3. Target SKU Matrix, Die Area Breakdown & Power Budget

| Parameter | **`CHM1-C8+1` (Lite)** | **`CHM1-C16+1` (Pro)** | **`CHM1-C32+1` (Max)** |
| :--- | :--- | :--- | :--- |
| **Target Segment** | Handhelds, Cyberdecks, Synth Workstations, SBCs | Thin Laptops, Mini-PCs, Pro Audio/Video Boxes | Desktops, Consoles, Workstations |
| **Total Core Count (`N+1`)** | **9 Cores** (`8 Elastic` + `1 Anchor GPU G0`) | **17 Cores** (`16 Elastic` + `1 Anchor GPU G0`) | **33 Cores** (`32 Elastic` + `1 Anchor GPU G0`) |
| **Dedicated Display GPU (`G0`)**| 1× Permanent `WAVE-16` Core (`320 GFLOPS` @ 2.5GHz) | 1× Permanent `WAVE-16` Core (`320 GFLOPS` @ 2.5GHz) | 1× Permanent `WAVE-16` Core (`320 GFLOPS` @ 2.5GHz) |
| **Total `VEC4` Lanes / ALUs** | 144 Lanes (576 FP32 ALUs) | 272 Lanes (1,088 FP32 ALUs) | 528 Lanes (2,112 FP32 ALUs) |
| **Resident Wavefronts (Max)** | Up to 144 Waves (2,304 Threads) | Up to 272 Waves (4,352 Threads) | Up to 528 Waves (8,448 Threads) |
| **Max CPU Threads (`FUSE-4`)** | Up to 32 Threads (4-Wide) | Up to 64 Threads (4-Wide) | Up to 128 Threads (4-Wide) |
| **`WAVE-16` Sustained Clock** | **2.50 GHz** (`0.74V` Vdd Target) | **2.50 GHz** (`0.74V` Vdd Target) | **2.50 GHz** (`0.74V` Vdd Target; 3.20 GHz OC) |
| **`FUSE-4` CPU Boost Clock** | **3.80 GHz** (`0.92V` Vdd, ~14 FO4) | **3.80 GHz** (`0.92V` Vdd, ~14 FO4) | **3.80 GHz** (`0.92V` Vdd, ~14 FO4; 4.20 GHz OC) |
| **Sustained FP32 (@ 2.50 GHz)** | **2.88 TFLOPS** | **5.44 TFLOPS** | **10.56 TFLOPS** (13.51 TFLOPS @ 3.2GHz OC) |
| **Peak Boost FP32 (@ 3.80 GHz)**| **4.38 TFLOPS** | **8.27 TFLOPS** | **16.05 TFLOPS** |
| **Sustained FP16 (@ 2.50 GHz)** | **5.76 TFLOPS** | **10.88 TFLOPS** | **21.12 TFLOPS** (32.10 TFLOPS Boost) |
| **Dedicated AI & OpenFSR NPU** | **45 TOPS** INT8/FP8 NPU + `G0` OpenFSR | **45 TOPS** INT8/FP8 NPU + `G0` OpenFSR | **45 TOPS** Dedicated NPU (**87+ TOPS** Combined) |
| **On-Core Liquid SRAM (128KB/C)**| **1.15 MB** (128 KB × 9) + 288 KB I$ | **2.18 MB** (128 KB × 17) + 544 KB I$ | **4.22 MB** (128 KB × 33) + 1.05 MB I$ |
| **L2 Cluster Cache (18c Hit)** | **4 MB** (2 MB × 2 Clusters) | **8 MB** (2 MB × 4 Clusters) | **16 MB** (2 MB × 8 Clusters) |
| **Unified 3D Base-Die L3 (48c)**| **16 MB** | **32 MB** | **64 MB** (Morton Delta DCC) |
| **Memory Interface** | 64-bit LPDDR5X-7500 (`60 GB/s`) | 128-bit CAMM2 / LPDDR5X (`136.5 GB/s`) | **256-bit Close-Coupled GDDR6 (`576 GB/s`) + CAMM2** |
| **PCIe Connectivity** | 12× PCIe 4.0 (8 Usable) | 20× PCIe 4.0 (16 Usable) | **28× PCIe 5.0 Lanes (24 Usable + 4 Chipset)** |
| **Target Power Envelope** | **15 W** (2nm CFET) / **21 W** (N4P) | **29 W** (2nm CFET) / **41 W** (N4P) | **55.8 W Target (60W Envelope on 2nm CFET)** / **80.3 W on N4P** |
| **Total Die Area (`212 mm²` Max)**| **`~64 mm²`** Combined Silicon | **`~118 mm²`** Combined Silicon | **`212.0 mm²`** (`82.5` Cores + `68.0` L2/L3 + `34.0` GDDR6 PHY + `19.5` PCIe/Disp + `8.0` NoC) |
| **Package & Socket Type** | FC-BGA784 (19 × 19 mm, Soldered) | FC-BGA1156 (25 × 25 mm, Soldered) | **Socket CH-X (2,400-Pin Low-Inductance LGA, 42 × 42 mm)** |

### 3.1 Explicit Power & Register-File Dynamic Energy Breakdown (`CHM1-C32+1` @ `2.50 GHz`, 33 Cores)
Reading and writing wide 2,048-bit vector registers across 33 cores is the largest dynamic power component in a SIMT processor. With EX-forwarding and the 2-entry Operand Reuse Latch intercepting `~46.5%` of source reads (`~5,120 active SRAM bits accessed per core per cycle` including writes) and a `73%` True ALU duty cycle:

| Subsystem Power Component | **Target 2nm CFET (`0.74V`, `45 fJ/bit` SRAM)** | **Current N4P FinFET Baseline (`0.82V`, `75 fJ/bit` SRAM)** |
| :--- | :--- | :--- |
| **Macro 2 VGPR + L1 SRAM Access (`5,120 bits/cycle/core`)** | **`19.0 W`** (`33 × 2.5 GHz × 5120 × 45 fJ`) | **`31.7 W`** (`33 × 2.5 GHz × 5120 × 75 fJ`) |
| **2,112 FP32 FMA ALUs (`73%` True ALU Duty Cycle)** | **`11.2 W`** (`1.45 pJ / FLOP`) | **`18.5 W`** (`2.40 pJ / FLOP`) |
| **16 MB L2 + 64 MB L3 + Coherent Mesh NoC** | **`11.5 W`** | **`16.0 W`** |
| **256-Bit GDDR6 PHY (`8 × 32b`) + 28× PCIe 5.0 SerDes** | **`14.2 W`** | **`14.2 W`** |
| **Total Full-SoC Sustained Power (@ `2.50 GHz`)** | **`55.8 W`** (**`189.1 GFLOPS/W`** within `60W` PPT) | **`80.3 W`** (**`131.5 GFLOPS/W`**; requires `85W` TDP or `1.95 GHz` @ `60W`) |

---

## 4. Discrete-Event Cache & DMA Morph Simulation (`Study 2` in `sim/chimera_sim.py`)

Rather than relying on static formulas, `simulate_emergent_morph_roundtrip()` in [sim/chimera_sim.py](file:///c:/Users/Jack/Documents/antigravity/epic-nobel/sim/chimera_sim.py) instantiates cycle-by-cycle set-associative cache state machines (`SetAssociativeCacheSim`, 8-way, 64B lines, tracking per-line `valid` and `dirty` bits) across a **2,000-operation pre-GPU CPU trace** (85% 24 KB hot stack/globals, 15% 80 KB heap, 35% writes), a `WAVE.SPAWN` GPU pass producing an **`8 KB` (`128-line × 64B`) dirty Z/Color tile**, a `SOLO.JOIN` return, and a **400-operation post-GPU CPU trace** sharing the **`512-bit/cycle` (`64 B/cycle`) L2 port**:

| Transition Phase / Overhead Component | **Baseline Naive 64 KB Shared Write-Back L1** | **CHIMERA 128 KB 3-Macro Hierarchy** |
| :--- | :--- | :--- |
| **1. `CPU -> GPU` (`WAVE.SPAWN`)** | **`415 cycles`** (Flushes `398` dirty 64B lines + 16c WCB tail) | **`1 cycle`** (Macro 0 stays pinned; Macro 2 holds only clean victim lines) |
| **2. `GPU -> CPU` Tile Drain (`8 KB` = `128 × 64B`)** | **`128 cycles`** (Blocking drain before L1 can be reused) | **Background DMA** (`128 lines` drained asynchronously over 64 B/c L2 port) |
| **3. Post-GPU CPU Refill & L2 Bus Contention (`400 Ops`)** | **`3,516 cycles`** (Cold L1 refill misses across 400 CPU ops) | **`29 cycles`** (`1c` switch + `28c` L2 bus contention between CPU misses & tile DMA) |
| **Total CPU-Visible Round-Trip Overhead** | **`4,059 cycles`** | **`30 cycles`** (**`158 cycles`** if an immediate re-spawn blocks on the `128c` tile DMA) |

---

## 5. Physical Silicon Hardening & GDSII Tapeout Verification (`tt_um_xorynstudios_chimera`)

To validate the single-cycle `FUSE` $\leftrightarrow$ `WAVE` hardware mode flip, 0-cycle wavefront context switching (`l1_miss_swap`), 2D Morton Z-order address swizzler, FMA datapath, and on-chip Z-buffer depth comparator in physical standard-cell silicon, an 8-bit micro-core slice (`tt_um_xorynstudios_chimera`) was synthesized, placed, routed, and DRC-verified on the **IHP SG13CMOS5L 130nm Open PDK** (`sg13cmos5l_stdcell_typ_1p20V_25C`).

| Metric | Measured Post-Route Value | Tool / Foundry Sign-off |
| :--- | :--- | :--- |
| **Target Process / Standard Cell Library** | **IHP SG13CMOS5L (130nm CMOS)** (`1.20V`, `25 °C`) | `sg13cmos5l_stdcell` |
| **RTL Linter & Elaboration** | **0 Warnings / 0 Errors** (`0.013s` walltime) | Verilator `v5.046` |
| **1×1 Tile Core Utilisation** | **`20.662 %`** (79.3% routing & expansion headroom) | LibreLane / OpenROAD |
| **Total Routed Wire Length** | **`12,809 µm` (`12.81 mm`)** | OpenROAD Detailed Router |
| **Active Logic Standard Cells** | **`484 cells`** (`117` AOI/OAI, `87` NAND, `86` Buf, `74` NOR/XNOR, `25` DFF, `25` Dly, `23` Inv, `21` OR/XOR, `16` AND, `10` Mux2) | Yosys / OpenROAD |
| **Decoupling & Fill Cells (`decap` / `fill`)** | **`2,069 cells`** | Post-Route Fill Insertion |
| **Foundry Precheck & DRC Sign-Off** | **10 / 10 Checks Passed (`0 DRC Violations`)** | KLayout SG13CMOS5L Precheck |
