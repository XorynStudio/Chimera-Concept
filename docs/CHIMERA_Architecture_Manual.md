# CHIMERA-1 Microarchitecture & ISA Reference Manual
**Document Revision:** 2.0 (3D-CFET 4.00 GHz Production Baseline)  
**Lead Architect:** Jack Record (Xoryn Studio / TenRose Silicon Division)  
**Target Clock Frequency:** 4.00 GHz (250.0 ps Clock Period / ~11 FO4 Gate Delays per Stage)  
**Process & Packaging:** Inverted 3D-Stacked 2nm CFET Compute Tile (Top) + Base L3/NPU Die (Bottom) with Backside Power Delivery (BSPD)  
**Core Topology:** `N+1` Unified Elastic CPU/GPU Cores (4-Wide Superscalar `FUSE-4` $\leftrightarrow$ 16-Lane `VEC4` SIMT `WAVE-16` + 1 Permanent `Core G0` 4K120 Anchor GPU)

---

## 1. Executive Summary & Architectural Philosophy

Modern computing suffers from the **Heterogeneous Schism**: CPUs and GPUs are designed as fundamentally incompatible architectures speaking separate instruction sets (`x86/ARM` vs. proprietary shader ISAs), separated by high-latency buses, OS kernel drivers, and duplicate memory hierarchies. Even in conventional APUs, scalar CPU cores sit idle during heavy GPU rendering while GPU shader cores cannot execute low-latency OS/branch control code.

**CHIMERA-1** eliminates the boundary between CPU and GPU at the transistor level. Every **CHIMERA-1 Elastic Core** is a unified, morphing compute engine composed of **16 Elastic `VEC4` (128-bit) Execution Lanes** coupled to a reconfigurable **64 KB "Liquid SRAM" Memory Array** and an integrated **Hardware Raster/Quad-Spawner**.

### 1.1 The Inverted 3D-CFET Stack & 4.00 GHz Timing Closure
To achieve **4.00 GHz (`250.0 ps` period)** within a strict **15W / 30W / 60W** hard power limit without thermal trapping:
1. **Complementary FET (CFET) Stacked Transistors:** N-type and P-type nanosheet transistors are stacked vertically inside each logic gate, cutting internal cell interconnect capacitance by ~50% and enabling 4.00 GHz operation at `0.75V–0.82V`.
2. **Inverted 3D Die Stacking (Hot Compute on Top, Cool SRAM on Bottom):** Unlike early 3D-cached CPUs that placed SRAM on top of hot logic cores, CHIMERA places the **16 MB / 32 MB / 64 MB Unified L3 Cache** and **45 TOPS NPU** on the **Bottom Base Die**, while the **4.00 GHz Elastic Compute Clusters** sit on the **Top Die** in direct contact with the copper integrated heatspreader (IHS) and cooler.
3. **Backside Power Delivery (BSPD):** High-current `VDD_CORE` power rails route through the underside of the wafer via nano-TSVs, isolating power delivery from top-side signal routing.

### 1.2 Silicon Tier Scalability (@ 4.00 GHz)

Each CHIMERA-1 Core houses **16 Lanes**, with each lane containing a **4-wide 32-bit (`VEC4`) / 8-wide 16-bit (`FP16x8`) Fused Multiply-Add (FMA) ALU** capable of **128 FP32 FLOPs/cycle** (**512 GFLOPS FP32 / 1.024 TFLOPS FP16 per core @ 4.00 GHz**). Every tier includes **1 Permanent Anchor GPU Core (`Core G0`)** dedicated to 4K@120Hz desktop composition and zero-cost **OpenFSR-NPU** upscaling.

| Silicon Tier | Core Count (`N+1`) | Total `VEC4` Lanes | FP32 FMA ALUs | **FP32 Compute @ 4.00 GHz** | **FP16 Shader/AI @ 4.00 GHz** | Unified Cache (`L1 / L2 / L3`) | Hard Power / Thermal Limit |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`CHM1-C8+1` (Lite)** | **9 Cores** (`8+1`) | 144 Lanes | 576 ALUs | **4.61 TFLOPS** | **9.22 TFLOPS** | `576KB` / `4MB` / **`16MB`** | **15 W** Hard / `105°C` $T_J$ |
| **`CHM1-C16+1` (Pro)** | **17 Cores** (`16+1`) | 272 Lanes | 1,088 ALUs | **8.70 TFLOPS** | **17.41 TFLOPS** | `1.08MB` / `8MB` / **`32MB`** | **30 W** Hard / `105°C` $T_J$ |
| **`CHM1-C32+1` (Max)** | **33 Cores** (`32+1`) | 528 Lanes | 2,112 ALUs | **16.89 TFLOPS** | **33.79 TFLOPS** | `2.11MB` / `16MB` / **`64MB`** | **60 W** Hard / `105°C` $T_J$ |

---

## 2. Core Execution Personalities & The `+1` Anchor Core (`G0`)

Each Elastic Core transitions dynamically between modes in **1 clock cycle (250 ps)**, governed either by inline `WAVE.SPAWN` / `SOLO.JOIN` instructions or by the **CHIMERA Matrix Control** OS daemon:

* **MODE 0: `FUSE-4` (Desktop Superscalar CPU Mode):**
  * Lanes group into 4-wide superscalar clusters (`[L0..L3]`, `[L4..L7]`, `[L8..L11]`, `[L12..L15]`) executing standard **64-bit RISC-V (`RV64GC`)** instructions at up to 4 instructions/cycle per thread (16 GIPS/thread @ 4.00 GHz).
* **MODE 1: `WAVE-16` (Unified 16-Lane GPU Shader/Compute Mode):**
  * All 16 `VEC4` lanes lock into a 16-wide SIMT wavefront executing `CHIMERA-X` vector/graphics instructions (`VFMA`, `VDOT3`, `BARY.EVAL`, `TEX2D`, `ROP.ZWRITE`) with 4 banked wavefront slots (64 threads) for **0-cycle memory latency hiding**.
* **THE `+1` ANCHOR GPU CORE (`Core G0`):**
  * Permanently hardwired in `WAVE-16` mode (`512 GFLOPS FP32 / 1.024 TFLOPS FP16`) adjacent to the **CHIMERA Vision Display Engine** and **45 TOPS NPU**.
  * Guarantees stutter-free 4K@120Hz desktop composition even when 100% of the main Elastic Cores (`8`, `16`, or `32`) are locked in `FUSE-4` CPU mode.
  * Teams with the **45 TOPS Systolic NPU** to run zero-cost **OpenFSR-NPU** temporal super-resolution and frame generation directly out of the shared L3 cache without stealing cycles from the main rendering cores.

---

## 3. The 64 KB "Liquid SRAM" Architecture

Each CHIMERA-1 Core features **64 KB of Ultra-Wide (2048-bit interface) 4-Bank Liquid SRAM** that physically re-maps its addressing multiplexers in 1 clock cycle:

| Liquid SRAM Sub-Bank (4 × 16 KB) | Configuration in `FUSE-4` (CPU Mode) | Configuration in `WAVE-16` (GPU Mode) |
| :--- | :--- | :--- |
| **Bank 0 (16 KB)** | Scalar & `VEC4` Register Window (`R0–R31` × 4 Threads) | **Wavefront 0 Register File** (8 KB) + **Wavefront 1 Register File** (8 KB) |
| **Bank 1 (16 KB)** | **L1 Data Cache — Way 0** (16 KB, 1-cycle hit) | **Wavefront 2 Register File** (8 KB) + **Wavefront 3 Register File** (8 KB) |
| **Bank 2 (16 KB)** | **L1 Data Cache — Way 1** (16 KB, 1-cycle hit) | **16 KB 2D Morton-Order Texture & Shared Tile Cache** |
| **Bank 3 (16 KB)** | **L1 Data Cache — Way 2** (16 KB, 1-cycle hit) | **16 KB On-Chip Fast Z-Buffer & Color Tile Buffer (ROP)** |

---

## 4. The 4.00 GHz 12-Stage CFET Pipeline (250.0 ps / Stage)

To achieve timing closure at **4.00 GHz** (~11 FO4 gate delays per stage), the CHIMERA-1 pipeline is partitioned into 12 balanced stages:

| Stage | Name | Budget | Hardware Operations |
| :---: | :--- | :---: | :--- |
| **1** | **`F1` — Fetch Address** | 250 ps | PC MUX, Branch Target Buffer (BTB) lookup, or **Hardware Quad-Spawner** 4×4 tile stepping. |
| **2** | **`F2` — I-Cache Read** | 250 ps | 512-bit Instruction Fetch from 32 KB L1 Instruction SRAM (4 × 32-bit bundle). |
| **3** | **`D1` — Pre-Decode** | 250 ps | `RV64GC` + `CHIMERA-X` opcode extraction and 4-wide intra-bundle dependency check. |
| **4** | **`D2` — Fusion & Broadcast** | 250 ps | Routes 4-instruction bundle across `[L0..L3]` (`FUSE-4`) OR broadcasts 1 instruction to all 16 lanes (`WAVE-16`). |
| **5** | **`REG1` — Bank Select** | 250 ps | Performs 0-cycle Wavefront ID pointer swap (`W0..W3`) on L1 texture/cache miss. |
| **6** | **`REG2` — SRAM Read** | 250 ps | Reads 128-bit `VEC4` operands `Va, Vb, Vc` from the active Liquid SRAM bank. |
| **7** | **`EX1` — Multiply / Align** | 250 ps | Single-cycle Int32 ALU/Branch; `FP32x4` / `FP16x8` multiplier tree & exponent alignment. |
| **8** | **`EX2` — Accumulate / Dot** | 250 ps | FMA adder completion, `VDOT3` / `VDOT4` reduction tree, and `BARY.EVAL` plane equation sum. |
| **9** | **`EX3` — Normalize & Clamp**| 250 ps | IEEE rounding, saturation clamp, and Fast Inverse Square Root (`VRSQ`) LUT refinement. |
| **10** | **`MEM1` — Morton Swizzle** | 250 ps | Linear address generation or **2D Morton Z-Order** coordinate bit-interleaving. |
| **11** | **`MEM2` — L1 / TEX / ROP** | 250 ps | Liquid SRAM L1 hit/miss check, 4-tap bilinear `TEX2D` blend, or 16-wide Z-Buffer test (`ROP.ZWRITE`). |
| **12** | **`WB` — Writeback** | 250 ps | Commits 128-bit `VEC4` result back to Liquid SRAM Register Bank or L2/L3 Tile Ring. |
