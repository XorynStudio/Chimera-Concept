// ============================================================================
// CHIMERA-1 Unified Elastic CPU/GPU Architecture
// File: rtl/chimera_pkg.sv
// Description: Master Silicon Package — ISA Opcodes, Modes, VEC4 Types,
//              and 10-Stage 3.0 GHz Pipeline Structures.
// Target Frequency: 3.00 GHz (333.3 ps period / <= 14 FO4 gate levels/stage)
// ============================================================================

package chimera_pkg;

    // ------------------------------------------------------------------------
    // Core Architectural Parameters
    // ------------------------------------------------------------------------
    parameter int NUM_LANES       = 16;   // 16 Elastic VEC4 Lanes per Core
    parameter int FUSE_WIDTH      = 4;    // 4-Wide Superscalar Fusion in CPU Mode
    parameter int NUM_WAVES       = 4;    // 4 Hardware Wavefront Slots (64 threads)
    parameter int NUM_REGS        = 32;   // V0..V31 per thread/lane
    parameter int SCALAR_BITS     = 32;   // 32-bit Scalar / Single Lane
    parameter int VEC4_BITS       = 128;  // 4 x 32-bit (FP32x4 / INT32x4) or 8 x FP16

    // ------------------------------------------------------------------------
    // Core Execution Personality (Single-Cycle Morphing State)
    // ------------------------------------------------------------------------
    typedef enum logic [1:0] {
        CORE_FUSE4  = 2'b00,  // 4-Wide Superscalar Desktop CPU Mode
        CORE_WAVE16 = 2'b01,  // 16-Lane Lockstep GPU Wavefront Mode
        CORE_HYBRID = 2'b10   // Split: L0..L3 CPU + L4..L15 GPU Wavefront
    } core_mode_e;

    // ------------------------------------------------------------------------
    // Instruction Vector Precision Mode (Bits [25:24])
    // ------------------------------------------------------------------------
    typedef enum logic [1:0] {
        PREC_SCALAR32 = 2'b00, // 32-bit Scalar (.x component)
        PREC_VEC4_32  = 2'b01, // 128-bit VEC4 (4x FP32 / INT32) -> 8 FLOPs/lane
        PREC_VEC8_16  = 2'b10, // 128-bit Packed FP16x8          -> 16 FLOPs/lane
        PREC_SWIZZLE  = 2'b11  // Predicated / Swizzled VEC4
    } prec_mode_e;

    // ------------------------------------------------------------------------
    // 6-Bit Master Opcode Map (Bits [31:26])
    // ------------------------------------------------------------------------
    typedef enum logic [5:0] {
        // Control & Hardware Morphing
        OP_NOP        = 6'h00,
        OP_FUSE_MODE  = 6'h01,
        OP_WAVE_SPAWN = 6'h02,
        OP_SOLO_JOIN  = 6'h03,
        OP_BRANCH     = 6'h04,

        // Scalar & Integer ALU
        OP_MOVI       = 6'h10,
        OP_ADD        = 6'h11,
        OP_SUB        = 6'h12,
        OP_MUL        = 6'h13,
        OP_LOGIC      = 6'h14,

        // VEC4 & FP16x8 TFLOPS Compute Engine
        OP_VADD       = 6'h20,
        OP_VSUB       = 6'h21,
        OP_VMUL       = 6'h22,
        OP_VFMA       = 6'h23, // Vd = Va * Vb + Vc (128 FP32 / 256 FP16 FLOPs/cycle/core)
        OP_VDOT3      = 6'h24,
        OP_VDOT4      = 6'h25,
        OP_VRSQ       = 6'h26,

        // Unified Morton Memory & Native Graphics SFU
        OP_LD         = 6'h30,
        OP_ST         = 6'h31,
        OP_BARY_EVAL  = 6'h32,
        OP_TEX2D      = 6'h33, // Triggers 0-cycle Wave Swap on L1 Cache Miss
        OP_ROP_ZWRITE = 6'h34
    } opcode_e;

    // ------------------------------------------------------------------------
    // 128-Bit VEC4 Data Type (Supports INT32x4, Q16.16x4, FP32x4, FP16x8)
    // ------------------------------------------------------------------------
    typedef struct packed {
        logic [31:0] w;
        logic [31:0] z;
        logic [31:0] y;
        logic [31:0] x;
    } vec4_t;

    // ------------------------------------------------------------------------
    // 32-Bit Instruction Word Format
    // ------------------------------------------------------------------------
    typedef struct packed {
        opcode_e     opcode;   // [31:26] 6-bit Opcode
        prec_mode_e  prec;     // [25:24] Precision / Vector Width
        logic [4:0]  vd;       // [23:19] Destination Register (V0..V31)
        logic [4:0]  va;       // [18:14] Source A Register
        logic [4:0]  vb;       // [13:9]  Source B Register
        logic [4:0]  vc;       // [8:4]   Source C Register (for VFMA / ROP)
        logic [3:0]  func;     // [3:0]   Sub-function / Swizzle / Imm low
    } inst_word_t;

endpackage : chimera_pkg
