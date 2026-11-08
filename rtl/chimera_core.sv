// ============================================================================
// CHIMERA-1 Unified Elastic CPU/GPU Architecture (3D Stacked CFET / BSPD)
// File: rtl/chimera_core.sv
// Description: 16-Lane VEC4 Core RTL (12-Stage 4.00 GHz Pipeline, 250 ps/stage)
//              - 512 GFLOPS (FP32) / 1.024 TFLOPS (FP16) per Core @ 4.00 GHz
//              - Supports IS_ANCHOR_GPU=1 for the Dedicated +1 Display Core (G0)
// ============================================================================

module chimera_core
    import chimera_pkg::*;
#(
    parameter int CORE_ID       = 0,
    parameter bit IS_ANCHOR_GPU = 1'b0  // 1 = Core G0 (Locked in WAVE-16 4K120 Mode)
)(
    input  logic                    clk,          // 4.00 GHz Master Clock (250.0 ps)
    input  logic                    rst_n,

    // Hardware Matrix Control Override (From CHIMERA Matrix Control Daemon)
    input  logic                    force_gpu_mode_i,

    // 512-bit Instruction Fetch Interface
    output logic [31:0]             icache_pc_o,
    input  inst_word_t [3:0]        icache_bundle_i,
    input  logic                    icache_valid_i,

    // 512-bit Coherent L2 / L3 Morton Ring Bus Interface
    output logic                    mem_req_o,
    output logic                    mem_we_o,
    output logic                    mem_morton_2d_o,
    output logic [31:0]             mem_addr_o,
    output vec4_t [3:0]             mem_wdata_o,
    input  vec4_t [3:0]             mem_rdata_i,
    input  logic                    mem_hit_i,

    // Telemetry
    output core_mode_e              active_mode_o,
    output logic [1:0]              active_wave_id_o,
    output logic [15:0]             lane_active_mask_o
);

    core_mode_e         core_mode_q;
    logic [1:0]         active_wave_q;
    logic [31:0]        pc_q [NUM_WAVES];
    logic [15:0]        lane_mask_q;

    assign active_mode_o      = core_mode_q;
    assign active_wave_id_o   = active_wave_q;
    assign lane_active_mask_o = lane_mask_q;
    assign icache_pc_o        = pc_q[active_wave_q];

    // ========================================================================
    // STAGES 1–4 (F1, F2, D1, D2): 4.0 GHz Fetch & 4-Wide Fusion / 16-Wide Mux
    // ========================================================================
    inst_word_t [3:0]   f2_bundle_q;
    inst_word_t         d2_lane_inst_q [NUM_LANES];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            f2_bundle_q <= '0;
            for (int l = 0; l < NUM_LANES; l++) d2_lane_inst_q[l] <= '0;
        end else begin
            f2_bundle_q <= icache_bundle_i;
            for (int l = 0; l < NUM_LANES; l++) begin
                if (core_mode_q == CORE_FUSE4 && !IS_ANCHOR_GPU) begin
                    d2_lane_inst_q[l] <= f2_bundle_q[l % FUSE_WIDTH];
                end else begin
                    d2_lane_inst_q[l] <= f2_bundle_q[0];
                end
            end
        end
    end

    // ========================================================================
    // STAGES 5–6 (REG1, REG2): 64 KB Liquid SRAM & 0-Cycle Wavefront Switcher
    // ========================================================================
    vec4_t      liquid_sram_regs [NUM_WAVES][NUM_LANES][NUM_REGS];
    inst_word_t reg_lane_inst_q  [NUM_LANES];
    vec4_t      reg_va_q         [NUM_LANES];
    vec4_t      reg_vb_q         [NUM_LANES];
    vec4_t      reg_vc_q         [NUM_LANES];
    logic [1:0] reg_wave_id_q;

    wire [1:0] next_wave_id = (!mem_hit_i && core_mode_q == CORE_WAVE16)
                              ? (active_wave_q + 2'd1)
                              : active_wave_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_wave_id_q <= 2'd0;
            for (int l = 0; l < NUM_LANES; l++) begin
                reg_lane_inst_q[l] <= '0;
                reg_va_q[l]        <= '0;
                reg_vb_q[l]        <= '0;
                reg_vc_q[l]        <= '0;
            end
        end else begin
            reg_wave_id_q <= next_wave_id;
            for (int l = 0; l < NUM_LANES; l++) begin
                reg_lane_inst_q[l] <= d2_lane_inst_q[l];
                reg_va_q[l] <= (d2_lane_inst_q[l].va == 5'd0)
                               ? '{w: 32'd0, z: 32'd0, y: (l / 4), x: (l % 4)}
                               : liquid_sram_regs[next_wave_id][l][d2_lane_inst_q[l].va];
                reg_vb_q[l] <= liquid_sram_regs[next_wave_id][l][d2_lane_inst_q[l].vb];
                reg_vc_q[l] <= liquid_sram_regs[next_wave_id][l][d2_lane_inst_q[l].vc];
            end
        end
    end

    // ========================================================================
    // STAGES 7–9 (EX1, EX2, EX3): 16x VEC4 128-Bit Pipelined ALUs (4.00 GHz)
    // ========================================================================
    inst_word_t ex3_lane_inst [NUM_LANES];
    vec4_t      ex3_result    [NUM_LANES];

    generate
        for (genvar g = 0; g < NUM_LANES; g++) begin : gen_vec4_lanes
            chimera_vec4_alu #(
                .LANE_ID(g)
            ) u_lane_alu (
                .clk       (clk),
                .rst_n     (rst_n),
                .lane_en_i (lane_mask_q[g]),
                .inst_i    (reg_lane_inst_q[g]),
                .va_i      (reg_va_q[g]),
                .vb_i      (reg_vb_q[g]),
                .vc_i      (reg_vc_q[g]),
                .inst_o    (ex3_lane_inst[g]),
                .result_o  (ex3_result[g])
            );
        end
    endgenerate

    // ========================================================================
    // STAGES 10–12 (MEM1, MEM2, WB): Morton Swizzle, ROP Z-Test, & Writeback
    // ========================================================================
    function automatic logic [31:0] morton_encode_2d(input logic [15:0] x, input logic [15:0] y);
        logic [31:0] z = 32'd0;
        for (int i = 0; i < 16; i++) begin
            z[2*i]     = x[i];
            z[2*i + 1] = y[i];
        end
        return z;
    endfunction

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            core_mode_q     <= IS_ANCHOR_GPU ? CORE_WAVE16 : CORE_FUSE4;
            active_wave_q   <= 2'd0;
            lane_mask_q     <= IS_ANCHOR_GPU ? 16'hFFFF : 16'h000F;
            mem_req_o       <= 1'b0;
            mem_we_o        <= 1'b0;
            mem_morton_2d_o <= 1'b0;
            mem_addr_o      <= 32'd0;
            mem_wdata_o     <= '0;
        end else begin
            active_wave_q <= next_wave_id;

            // Core G0 (IS_ANCHOR_GPU=1) never leaves WAVE-16 mode!
            if (IS_ANCHOR_GPU || force_gpu_mode_i || ex3_lane_inst[0].opcode == OP_WAVE_SPAWN) begin
                core_mode_q <= CORE_WAVE16;
                lane_mask_q <= 16'hFFFF;
            end else if (ex3_lane_inst[0].opcode == OP_SOLO_JOIN) begin
                core_mode_q <= CORE_FUSE4;
                lane_mask_q <= 16'h000F;
            end

            mem_req_o       <= (ex3_lane_inst[0].opcode inside {OP_LD, OP_ST, OP_TEX2D, OP_ROP_ZWRITE});
            mem_we_o        <= (ex3_lane_inst[0].opcode inside {OP_ST, OP_ROP_ZWRITE});
            mem_morton_2d_o <= (ex3_lane_inst[0].opcode inside {OP_TEX2D, OP_ROP_ZWRITE});
            mem_addr_o      <= (ex3_lane_inst[0].opcode inside {OP_TEX2D, OP_ROP_ZWRITE})
                               ? morton_encode_2d(ex3_result[0].x[15:0], ex3_result[0].y[15:0])
                               : ex3_result[0].x;
            mem_wdata_o[0]  <= ex3_result[0];
            mem_wdata_o[1]  <= ex3_result[1];
            mem_wdata_o[2]  <= ex3_result[2];
            mem_wdata_o[3]  <= ex3_result[3];

            for (int l = 0; l < NUM_LANES; l++) begin
                if (lane_mask_q[l] && ex3_lane_inst[l].vd != 5'd0) begin
                    liquid_sram_regs[active_wave_q][l][ex3_lane_inst[l].vd] <= ex3_result[l];
                end
            end
        end
    end

endmodule : chimera_core
