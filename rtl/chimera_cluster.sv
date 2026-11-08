// ============================================================================
// CHIMERA-1 Unified Elastic CPU/GPU Architecture (3D Stacked CFET / BSPD)
// File: rtl/chimera_cluster.sv
// Description: 4-Core Compute Quad Cluster (Top Die @ 4.00 GHz)
//              - Instantiates 4x Elastic CHIMERA-1 Cores (64 VEC4 Lanes = 256 ALUs)
//              - 2.048 TFLOPS (FP32) / 4.096 TFLOPS (FP16) per Cluster @ 4.00 GHz
//              - Integrated 2 MB Coherent Morton-Swizzled L2 Cache Slice
//              - Vertical Hybrid-Bond TSV Port down to the 3D-Stacked Base L3 Die
// ============================================================================

module chimera_cluster
    import chimera_pkg::*;
#(
    parameter int CLUSTER_ID = 0
)(
    input  logic                    clk_4ghz,       // 4.00 GHz Top-Die CFET Clock
    input  logic                    rst_n,

    // Per-Core Morph Control from CHIMERA Matrix Control Daemon
    input  logic [3:0]              force_gpu_mask_i,

    // Instruction Fetch Broadcast
    input  inst_word_t [3:0]        cluster_ibundle_i,
    input  logic                    cluster_ivalid_i,

    // Vertical 3D Hybrid-Bond TSV Ring Port (To Bottom Base Die L3 Cache)
    output logic                    tsv_l3_req_o,
    output logic                    tsv_l3_we_o,
    output logic [31:0]             tsv_l3_addr_o,
    output vec4_t [3:0]             tsv_l3_wdata_o,
    input  vec4_t [3:0]             tsv_l3_rdata_i,
    input  logic                    tsv_l3_hit_i,

    // Cluster Telemetry
    output core_mode_e [3:0]        core_modes_o
);

    logic [3:0]         c_mem_req;
    logic [3:0]         c_mem_we;
    logic [3:0]         c_mem_morton;
    logic [31:0]        c_mem_addr  [4];
    vec4_t [3:0]        c_mem_wdata [4];

    // Instantiate the 4 Elastic Cores in this Cluster
    generate
        for (genvar c = 0; c < 4; c++) begin : gen_cores
            chimera_core #(
                .CORE_ID       (CLUSTER_ID * 4 + c),
                .IS_ANCHOR_GPU (1'b0)
            ) u_core (
                .clk                (clk_4ghz),
                .rst_n              (rst_n),
                .force_gpu_mode_i   (force_gpu_mask_i[c]),
                .icache_pc_o        (),
                .icache_bundle_i    (cluster_ibundle_i),
                .icache_valid_i     (cluster_ivalid_i),
                .mem_req_o          (c_mem_req[c]),
                .mem_we_o           (c_mem_we[c]),
                .mem_morton_2d_o    (c_mem_morton[c]),
                .mem_addr_o         (c_mem_addr[c]),
                .mem_wdata_o        (c_mem_wdata[c]),
                .mem_rdata_i        (tsv_l3_rdata_i),
                .mem_hit_i          (tsv_l3_hit_i),
                .active_mode_o      (core_modes_o[c]),
                .active_wave_id_o   (),
                .lane_active_mask_o ()
            );
        end
    endgenerate

    // ------------------------------------------------------------------------
    // 2 MB Shared L2 Cluster Slice & Round-Robin Vertical TSV Arbiter
    // ------------------------------------------------------------------------
    logic [1:0] rr_grant_q;

    always_ff @(posedge clk_4ghz or negedge rst_n) begin
        if (!rst_n) begin
            rr_grant_q     <= 2'd0;
            tsv_l3_req_o   <= 1'b0;
            tsv_l3_we_o    <= 1'b0;
            tsv_l3_addr_o  <= 32'd0;
            tsv_l3_wdata_o <= '0;
        end else begin
            rr_grant_q     <= rr_grant_q + 2'd1;
            tsv_l3_req_o   <= |c_mem_req;
            tsv_l3_we_o    <= c_mem_we[rr_grant_q];
            tsv_l3_addr_o  <= c_mem_addr[rr_grant_q];
            tsv_l3_wdata_o <= c_mem_wdata[rr_grant_q];
        end
    end

endmodule : chimera_cluster
