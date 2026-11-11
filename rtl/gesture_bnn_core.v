`timescale 1ns/1ps

// Standalone serial XNOR-popcount inference core for the Gesture Tiny-BNN.
//
// I_mask[0] is the first row-major pixel of the normalized 32x32 mask.
// Input/weight bit 1 represents +1 and bit 0 represents -1.  One binary
// multiply is evaluated per clock.  The default 96/48 V1 structure takes
// 103153 clocks including the result cycle; other hidden widths scale with
// their binary weight count.  Weight bits are stored in neuron-major 32-bit
// chunks.  A chunk is synchronously prefetched every 32 features so TD can
// infer compact ERAMs instead of thousands of distributed-ROM LUTs.  I_mask
// must remain stable while O_busy is asserted.
module gesture_bnn_core #(
    parameter integer H1_UNITS = 96,
    parameter integer H2_UNITS = 48,
    parameter W1_MEM_FILE = "bnn_w1_32.mem",
    parameter W2_MEM_FILE = "bnn_w2_32.mem",
    parameter W3_MEM_FILE = "bnn_w3_32.mem",
    parameter T1_MEM_FILE = "bnn_t1.mem",
    parameter T2_MEM_FILE = "bnn_t2.mem",
    parameter B3_MEM_FILE = "bnn_b3.mem"
) (
    input  wire                 I_clk,
    input  wire                 I_rst_n,
    input  wire                 I_start,
    input  wire [1023:0]        I_mask,
    output reg                  O_busy,
    output reg                  O_result_valid,
    output reg  [2:0]           O_gesture,
    output reg  [7:0]           O_confidence,
    output reg signed [31:0]    O_class_score_1,
    output reg signed [31:0]    O_class_score_2,
    output reg signed [31:0]    O_class_score_3,
    output reg signed [31:0]    O_class_score_4,
    output reg signed [31:0]    O_class_score_5
);
    localparam [2:0] ST_IDLE   = 3'd0;
    localparam [2:0] ST_LAYER1 = 3'd1;
    localparam [2:0] ST_LAYER2 = 3'd2;
    localparam [2:0] ST_LAYER3 = 3'd3;
    localparam [2:0] ST_RESULT = 3'd4;

    localparam integer INPUT_BITS = 1024;
    localparam integer OUTPUT_UNITS = 5;
    localparam integer L1_CHUNKS = INPUT_BITS / 32;
    localparam integer L2_CHUNKS = (H1_UNITS + 31) / 32;
    localparam integer L3_CHUNKS = (H2_UNITS + 31) / 32;
    localparam integer L1_DEPTH = H1_UNITS * L1_CHUNKS;
    localparam integer L2_DEPTH = H2_UNITS * L2_CHUNKS;
    localparam integer L3_DEPTH = OUTPUT_UNITS * L3_CHUNKS;
    localparam integer L1_ADDR_WIDTH = (L1_DEPTH <= 2) ? 1 : $clog2(L1_DEPTH);
    localparam integer L2_ADDR_WIDTH = (L2_DEPTH <= 2) ? 1 : $clog2(L2_DEPTH);
    localparam integer L3_ADDR_WIDTH = (L3_DEPTH <= 2) ? 1 : $clog2(L3_DEPTH);

    reg [10:0]   F_t1 [0:H1_UNITS-1];
    reg [10:0]   F_t2 [0:H2_UNITS-1];
    reg [31:0]   F_b3 [0:4];

    initial begin
        $readmemh(T1_MEM_FILE, F_t1);
        $readmemh(T2_MEM_FILE, F_t2);
        $readmemh(B3_MEM_FILE, F_b3);
    end

    reg [2:0] S_state;
    reg [H1_UNITS-1:0] S_h1_activation;
    reg [H2_UNITS-1:0] S_h2_activation;

    reg [8:0] S_l1_neuron;
    reg [9:0] S_l1_feature;
    reg [10:0] S_l1_popcount;
    wire [L1_ADDR_WIDTH-1:0] C_l1_neuron_base =
        S_l1_neuron * L1_CHUNKS;
    wire [9:0] C_l1_following_feature = S_l1_feature + 1'b1;
    wire [L1_ADDR_WIDTH-1:0] C_l1_rom_address =
        (S_state != ST_LAYER1) ? {L1_ADDR_WIDTH{1'b0}} :
        ((S_l1_feature == INPUT_BITS-1) ?
            ((S_l1_neuron == H1_UNITS-1) ? {L1_ADDR_WIDTH{1'b0}} :
                (C_l1_neuron_base + L1_CHUNKS)) :
            (C_l1_neuron_base +
                C_l1_following_feature[9:5]));
    wire [31:0] C_l1_weight_chunk;
    wire C_l1_match =
        ~(I_mask[S_l1_feature] ^ C_l1_weight_chunk[S_l1_feature[4:0]]);
    wire [10:0] C_l1_candidate = S_l1_popcount + C_l1_match;

    reg [8:0] S_l2_neuron;
    reg [8:0] S_l2_feature;
    reg [10:0] S_l2_popcount;
    wire [8:0] C_l2_following_feature = S_l2_feature + 1'b1;
    wire [L2_ADDR_WIDTH-1:0] C_l2_neuron_base =
        S_l2_neuron * L2_CHUNKS;
    wire [L2_ADDR_WIDTH-1:0] C_l2_rom_address =
        (S_state != ST_LAYER2) ? {L2_ADDR_WIDTH{1'b0}} :
        ((S_l2_feature == H1_UNITS-1) ?
            ((S_l2_neuron == H2_UNITS-1) ? {L2_ADDR_WIDTH{1'b0}} :
                (C_l2_neuron_base + L2_CHUNKS)) :
            (C_l2_neuron_base + C_l2_following_feature[8:5]));
    wire [31:0] C_l2_weight_chunk;
    wire C_l2_match =
        ~(S_h1_activation[S_l2_feature] ^
          C_l2_weight_chunk[S_l2_feature[4:0]]);
    wire [10:0] C_l2_candidate = S_l2_popcount + C_l2_match;

    reg [2:0] S_l3_neuron;
    reg [8:0] S_l3_feature;
    reg [10:0] S_l3_popcount;
    wire [8:0] C_l3_following_feature = S_l3_feature + 1'b1;
    wire [L3_ADDR_WIDTH-1:0] C_l3_neuron_base =
        S_l3_neuron * L3_CHUNKS;
    wire [L3_ADDR_WIDTH-1:0] C_l3_rom_address =
        (S_state != ST_LAYER3) ? {L3_ADDR_WIDTH{1'b0}} :
        ((S_l3_feature == H2_UNITS-1) ?
            ((S_l3_neuron == OUTPUT_UNITS-1) ? {L3_ADDR_WIDTH{1'b0}} :
                (C_l3_neuron_base + L3_CHUNKS)) :
            (C_l3_neuron_base + C_l3_following_feature[8:5]));
    wire [31:0] C_l3_weight_chunk;
    wire C_l3_match =
        ~(S_h2_activation[S_l3_feature] ^
          C_l3_weight_chunk[S_l3_feature[4:0]]);
    wire [10:0] C_l3_candidate = S_l3_popcount + C_l3_match;
    wire signed [31:0] C_l3_count_signed = {21'd0, C_l3_candidate};
    wire signed [31:0] C_l3_bias = $signed(F_b3[S_l3_neuron]);
    wire signed [31:0] C_l3_score =
        (C_l3_count_signed <<< 1) - H2_UNITS + C_l3_bias;

    reg signed [31:0] S_best_score;
    reg signed [31:0] S_second_score;
    reg [2:0] S_best_class;
    reg signed [31:0] C_next_best_score;
    reg signed [31:0] C_next_second_score;
    reg [2:0] C_next_best_class;

    always @* begin
        C_next_best_score = S_best_score;
        C_next_second_score = S_second_score;
        C_next_best_class = S_best_class;
        if(C_l3_score > S_best_score) begin
            C_next_best_score = C_l3_score;
            C_next_second_score = S_best_score;
            C_next_best_class = S_l3_neuron;
        end else if(C_l3_score > S_second_score) begin
            C_next_second_score = C_l3_score;
        end
    end

    function [7:0] F_confidence;
        input signed [31:0] I_best;
        input signed [31:0] I_second;
        reg signed [32:0] S_margin;
        begin
            S_margin = I_best - I_second;
            if(S_margin <= 0)
                F_confidence = 0;
            else if(S_margin > 255)
                F_confidence = 8'hff;
            else
                F_confidence = S_margin[7:0];
        end
    endfunction

`ifdef BNN_SIM_DEBUG
    reg [10:0] S_debug_l1_popcount [0:H1_UNITS-1];
    reg [10:0] S_debug_l2_popcount [0:H2_UNITS-1];
    reg [10:0] S_debug_l3_popcount [0:OUTPUT_UNITS-1];
`endif

    // One registered read port per layer.  Keeping the memory access in a
    // dedicated module matches Tang Dynasty's ERAM inference template and
    // prevents branch-specific copies of the same weight store.
    gesture_bnn_weight_rom #(
        .ADDRESS_WIDTH(L1_ADDR_WIDTH),
        .DEPTH(L1_DEPTH),
        .MEM_FILE(W1_MEM_FILE)
    ) U_W1_ROM (
        .I_clk(I_clk),
        .I_address(C_l1_rom_address),
        .O_data(C_l1_weight_chunk)
    );

    gesture_bnn_weight_rom #(
        .ADDRESS_WIDTH(L2_ADDR_WIDTH),
        .DEPTH(L2_DEPTH),
        .MEM_FILE(W2_MEM_FILE)
    ) U_W2_ROM (
        .I_clk(I_clk),
        .I_address(C_l2_rom_address),
        .O_data(C_l2_weight_chunk)
    );

    gesture_bnn_weight_rom #(
        .ADDRESS_WIDTH(L3_ADDR_WIDTH),
        .DEPTH(L3_DEPTH),
        .MEM_FILE(W3_MEM_FILE)
    ) U_W3_ROM (
        .I_clk(I_clk),
        .I_address(C_l3_rom_address),
        .O_data(C_l3_weight_chunk)
    );

    always @(posedge I_clk or negedge I_rst_n) begin
        if(!I_rst_n) begin
            S_state <= ST_IDLE;
            S_h1_activation <= 0;
            S_h2_activation <= 0;
            S_l1_neuron <= 0;
            S_l1_feature <= 0;
            S_l1_popcount <= 0;
            S_l2_neuron <= 0;
            S_l2_feature <= 0;
            S_l2_popcount <= 0;
            S_l3_neuron <= 0;
            S_l3_feature <= 0;
            S_l3_popcount <= 0;
            S_best_score <= -32'sd2147483647;
            S_second_score <= -32'sd2147483647;
            S_best_class <= 0;
            O_busy <= 1'b0;
            O_result_valid <= 1'b0;
            O_gesture <= 0;
            O_confidence <= 0;
            O_class_score_1 <= 0;
            O_class_score_2 <= 0;
            O_class_score_3 <= 0;
            O_class_score_4 <= 0;
            O_class_score_5 <= 0;
        end else begin
            O_result_valid <= 1'b0;
            case(S_state)
                ST_IDLE: begin
                    O_busy <= 1'b0;
                    if(I_start) begin
                        S_h1_activation <= 0;
                        S_h2_activation <= 0;
                        S_l1_neuron <= 0;
                        S_l1_feature <= 0;
                        S_l1_popcount <= 0;
                        S_best_score <= -32'sd2147483647;
                        S_second_score <= -32'sd2147483647;
                        S_best_class <= 0;
                        O_busy <= 1'b1;
                        O_confidence <= 0;
                        S_state <= ST_LAYER1;
                    end
                end

                ST_LAYER1: begin
                    if(S_l1_feature == INPUT_BITS-1) begin
                        S_h1_activation[S_l1_neuron] <=
                            (C_l1_candidate >= F_t1[S_l1_neuron]);
`ifdef BNN_SIM_DEBUG
                        S_debug_l1_popcount[S_l1_neuron] <= C_l1_candidate;
`endif
                        S_l1_feature <= 0;
                        S_l1_popcount <= 0;
                        if(S_l1_neuron == H1_UNITS-1) begin
                            S_l2_neuron <= 0;
                            S_l2_feature <= 0;
                            S_l2_popcount <= 0;
                            S_state <= ST_LAYER2;
                        end else begin
                            S_l1_neuron <= S_l1_neuron + 1'b1;
                        end
                    end else begin
                        S_l1_feature <= S_l1_feature + 1'b1;
                        S_l1_popcount <= C_l1_candidate;
                    end
                end

                ST_LAYER2: begin
                    if(S_l2_feature == H1_UNITS-1) begin
                        S_h2_activation[S_l2_neuron] <=
                            (C_l2_candidate >= F_t2[S_l2_neuron]);
`ifdef BNN_SIM_DEBUG
                        S_debug_l2_popcount[S_l2_neuron] <= C_l2_candidate;
`endif
                        S_l2_feature <= 0;
                        S_l2_popcount <= 0;
                        if(S_l2_neuron == H2_UNITS-1) begin
                            S_l3_neuron <= 0;
                            S_l3_feature <= 0;
                            S_l3_popcount <= 0;
                            S_state <= ST_LAYER3;
                        end else begin
                            S_l2_neuron <= S_l2_neuron + 1'b1;
                        end
                    end else begin
                        S_l2_feature <= S_l2_feature + 1'b1;
                        S_l2_popcount <= C_l2_candidate;
                    end
                end

                ST_LAYER3: begin
                    if(S_l3_feature == H2_UNITS-1) begin
`ifdef BNN_SIM_DEBUG
                        S_debug_l3_popcount[S_l3_neuron] <= C_l3_candidate;
`endif
                        case(S_l3_neuron)
                            3'd0: O_class_score_1 <= C_l3_score;
                            3'd1: O_class_score_2 <= C_l3_score;
                            3'd2: O_class_score_3 <= C_l3_score;
                            3'd3: O_class_score_4 <= C_l3_score;
                            3'd4: O_class_score_5 <= C_l3_score;
                            default: O_class_score_1 <= C_l3_score;
                        endcase
                        S_best_score <= C_next_best_score;
                        S_second_score <= C_next_second_score;
                        S_best_class <= C_next_best_class;
                        S_l3_feature <= 0;
                        S_l3_popcount <= 0;
                        if(S_l3_neuron == 3'd4) begin
                            S_state <= ST_RESULT;
                        end else begin
                            S_l3_neuron <= S_l3_neuron + 1'b1;
                        end
                    end else begin
                        S_l3_feature <= S_l3_feature + 1'b1;
                        S_l3_popcount <= C_l3_candidate;
                    end
                end

                ST_RESULT: begin
                    O_gesture <= S_best_class + 1'b1;
                    O_confidence <= F_confidence(S_best_score, S_second_score);
                    O_result_valid <= 1'b1;
                    O_busy <= 1'b0;
                    S_state <= ST_IDLE;
                end

                default: S_state <= ST_IDLE;
            endcase
        end
    end
endmodule

// 32-bit, single-read-port synchronous ROM used for all three BNN layers.
// The explicit registered output and Tang Dynasty "bram" attribute are the
// supported inference pattern for PH1P_LOGIC_ERAM resources.
module gesture_bnn_weight_rom #(
    parameter ADDRESS_WIDTH = 12,
    parameter DEPTH = 3072,
    parameter MEM_FILE = "bnn_weights.mem"
) (
    input  wire                       I_clk,
    input  wire [ADDRESS_WIDTH-1:0]   I_address,
    output reg  [31:0]                O_data
);
    (* rom_style = "bram" *) reg [31:0] F_memory [0:DEPTH-1];

    initial begin
        $readmemh(MEM_FILE, F_memory);
    end

    always @(posedge I_clk) begin
        O_data <= F_memory[I_address];
    end
endmodule
