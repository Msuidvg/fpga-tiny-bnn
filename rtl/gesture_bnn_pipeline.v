`timescale 1ns/1ps

// Bit-exact S5-mask to Tiny-BNN pipeline used before integration with the
// video top level.  Coordinates are relative to the 256x144 sampled ROI.
//
//   majority mask -> largest 8-connected component -> Python-equivalent
//   32x32 ROI normalization -> serial XNOR/popcount Tiny-BNN
module gesture_bnn_pipeline #(
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
    input  wire                 I_frame_start,
    input  wire                 I_sample_valid,
    input  wire                 I_row_end,
    input  wire                 I_final_row,
    input  wire [7:0]           I_col,
    input  wire [7:0]           I_row,
    input  wire                 I_mask,

    output wire                 O_label_overflow,
    output wire                 O_bbox_valid,
    output wire                 O_bbox_frame_done,
    output wire [7:0]           O_bbox_x,
    output wire [7:0]           O_bbox_y,
    output wire [8:0]           O_bbox_width,
    output wire [7:0]           O_bbox_height,
    output wire [17:0]          O_component_area,
    output wire                 O_normalizer_busy,
    output wire                 O_normalized_valid,
    output wire [1023:0]        O_normalized_mask,
    output wire                 O_bnn_busy,
    output wire                 O_result_valid,
    output wire [2:0]           O_gesture,
    output wire [7:0]           O_confidence,
    output wire signed [31:0]   O_class_score_1,
    output wire signed [31:0]   O_class_score_2,
    output wire signed [31:0]   O_class_score_3,
    output wire signed [31:0]   O_class_score_4,
    output wire signed [31:0]   O_class_score_5
);
    wire [8:0] C_bbox_x;

    gesture_bnn_largest_component_bbox #(
        .ROI_WIDTH(256),
        .LABEL_COUNT(31),
        .MIN_COMPONENT_PIXELS(80)
    ) U_CCL (
        .I_clk(I_clk),
        .I_rst_n(I_rst_n),
        .I_frame_start(I_frame_start),
        .I_sample_valid(I_sample_valid),
        .I_row_end(I_row_end),
        .I_final_row(I_final_row),
        .I_col({1'b0, I_col}),
        .I_x({1'b0, I_col}),
        .I_y(I_row),
        .I_mask(I_mask),
        .O_bbox_valid(O_bbox_valid),
        .O_bbox_x(C_bbox_x),
        .O_bbox_y(O_bbox_y),
        .O_bbox_width(O_bbox_width),
        .O_bbox_height(O_bbox_height),
        .O_component_area(O_component_area),
        .O_label_overflow(O_label_overflow),
        .O_frame_done(O_bbox_frame_done)
    );

    assign O_bbox_x = C_bbox_x[7:0];

    gesture_bnn_roi_normalizer U_NORMALIZER (
        .I_clk(I_clk),
        .I_rst_n(I_rst_n),
        .I_frame_start(I_frame_start),
        .I_sample_valid(I_sample_valid),
        .I_col(I_col),
        .I_row(I_row),
        .I_mask(I_mask),
        .I_bbox_valid(O_bbox_frame_done && O_bbox_valid),
        .I_bbox_x(O_bbox_x),
        .I_bbox_y(O_bbox_y),
        .I_bbox_width(O_bbox_width),
        .I_bbox_height(O_bbox_height),
        .O_busy(O_normalizer_busy),
        .O_result_valid(O_normalized_valid),
        .O_normalized_mask(O_normalized_mask)
    );

    gesture_bnn_core #(
        .H1_UNITS(H1_UNITS),
        .H2_UNITS(H2_UNITS),
        .W1_MEM_FILE(W1_MEM_FILE),
        .W2_MEM_FILE(W2_MEM_FILE),
        .W3_MEM_FILE(W3_MEM_FILE),
        .T1_MEM_FILE(T1_MEM_FILE),
        .T2_MEM_FILE(T2_MEM_FILE),
        .B3_MEM_FILE(B3_MEM_FILE)
    ) U_BNN (
        .I_clk(I_clk),
        .I_rst_n(I_rst_n),
        .I_start(O_normalized_valid),
        .I_mask(O_normalized_mask),
        .O_busy(O_bnn_busy),
        .O_result_valid(O_result_valid),
        .O_gesture(O_gesture),
        .O_confidence(O_confidence),
        .O_class_score_1(O_class_score_1),
        .O_class_score_2(O_class_score_2),
        .O_class_score_3(O_class_score_3),
        .O_class_score_4(O_class_score_4),
        .O_class_score_5(O_class_score_5)
    );
endmodule
