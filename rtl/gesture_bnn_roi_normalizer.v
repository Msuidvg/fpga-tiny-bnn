`timescale 1ns/1ps

// Buffer one 256x144 S5 majority mask and reproduce the Python BNN crop:
//   bbox -> independent 10% x/y margin (banker's rounding) -> clip to frame
//   -> center crop on a square zero canvas -> 32x32 nearest-neighbour resize.
//
// Bbox coordinates are relative to the 256x144 sampled ROI, not global HDMI
// coordinates.  I_bbox_valid must arrive after the final mask sample is stored.
module gesture_bnn_roi_normalizer (
    input  wire          I_clk,
    input  wire          I_rst_n,
    input  wire          I_frame_start,
    input  wire          I_sample_valid,
    input  wire [7:0]    I_col,
    input  wire [7:0]    I_row,
    input  wire          I_mask,
    input  wire          I_bbox_valid,
    input  wire [7:0]    I_bbox_x,
    input  wire [7:0]    I_bbox_y,
    input  wire [8:0]    I_bbox_width,
    input  wire [7:0]    I_bbox_height,
    output reg           O_busy,
    output reg           O_result_valid,
    output reg [1023:0]  O_normalized_mask
);
    localparam [1:0] ST_IDLE      = 2'd0;
    localparam [1:0] ST_NORMALIZE = 2'd1;
    localparam [1:0] ST_DRAIN     = 2'd2;
    localparam [1:0] ST_RESULT    = 2'd3;

    wire [15:0] C_write_address = {I_row, I_col};

    function [8:0] F_round_tenth_even;
        input [8:0] I_value;
        reg [8:0] quotient;
        reg [3:0] remainder;
        begin
            quotient = I_value / 10;
            remainder = I_value % 10;
            if((remainder > 5) || ((remainder == 5) && quotient[0]))
                F_round_tenth_even = quotient + 1'b1;
            else
                F_round_tenth_even = quotient;
        end
    endfunction

    wire [8:0] C_bbox_x = {1'b0, I_bbox_x};
    wire [8:0] C_bbox_y = {1'b0, I_bbox_y};
    wire [8:0] C_bbox_height = {1'b0, I_bbox_height};
    wire [8:0] C_pad_x = F_round_tenth_even(I_bbox_width);
    wire [8:0] C_pad_y = F_round_tenth_even(C_bbox_height);
    wire [8:0] C_x0 = (C_bbox_x >= C_pad_x) ? C_bbox_x - C_pad_x : 0;
    wire [8:0] C_y0 = (C_bbox_y >= C_pad_y) ? C_bbox_y - C_pad_y : 0;
    wire [9:0] C_x1_unclipped =
        {1'b0, C_bbox_x} + {1'b0, I_bbox_width} + {1'b0, C_pad_x};
    wire [9:0] C_y1_unclipped =
        {1'b0, C_bbox_y} + {1'b0, C_bbox_height} + {1'b0, C_pad_y};
    wire [8:0] C_x1 = (C_x1_unclipped > 10'd256) ?
                       9'd256 : C_x1_unclipped[8:0];
    wire [8:0] C_y1 = (C_y1_unclipped > 10'd144) ?
                       9'd144 : C_y1_unclipped[8:0];
    wire [8:0] C_crop_width = C_x1 - C_x0;
    wire [8:0] C_crop_height = C_y1 - C_y0;
    wire [8:0] C_side =
        (C_crop_width >= C_crop_height) ? C_crop_width : C_crop_height;
    wire [8:0] C_dst_x = (C_side - C_crop_width) >> 1;
    wire [8:0] C_dst_y = (C_side - C_crop_height) >> 1;

    reg [1:0] S_state;
    reg [8:0] S_x0;
    reg [8:0] S_y0;
    reg [8:0] S_crop_width;
    reg [8:0] S_crop_height;
    reg [8:0] S_side;
    reg [8:0] S_dst_x;
    reg [8:0] S_dst_y;
    reg [9:0] S_output_index;
    reg [9:0] S_read_output_index;
    reg S_read_inside_crop;
    reg S_read_pending;

    wire [4:0] C_output_x = S_output_index[4:0];
    wire [4:0] C_output_y = S_output_index[9:5];
    wire [13:0] C_square_x_product = C_output_x * S_side;
    wire [13:0] C_square_y_product = C_output_y * S_side;
    wire [8:0] C_square_x = C_square_x_product >> 5;
    wire [8:0] C_square_y = C_square_y_product >> 5;
    wire C_inside_crop =
        (C_square_x >= S_dst_x) &&
        (C_square_x < S_dst_x + S_crop_width) &&
        (C_square_y >= S_dst_y) &&
        (C_square_y < S_dst_y + S_crop_height);
    wire [8:0] C_source_x = S_x0 + C_square_x - S_dst_x;
    wire [8:0] C_source_y = S_y0 + C_square_y - S_dst_y;
    wire [15:0] C_read_address = {C_source_y[7:0], C_source_x[7:0]};
    wire [15:0] C_safe_read_address =
        C_inside_crop ? C_read_address : 16'd0;
    wire C_mask_mem_data;

    // A true dual-port synchronous RAM follows Tang Dynasty's supported ERAM
    // inference template. Capture writes and ROI reads never overlap in the
    // pipeline, but separate ports keep the streaming interface simple.
    gesture_bnn_mask_frame_ram U_MASK_FRAME_RAM (
        .I_clk(I_clk),
        .I_write_enable(I_sample_valid && (I_row < 8'd144)),
        .I_write_address(C_write_address),
        .I_write_data(I_mask),
        .I_read_enable(S_state == ST_NORMALIZE),
        .I_read_address(C_safe_read_address),
        .O_read_data(C_mask_mem_data)
    );

    always @(posedge I_clk or negedge I_rst_n) begin
        if(!I_rst_n) begin
            S_state <= ST_IDLE;
            S_x0 <= 0;
            S_y0 <= 0;
            S_crop_width <= 0;
            S_crop_height <= 0;
            S_side <= 0;
            S_dst_x <= 0;
            S_dst_y <= 0;
            S_output_index <= 0;
            S_read_output_index <= 0;
            S_read_inside_crop <= 1'b0;
            S_read_pending <= 1'b0;
            O_busy <= 1'b0;
            O_result_valid <= 1'b0;
            O_normalized_mask <= 0;
        end else begin
            O_result_valid <= 1'b0;

            case(S_state)
                ST_IDLE: begin
                    O_busy <= 1'b0;
                    if(I_bbox_valid && (I_bbox_width != 0) &&
                       (I_bbox_height != 0)) begin
                        S_x0 <= C_x0;
                        S_y0 <= C_y0;
                        S_crop_width <= C_crop_width;
                        S_crop_height <= C_crop_height;
                        S_side <= C_side;
                        S_dst_x <= C_dst_x;
                        S_dst_y <= C_dst_y;
                        S_output_index <= 0;
                        S_read_output_index <= 0;
                        S_read_inside_crop <= 1'b0;
                        S_read_pending <= 1'b0;
                        O_normalized_mask <= 0;
                        O_busy <= 1'b1;
                        S_state <= ST_NORMALIZE;
                    end
                end

                ST_NORMALIZE: begin
                    if(S_read_pending)
                        O_normalized_mask[S_read_output_index] <=
                            S_read_inside_crop ? C_mask_mem_data : 1'b0;
                    S_read_output_index <= S_output_index;
                    S_read_inside_crop <= C_inside_crop;
                    S_read_pending <= 1'b1;
                    if(S_output_index == 10'd1023) begin
                        S_state <= ST_DRAIN;
                    end else begin
                        S_output_index <= S_output_index + 1'b1;
                    end
                end

                ST_DRAIN: begin
                    O_normalized_mask[S_read_output_index] <=
                        S_read_inside_crop ? C_mask_mem_data : 1'b0;
                    S_read_pending <= 1'b0;
                    S_state <= ST_RESULT;
                end

                ST_RESULT: begin
                    O_busy <= 1'b0;
                    O_result_valid <= 1'b1;
                    S_state <= ST_IDLE;
                end

                default: S_state <= ST_IDLE;
            endcase
        end
    end

    // I_frame_start intentionally does not clear the full mask RAM: a complete
    // 256x144 frame overwrites every location before I_bbox_valid is asserted.
    wire C_unused_frame_start = I_frame_start;
endmodule

module gesture_bnn_mask_frame_ram (
    input  wire        I_clk,
    input  wire        I_write_enable,
    input  wire [15:0] I_write_address,
    input  wire        I_write_data,
    input  wire        I_read_enable,
    input  wire [15:0] I_read_address,
    output reg         O_read_data
);
    (* ram_style = "bram" *) reg F_memory [0:36863];

    always @(posedge I_clk) begin
        if(I_write_enable)
            F_memory[I_write_address] <= I_write_data;
    end

    always @(posedge I_clk) begin
        if(I_read_enable)
            O_read_data <= F_memory[I_read_address];
    end
endmodule
