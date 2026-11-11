`timescale 1ns/1ps

`include "bnn_roi_test_vectors.vh"

module tb_gesture_bnn_roi_normalizer;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg frame_start = 1'b0;
    reg sample_valid = 1'b0;
    reg [7:0] col = 0;
    reg [7:0] row = 0;
    reg mask_bit = 1'b0;
    reg bbox_valid = 1'b0;
    reg [7:0] bbox_x = 0;
    reg [7:0] bbox_y = 0;
    reg [8:0] bbox_width = 0;
    reg [7:0] bbox_height = 0;
    wire busy;
    wire result_valid;
    wire [1023:0] normalized_mask;

    reg [36863:0] raw_masks [0:`BNN_ROI_VECTOR_COUNT-1];
    reg [1023:0] expected_masks [0:`BNN_ROI_VECTOR_COUNT-1];
    reg [33:0] bboxes [0:`BNN_ROI_VECTOR_COUNT-1];

    integer vector_index;
    integer pixel_index;
    integer output_index;
    integer cycles;
    integer reference_cycles;
    integer mismatch_count;
    integer first_mismatch;

    always #5 clk = ~clk;

    gesture_bnn_roi_normalizer dut (
        .I_clk(clk),
        .I_rst_n(rst_n),
        .I_frame_start(frame_start),
        .I_sample_valid(sample_valid),
        .I_col(col),
        .I_row(row),
        .I_mask(mask_bit),
        .I_bbox_valid(bbox_valid),
        .I_bbox_x(bbox_x),
        .I_bbox_y(bbox_y),
        .I_bbox_width(bbox_width),
        .I_bbox_height(bbox_height),
        .O_busy(busy),
        .O_result_valid(result_valid),
        .O_normalized_mask(normalized_mask)
    );

    initial begin
        $readmemh(`BNN_ROI_RAW_FILE, raw_masks);
        $readmemh(`BNN_ROI_EXPECTED_FILE, expected_masks);
        $readmemh(`BNN_ROI_BBOX_FILE, bboxes);

        reference_cycles = -1;
        repeat(4) @(posedge clk);
        rst_n <= 1'b1;

        for(vector_index = 0; vector_index < `BNN_ROI_VECTOR_COUNT;
            vector_index = vector_index + 1) begin
            @(negedge clk);
            frame_start <= 1'b1;
            @(negedge clk);
            frame_start <= 1'b0;

            for(pixel_index = 0; pixel_index < 36864;
                pixel_index = pixel_index + 1) begin
                @(negedge clk);
                sample_valid <= 1'b1;
                col <= pixel_index[7:0];
                row <= pixel_index >> 8;
                mask_bit <= raw_masks[vector_index][pixel_index];
            end
            @(negedge clk);
            sample_valid <= 1'b0;
            mask_bit <= 1'b0;
            if(bboxes[vector_index][8])
                $fatal(1, "vector %0d bbox x does not fit 8 bits", vector_index);
            bbox_x <= bboxes[vector_index][7:0];
            bbox_y <= bboxes[vector_index][16:9];
            bbox_width <= bboxes[vector_index][25:17];
            bbox_height <= bboxes[vector_index][33:26];
            bbox_valid <= 1'b1;
            @(negedge clk);
            bbox_valid <= 1'b0;

            cycles = 0;
            while(result_valid !== 1'b1 && cycles < 1100) begin
                @(negedge clk);
                cycles = cycles + 1;
            end
            if(result_valid !== 1'b1)
                $fatal(1, "vector %0d normalizer timeout", vector_index);
            if(reference_cycles < 0)
                reference_cycles = cycles;
            else if(cycles != reference_cycles)
                $fatal(1,
                    "vector %0d latency changed expected=%0d actual=%0d",
                    vector_index, reference_cycles, cycles);

            if(normalized_mask !== expected_masks[vector_index]) begin
                mismatch_count = 0;
                first_mismatch = -1;
                for(output_index = 0; output_index < 1024;
                    output_index = output_index + 1) begin
                    if(normalized_mask[output_index] !==
                       expected_masks[vector_index][output_index]) begin
                        mismatch_count = mismatch_count + 1;
                        if(first_mismatch < 0)
                            first_mismatch = output_index;
                    end
                end
                $fatal(1,
                    "vector %0d normalized mask mismatch pixels=%0d first=%0d",
                    vector_index, mismatch_count, first_mismatch);
            end
            if((vector_index + 1) % 20 == 0)
                $display("checked %0d/%0d ROI vectors", vector_index + 1,
                    `BNN_ROI_VECTOR_COUNT);
            @(negedge clk);
        end

        $display(
            "PASS tb_gesture_bnn_roi_normalizer vectors=%0d latency_clocks=%0d",
            `BNN_ROI_VECTOR_COUNT, reference_cycles);
        $finish;
    end
endmodule
