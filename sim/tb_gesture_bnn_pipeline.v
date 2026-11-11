`timescale 1ns/1ps

`include "bnn_roi_test_vectors.vh"

`ifndef BNN_PIPELINE_VECTOR_LIMIT
`define BNN_PIPELINE_VECTOR_LIMIT `BNN_ROI_VECTOR_COUNT
`endif

module tb_gesture_bnn_pipeline;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg frame_start = 1'b0;
    reg sample_valid = 1'b0;
    reg row_end = 1'b0;
    reg final_row = 1'b0;
    reg [7:0] col = 0;
    reg [7:0] row = 0;
    reg mask_bit = 1'b0;

    wire label_overflow;
    wire bbox_valid;
    wire bbox_frame_done;
    wire [7:0] bbox_x;
    wire [7:0] bbox_y;
    wire [8:0] bbox_width;
    wire [7:0] bbox_height;
    wire [17:0] component_area;
    wire normalizer_busy;
    wire normalized_valid;
    wire [1023:0] normalized_mask;
    wire bnn_busy;
    wire result_valid;
    wire [2:0] gesture;
    wire [7:0] confidence;
    wire signed [31:0] class_score_1;
    wire signed [31:0] class_score_2;
    wire signed [31:0] class_score_3;
    wire signed [31:0] class_score_4;
    wire signed [31:0] class_score_5;

    reg [36863:0] current_raw_mask;
    reg [1023:0] expected_normalized;
    reg [33:0] expected_bbox;
    reg [17:0] expected_area;
    reg [2:0] expected_gesture;
    reg [7:0] expected_confidence;
    reg [159:0] expected_scores;
    integer raw_file;
    integer normalized_file;
    integer bbox_file;
    integer area_file;
    integer gesture_file;
    integer confidence_file;
    integer scores_file;
    integer scan_result;
    integer vector_index;
    integer x;
    integer y;
    integer cycles;
    integer normalizer_latency;
    integer bnn_latency;
    integer minimum_normalizer_wait;
    integer maximum_normalizer_wait;
    integer reference_bnn_latency;

    always #5 clk = ~clk;

    gesture_bnn_pipeline #(
        .W1_MEM_FILE(`BNN_ROI_W1_FILE),
        .W2_MEM_FILE(`BNN_ROI_W2_FILE),
        .W3_MEM_FILE(`BNN_ROI_W3_FILE),
        .T1_MEM_FILE(`BNN_ROI_T1_FILE),
        .T2_MEM_FILE(`BNN_ROI_T2_FILE),
        .B3_MEM_FILE(`BNN_ROI_B3_FILE)
    ) dut (
        .I_clk(clk),
        .I_rst_n(rst_n),
        .I_frame_start(frame_start),
        .I_sample_valid(sample_valid),
        .I_row_end(row_end),
        .I_final_row(final_row),
        .I_col(col),
        .I_row(row),
        .I_mask(mask_bit),
        .O_label_overflow(label_overflow),
        .O_bbox_valid(bbox_valid),
        .O_bbox_frame_done(bbox_frame_done),
        .O_bbox_x(bbox_x),
        .O_bbox_y(bbox_y),
        .O_bbox_width(bbox_width),
        .O_bbox_height(bbox_height),
        .O_component_area(component_area),
        .O_normalizer_busy(normalizer_busy),
        .O_normalized_valid(normalized_valid),
        .O_normalized_mask(normalized_mask),
        .O_bnn_busy(bnn_busy),
        .O_result_valid(result_valid),
        .O_gesture(gesture),
        .O_confidence(confidence),
        .O_class_score_1(class_score_1),
        .O_class_score_2(class_score_2),
        .O_class_score_3(class_score_3),
        .O_class_score_4(class_score_4),
        .O_class_score_5(class_score_5)
    );

    task start_new_frame;
        begin
            @(negedge clk);
            frame_start = 1'b1;
            @(posedge clk);
            @(negedge clk);
            frame_start = 1'b0;
            repeat(4) @(posedge clk);
        end
    endtask

    task send_current_frame;
        begin
            start_new_frame();
            for(y = 0; y < 144; y = y + 1) begin
                for(x = 0; x < 256; x = x + 1) begin
                    @(negedge clk);
                    sample_valid = 1'b1;
                    col = x;
                    row = y;
                    mask_bit = current_raw_mask[y*256+x];
                    row_end = (x == 255);
                    final_row = row_end && (y == 143);
                    @(posedge clk);
                    @(negedge clk);
                    sample_valid = 1'b0;
                    row_end = 1'b0;
                    final_row = 1'b0;
                    mask_bit = 1'b0;
                    repeat(3) @(posedge clk);
                end
                // 31 CCL cleanup clocks plus timing margin.
                repeat(40) @(posedge clk);
            end
        end
    endtask

    task read_current_vector;
        begin
            scan_result = $fscanf(raw_file, "%h", current_raw_mask);
            if(scan_result != 1) $fatal(1, "failed raw vector %0d", vector_index);
            scan_result = $fscanf(normalized_file, "%h", expected_normalized);
            if(scan_result != 1) $fatal(1, "failed normalized vector %0d", vector_index);
            scan_result = $fscanf(bbox_file, "%h", expected_bbox);
            if(scan_result != 1) $fatal(1, "failed bbox vector %0d", vector_index);
            scan_result = $fscanf(area_file, "%h", expected_area);
            if(scan_result != 1) $fatal(1, "failed area vector %0d", vector_index);
            scan_result = $fscanf(gesture_file, "%h", expected_gesture);
            if(scan_result != 1) $fatal(1, "failed gesture vector %0d", vector_index);
            scan_result = $fscanf(confidence_file, "%h", expected_confidence);
            if(scan_result != 1) $fatal(1, "failed confidence vector %0d", vector_index);
            scan_result = $fscanf(scores_file, "%h", expected_scores);
            if(scan_result != 1) $fatal(1, "failed scores vector %0d", vector_index);
        end
    endtask

    task check_bbox;
        begin
            if(label_overflow) $fatal(1, "vector %0d label overflow", vector_index);
            if(!bbox_valid) $fatal(1, "vector %0d bbox invalid", vector_index);
            if((bbox_x !== expected_bbox[8:0]) ||
               (bbox_y !== expected_bbox[16:9]) ||
               (bbox_width !== expected_bbox[25:17]) ||
               (bbox_height !== expected_bbox[33:26]) ||
               (component_area !== expected_area))
                $fatal(1, "vector %0d CCL mismatch", vector_index);
        end
    endtask

    task check_bnn_result;
        begin
            if(gesture !== expected_gesture)
                $fatal(1, "vector %0d gesture expected=%0d actual=%0d",
                    vector_index, expected_gesture, gesture);
            if(confidence !== expected_confidence)
                $fatal(1, "vector %0d confidence expected=%0d actual=%0d",
                    vector_index, expected_confidence, confidence);
            if(class_score_1 !== $signed(expected_scores[31:0]) ||
               class_score_2 !== $signed(expected_scores[63:32]) ||
               class_score_3 !== $signed(expected_scores[95:64]) ||
               class_score_4 !== $signed(expected_scores[127:96]) ||
               class_score_5 !== $signed(expected_scores[159:128]))
                $fatal(1, "vector %0d class-score mismatch", vector_index);
        end
    endtask

    initial begin
        raw_file = $fopen(`BNN_ROI_RAW_FILE, "r");
        normalized_file = $fopen(`BNN_ROI_EXPECTED_FILE, "r");
        bbox_file = $fopen(`BNN_ROI_BBOX_FILE, "r");
        area_file = $fopen(`BNN_ROI_AREA_FILE, "r");
        gesture_file = $fopen(`BNN_ROI_GESTURE_FILE, "r");
        confidence_file = $fopen(`BNN_ROI_CONFIDENCE_FILE, "r");
        scores_file = $fopen(`BNN_ROI_SCORES_FILE, "r");
        if((raw_file == 0) || (normalized_file == 0) ||
           (bbox_file == 0) || (area_file == 0) ||
           (gesture_file == 0) || (confidence_file == 0) ||
           (scores_file == 0))
            $fatal(1, "failed to open pipeline parity vector files");

        minimum_normalizer_wait = 1100;
        maximum_normalizer_wait = 0;
        reference_bnn_latency = -1;
        repeat(4) @(posedge clk);
        rst_n = 1'b1;

        for(vector_index = 0; vector_index < `BNN_PIPELINE_VECTOR_LIMIT;
            vector_index = vector_index + 1) begin
            read_current_vector();
            send_current_frame();
            check_bbox();

            cycles = 0;
            while((normalized_valid !== 1'b1) && (cycles < 1100)) begin
                @(negedge clk);
                cycles = cycles + 1;
            end
            if(normalized_valid !== 1'b1)
                $fatal(1, "vector %0d normalizer timeout", vector_index);
            normalizer_latency = cycles;
            if(normalized_mask !== expected_normalized)
                $fatal(1, "vector %0d normalized-mask mismatch", vector_index);
            if(normalizer_latency < minimum_normalizer_wait)
                minimum_normalizer_wait = normalizer_latency;
            if(normalizer_latency > maximum_normalizer_wait)
                maximum_normalizer_wait = normalizer_latency;

            cycles = 0;
            while((result_valid !== 1'b1) && (cycles < 103300)) begin
                @(negedge clk);
                cycles = cycles + 1;
            end
            if(result_valid !== 1'b1)
                $fatal(1, "vector %0d BNN timeout", vector_index);
            bnn_latency = cycles;
            if(reference_bnn_latency < 0)
                reference_bnn_latency = bnn_latency;
            else if(bnn_latency != reference_bnn_latency)
                $fatal(1, "vector %0d BNN latency changed", vector_index);
            check_bnn_result();
            if((vector_index + 1) % 10 == 0)
                $display("checked %0d/%0d pipeline vectors", vector_index + 1,
                    `BNN_PIPELINE_VECTOR_LIMIT);
            @(negedge clk);
        end

        $fclose(raw_file);
        $fclose(normalized_file);
        $fclose(bbox_file);
        $fclose(area_file);
        $fclose(gesture_file);
        $fclose(confidence_file);
        $fclose(scores_file);
        $display("PASS tb_gesture_bnn_pipeline vectors=%0d normalize_wait=%0d..%0d bnn_wait=%0d",
            `BNN_PIPELINE_VECTOR_LIMIT, minimum_normalizer_wait,
            maximum_normalizer_wait, reference_bnn_latency);
        $finish;
    end
endmodule
