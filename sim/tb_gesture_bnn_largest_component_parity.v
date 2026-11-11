`timescale 1ns/1ps

`include "bnn_roi_test_vectors.vh"

`ifndef BNN_CCL_VECTOR_LIMIT
`define BNN_CCL_VECTOR_LIMIT `BNN_ROI_VECTOR_COUNT
`endif

module tb_gesture_bnn_largest_component_parity;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg frame_start = 1'b0;
    reg sample_valid = 1'b0;
    reg row_end = 1'b0;
    reg final_row = 1'b0;
    reg [8:0] col = 0;
    reg [8:0] px = 0;
    reg [7:0] py = 0;
    reg mask_bit = 1'b0;
    wire bbox_valid;
    wire [8:0] bbox_x;
    wire [7:0] bbox_y;
    wire [8:0] bbox_width;
    wire [7:0] bbox_height;
    wire [17:0] component_area;
    wire label_overflow;

    reg [36863:0] current_raw_mask;
    reg [33:0] expected_bbox;
    reg [17:0] expected_area;
    integer vector_index;
    integer x;
    integer y;
    integer raw_file;
    integer bbox_file;
    integer area_file;
    integer scan_result;

    always #5 clk = ~clk;

    gesture_bnn_largest_component_bbox #(
        .ROI_WIDTH(256),
        .LABEL_COUNT(31),
        .MIN_COMPONENT_PIXELS(80)
    ) dut (
        .I_clk(clk),
        .I_rst_n(rst_n),
        .I_frame_start(frame_start),
        .I_sample_valid(sample_valid),
        .I_row_end(row_end),
        .I_final_row(final_row),
        .I_col(col),
        .I_x(px),
        .I_y(py),
        .I_mask(mask_bit),
        .O_bbox_valid(bbox_valid),
        .O_bbox_x(bbox_x),
        .O_bbox_y(bbox_y),
        .O_bbox_width(bbox_width),
        .O_bbox_height(bbox_height),
        .O_component_area(component_area),
        .O_label_overflow(label_overflow)
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
                    px = x;
                    py = y;
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
                // The 31-label tracker needs one cleanup clock per label.
                // Real video blanking is much longer; mirror that margin here.
                repeat(40) @(posedge clk);
            end
            repeat(40) @(posedge clk);
        end
    endtask

    initial begin
        raw_file = $fopen(`BNN_ROI_RAW_FILE, "r");
        bbox_file = $fopen(`BNN_ROI_BBOX_FILE, "r");
        area_file = $fopen(`BNN_ROI_AREA_FILE, "r");
        if((raw_file == 0) || (bbox_file == 0) || (area_file == 0))
            $fatal(1, "failed to open CCL parity vector files");
        repeat(4) @(posedge clk);
        rst_n = 1'b1;

        for(vector_index = 0; vector_index < `BNN_CCL_VECTOR_LIMIT;
            vector_index = vector_index + 1) begin
            scan_result = $fscanf(raw_file, "%h", current_raw_mask);
            if(scan_result != 1)
                $fatal(1, "failed to read raw vector %0d", vector_index);
            scan_result = $fscanf(bbox_file, "%h", expected_bbox);
            if(scan_result != 1)
                $fatal(1, "failed to read bbox vector %0d", vector_index);
            scan_result = $fscanf(area_file, "%h", expected_area);
            if(scan_result != 1)
                $fatal(1, "failed to read area vector %0d", vector_index);
            send_current_frame();
            if(label_overflow)
                $fatal(1, "vector %0d label table overflow", vector_index);
            if(!bbox_valid)
                $fatal(1, "vector %0d bbox invalid", vector_index);
            if((bbox_x !== expected_bbox[8:0]) ||
               (bbox_y !== expected_bbox[16:9]) ||
               (bbox_width !== expected_bbox[25:17]) ||
               (bbox_height !== expected_bbox[33:26]))
                $fatal(1,
                    "vector %0d bbox actual=(%0d,%0d,%0d,%0d area=%0d) expected=(%0d,%0d,%0d,%0d area=%0d)",
                    vector_index, bbox_x, bbox_y, bbox_width, bbox_height,
                    component_area,
                    expected_bbox[8:0],
                    expected_bbox[16:9],
                    expected_bbox[25:17],
                    expected_bbox[33:26],
                    expected_area);
            if(component_area !== expected_area)
                $fatal(1,
                    "vector %0d area actual=%0d expected=%0d",
                    vector_index, component_area, expected_area);
            if((vector_index + 1) % 20 == 0)
                $display("checked %0d/%0d CCL vectors", vector_index + 1,
                    `BNN_ROI_VECTOR_COUNT);
        end
        $fclose(raw_file);
        $fclose(bbox_file);
        $fclose(area_file);
        $display("PASS tb_gesture_bnn_largest_component_parity vectors=%0d",
            `BNN_ROI_VECTOR_COUNT);
        $finish;
    end
endmodule
