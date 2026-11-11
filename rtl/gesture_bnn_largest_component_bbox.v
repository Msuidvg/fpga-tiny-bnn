`timescale 1ns/1ps

// BNN-side 8-connected component tracker for the S5 majority mask.
//
// A one-row label memory supplies the four causal neighbours: left,
// up-left, up and up-right.  Component merges are spread over the three idle
// source clocks between adjacent DS_SHIFT=2 samples.  Labels that can no
// longer touch the next sampled row are finalized and recycled, so a small
// label table is sufficient without a full-frame label RAM.
//
// The result registers always hold the largest completed component from the
// most recently processed frame.  Coordinates remain in sampled-image space.
module gesture_bnn_largest_component_bbox #(
    parameter integer ROI_WIDTH = 256,
    parameter integer LABEL_COUNT = 31,
    parameter integer MIN_COMPONENT_PIXELS = 80
)(
    input  wire        I_clk,
    input  wire        I_rst_n,
    input  wire        I_frame_start,
    input  wire        I_sample_valid,
    input  wire        I_row_end,
    input  wire        I_final_row,
    input  wire [8:0]  I_col,
    input  wire [8:0]  I_x,
    input  wire [7:0]  I_y,
    input  wire        I_mask,

    output reg         O_bbox_valid,
    output reg  [8:0]  O_bbox_x,
    output reg  [7:0]  O_bbox_y,
    output reg  [8:0]  O_bbox_width,
    output reg  [7:0]  O_bbox_height,
    output reg  [17:0] O_component_area,
    output reg         O_label_overflow,
    output reg         O_frame_done
);
    // Majority masks contain more small islands than the opened-mask path.
    // The independent BNN tracker therefore uses 31 recyclable 5-bit labels.
    reg [4:0] S_label_line [0:ROI_WIDTH-1]; // synthesis ram_style=dram
    reg [4:0] S_parent [0:LABEL_COUNT];
    reg [17:0] S_area [0:LABEL_COUNT];
    reg [8:0] S_min_x [0:LABEL_COUNT];
    reg [8:0] S_max_x [0:LABEL_COUNT];
    reg [7:0] S_min_y [0:LABEL_COUNT];
    reg [7:0] S_max_y [0:LABEL_COUNT];
    reg [LABEL_COUNT:0] S_row_root_active;
    reg [LABEL_COUNT:0] S_row_label_present;

    reg [4:0] S_left_label;
    reg [4:0] S_up_left_label;
    reg [17:0] S_best_area;
    reg S_first_row;

    reg S_busy;
    reg [1:0] S_merge_phase;
    reg [4:0] S_pending_base;
    // Keep the four resolved neighbour roots and select one distinct merge
    // target per idle phase.  This avoids compacting all merge candidates in
    // the sample clock, which previously synthesized to a 20-level path.
    reg [4:0] S_pending_root_left;
    reg [4:0] S_pending_root_up_left;
    reg [4:0] S_pending_root_up;
    reg [4:0] S_pending_root_up_right;
    reg [8:0] S_pending_x;
    reg [7:0] S_pending_y;
    reg [4:0] S_merged_label_0;
    reg [4:0] S_merged_label_1;
    reg S_pending_row_end;
    reg S_pending_final_row;
    reg [17:0] S_work_area;
    reg [8:0] S_work_min_x;
    reg [8:0] S_work_max_x;
    reg [7:0] S_work_min_y;
    reg [7:0] S_work_max_y;

    reg S_cleanup_active;
    reg S_cleanup_final_row;
    reg [5:0] S_cleanup_index;

    integer R_i;
    integer C_i;
    reg [4:0] C_root_left;
    reg [4:0] C_root_up_left;
    reg [4:0] C_root_up;
    reg [4:0] C_root_up_right;
    reg [4:0] C_base_label;
    reg [4:0] C_free_label;
    reg [31:1] C_label_free;
    reg [4:0] C_free_group_0;
    reg [4:0] C_free_group_1;
    reg [4:0] C_free_group_2;
    reg [4:0] C_free_group_3;
    reg [4:0] C_free_group_4;
    reg [4:0] C_free_group_5;
    reg [4:0] C_free_group_6;
    reg [4:0] C_free_group_7;
    reg [4:0] C_assigned_label;
    reg [4:0] C_phase_merge_label;
    reg [17:0] C_work_area_next;
    reg [8:0] C_work_min_x_next;
    reg [8:0] C_work_max_x_next;
    reg [7:0] C_work_min_y_next;
    reg [7:0] C_work_max_y_next;

    function [4:0] F_root;
        input [4:0] I_label;
        reg [4:0] p1;
        begin
            if(I_label == 0) begin
                F_root = 0;
            end else begin
                p1 = S_parent[I_label];
                F_root = (p1 == 0) ? I_label : p1;
            end
        end
    endfunction

    // Resolve neighbours and allocate a label only on a foreground sample.
    always @* begin
        C_root_left = F_root(S_left_label);
        C_root_up_left = S_first_row ? 5'd0 : F_root(S_up_left_label);
        C_root_up = S_first_row ? 5'd0 :
            F_root(S_label_line[I_col]);
        C_root_up_right = (!S_first_row && (I_col < ROI_WIDTH-1)) ?
            F_root(S_label_line[I_col + 1'b1]) : 5'd0;

        C_base_label = 0;
        if(C_root_left != 0)
            C_base_label = C_root_left;
        if((C_root_up_left != 0) &&
           ((C_base_label == 0) || (C_root_up_left < C_base_label)))
            C_base_label = C_root_up_left;
        if((C_root_up != 0) &&
           ((C_base_label == 0) || (C_root_up < C_base_label)))
            C_base_label = C_root_up;
        if((C_root_up_right != 0) &&
           ((C_base_label == 0) || (C_root_up_right < C_base_label)))
            C_base_label = C_root_up_right;

        C_label_free = 0;
        for(C_i = 1; C_i <= LABEL_COUNT; C_i = C_i + 1) begin
            C_label_free[C_i] = (S_area[C_i] == 0) &&
                                (S_parent[C_i] == 0) &&
                                !S_row_label_present[C_i];
        end
        // Eight four-entry groups preserve the lowest-free-label policy
        // without a single 31-deep priority chain.
        C_free_group_0 = C_label_free[1] ? 5'd1 :
                         C_label_free[2] ? 5'd2 :
                         C_label_free[3] ? 5'd3 :
                         C_label_free[4] ? 5'd4 : 5'd0;
        C_free_group_1 = C_label_free[5] ? 5'd5 :
                         C_label_free[6] ? 5'd6 :
                         C_label_free[7] ? 5'd7 :
                         C_label_free[8] ? 5'd8 : 5'd0;
        C_free_group_2 = C_label_free[9]  ? 5'd9  :
                         C_label_free[10] ? 5'd10 :
                         C_label_free[11] ? 5'd11 :
                         C_label_free[12] ? 5'd12 : 5'd0;
        C_free_group_3 = C_label_free[13] ? 5'd13 :
                         C_label_free[14] ? 5'd14 :
                         C_label_free[15] ? 5'd15 :
                         C_label_free[16] ? 5'd16 : 5'd0;
        C_free_group_4 = C_label_free[17] ? 5'd17 :
                         C_label_free[18] ? 5'd18 :
                         C_label_free[19] ? 5'd19 :
                         C_label_free[20] ? 5'd20 : 5'd0;
        C_free_group_5 = C_label_free[21] ? 5'd21 :
                         C_label_free[22] ? 5'd22 :
                         C_label_free[23] ? 5'd23 :
                         C_label_free[24] ? 5'd24 : 5'd0;
        C_free_group_6 = C_label_free[25] ? 5'd25 :
                         C_label_free[26] ? 5'd26 :
                         C_label_free[27] ? 5'd27 :
                         C_label_free[28] ? 5'd28 : 5'd0;
        C_free_group_7 = C_label_free[29] ? 5'd29 :
                         C_label_free[30] ? 5'd30 :
                         C_label_free[31] ? 5'd31 : 5'd0;
        C_free_label = (C_free_group_0 != 0) ? C_free_group_0 :
                       (C_free_group_1 != 0) ? C_free_group_1 :
                       (C_free_group_2 != 0) ? C_free_group_2 :
                       (C_free_group_3 != 0) ? C_free_group_3 :
                       (C_free_group_4 != 0) ? C_free_group_4 :
                       (C_free_group_5 != 0) ? C_free_group_5 :
                       (C_free_group_6 != 0) ? C_free_group_6 :
                                              C_free_group_7;
        C_assigned_label = (C_base_label != 0) ?
                           C_base_label : C_free_label;

    end

    always @* begin
        // At most three roots can differ from the selected base.  Select the
        // next distinct one from registered roots instead of building a
        // priority/duplicate-removal cascade in the input sample cycle.
        C_phase_merge_label = 0;
        if((S_pending_root_left != 0) &&
           (S_pending_root_left != S_pending_base) &&
           (S_pending_root_left != S_merged_label_0) &&
           (S_pending_root_left != S_merged_label_1))
            C_phase_merge_label = S_pending_root_left;
        else if((S_pending_root_up_left != 0) &&
                (S_pending_root_up_left != S_pending_base) &&
                (S_pending_root_up_left != S_merged_label_0) &&
                (S_pending_root_up_left != S_merged_label_1))
            C_phase_merge_label = S_pending_root_up_left;
        else if((S_pending_root_up != 0) &&
                (S_pending_root_up != S_pending_base) &&
                (S_pending_root_up != S_merged_label_0) &&
                (S_pending_root_up != S_merged_label_1))
            C_phase_merge_label = S_pending_root_up;
        else if((S_pending_root_up_right != 0) &&
                (S_pending_root_up_right != S_pending_base) &&
                (S_pending_root_up_right != S_merged_label_0) &&
                (S_pending_root_up_right != S_merged_label_1))
            C_phase_merge_label = S_pending_root_up_right;

        // The input sample cycle only resolves and registers the label.  Start
        // the component update on merge phase zero, when the label and pixel
        // coordinates are already registered.  This removes the long
        // line-RAM -> root/free-label -> statistics-table path from the
        // 75 MHz pixel cycle while retaining all three merge slots.
        if(S_merge_phase == 0) begin
            C_work_area_next = S_area[S_pending_base] + 1'b1;
            C_work_min_x_next = (S_area[S_pending_base] == 0) ?
                                S_pending_x :
                                ((S_pending_x < S_min_x[S_pending_base]) ?
                                 S_pending_x : S_min_x[S_pending_base]);
            C_work_max_x_next = (S_area[S_pending_base] == 0) ?
                                S_pending_x :
                                ((S_pending_x > S_max_x[S_pending_base]) ?
                                 S_pending_x : S_max_x[S_pending_base]);
            C_work_min_y_next = (S_area[S_pending_base] == 0) ?
                                S_pending_y :
                                ((S_pending_y < S_min_y[S_pending_base]) ?
                                 S_pending_y : S_min_y[S_pending_base]);
            C_work_max_y_next = (S_area[S_pending_base] == 0) ?
                                S_pending_y :
                                ((S_pending_y > S_max_y[S_pending_base]) ?
                                 S_pending_y : S_max_y[S_pending_base]);
        end else begin
            C_work_area_next = S_work_area;
            C_work_min_x_next = S_work_min_x;
            C_work_max_x_next = S_work_max_x;
            C_work_min_y_next = S_work_min_y;
            C_work_max_y_next = S_work_max_y;
        end
        if(C_phase_merge_label != 0) begin
            C_work_area_next = C_work_area_next + S_area[C_phase_merge_label];
            if(S_min_x[C_phase_merge_label] < C_work_min_x_next)
                C_work_min_x_next = S_min_x[C_phase_merge_label];
            if(S_max_x[C_phase_merge_label] > C_work_max_x_next)
                C_work_max_x_next = S_max_x[C_phase_merge_label];
            if(S_min_y[C_phase_merge_label] < C_work_min_y_next)
                C_work_min_y_next = S_min_y[C_phase_merge_label];
            if(S_max_y[C_phase_merge_label] > C_work_max_y_next)
                C_work_max_y_next = S_max_y[C_phase_merge_label];
        end
    end

    always @(posedge I_clk or negedge I_rst_n) begin
        if(!I_rst_n) begin
            S_left_label <= 0;
            S_up_left_label <= 0;
            S_best_area <= 0;
            S_first_row <= 1'b1;
            S_row_root_active <= 0;
            S_row_label_present <= 0;
            S_busy <= 1'b0;
            S_merge_phase <= 0;
            S_pending_base <= 0;
            S_pending_root_left <= 0;
            S_pending_root_up_left <= 0;
            S_pending_root_up <= 0;
            S_pending_root_up_right <= 0;
            S_pending_x <= 0;
            S_pending_y <= 0;
            S_merged_label_0 <= 0;
            S_merged_label_1 <= 0;
            S_pending_row_end <= 1'b0;
            S_pending_final_row <= 1'b0;
            S_work_area <= 0;
            S_work_min_x <= 0;
            S_work_max_x <= 0;
            S_work_min_y <= 0;
            S_work_max_y <= 0;
            S_cleanup_active <= 1'b0;
            S_cleanup_final_row <= 1'b0;
            S_cleanup_index <= 1;
            O_bbox_valid <= 1'b0;
            O_bbox_x <= 0;
            O_bbox_y <= 0;
            O_bbox_width <= 0;
            O_bbox_height <= 0;
            O_component_area <= 0;
            O_label_overflow <= 1'b0;
            O_frame_done <= 1'b0;
            for(R_i = 0; R_i <= LABEL_COUNT; R_i = R_i + 1) begin
                S_parent[R_i] <= 0;
                S_area[R_i] <= 0;
                S_min_x[R_i] <= 0;
                S_max_x[R_i] <= 0;
                S_min_y[R_i] <= 0;
                S_max_y[R_i] <= 0;
            end
        end else begin
            O_frame_done <= 1'b0;
            if(I_frame_start) begin
                S_best_area <= 0;
                S_left_label <= 0;
                S_up_left_label <= 0;
                S_first_row <= 1'b1;
                S_row_root_active <= 0;
                S_row_label_present <= 0;
                S_busy <= 1'b0;
                S_cleanup_active <= 1'b0;
                O_label_overflow <= 1'b0;
            end

            if(S_cleanup_active) begin
                S_row_root_active[S_cleanup_index] <= 1'b0;
                S_row_label_present[S_cleanup_index] <= 1'b0;
                if(S_parent[S_cleanup_index] != 0) begin
                    if(S_cleanup_final_row ||
                       !S_row_label_present[S_cleanup_index])
                        S_parent[S_cleanup_index] <= 0;
                end else if((S_area[S_cleanup_index] != 0) &&
                            (S_cleanup_final_row ||
                             !S_row_root_active[S_cleanup_index])) begin
                    if(S_area[S_cleanup_index] > S_best_area) begin
                        S_best_area <= S_area[S_cleanup_index];
                        O_bbox_valid <=
                            (S_area[S_cleanup_index] >= MIN_COMPONENT_PIXELS);
                        O_bbox_x <= S_min_x[S_cleanup_index];
                        O_bbox_y <= S_min_y[S_cleanup_index];
                        O_bbox_width <= S_max_x[S_cleanup_index] -
                                        S_min_x[S_cleanup_index] + 1'b1;
                        O_bbox_height <= S_max_y[S_cleanup_index] -
                                         S_min_y[S_cleanup_index] + 1'b1;
                        O_component_area <= S_area[S_cleanup_index];
                    end
                    S_area[S_cleanup_index] <= 0;
                end

                if(S_cleanup_index == LABEL_COUNT) begin
                    S_cleanup_active <= 1'b0;
                    S_cleanup_index <= 1;
                    if(S_cleanup_final_row)
                        O_frame_done <= 1'b1;
                end else begin
                    S_cleanup_index <= S_cleanup_index + 1'b1;
                end
            end

            if(S_busy) begin
                S_work_area <= C_work_area_next;
                S_work_min_x <= C_work_min_x_next;
                S_work_max_x <= C_work_max_x_next;
                S_work_min_y <= C_work_min_y_next;
                S_work_max_y <= C_work_max_y_next;

                if(C_phase_merge_label != 0) begin
                    S_parent[C_phase_merge_label] <= S_pending_base;
                    S_area[C_phase_merge_label] <= 0;
                    S_row_root_active[C_phase_merge_label] <= 1'b0;
                    for(R_i = 1; R_i <= LABEL_COUNT; R_i = R_i + 1) begin
                        if(S_parent[R_i] == C_phase_merge_label)
                            S_parent[R_i] <= S_pending_base;
                    end
                end

                if(S_merge_phase == 0)
                    S_merged_label_0 <= C_phase_merge_label;
                else if(S_merge_phase == 1)
                    S_merged_label_1 <= C_phase_merge_label;

                if(S_merge_phase == 2) begin
                    S_area[S_pending_base] <= C_work_area_next;
                    S_min_x[S_pending_base] <= C_work_min_x_next;
                    S_max_x[S_pending_base] <= C_work_max_x_next;
                    S_min_y[S_pending_base] <= C_work_min_y_next;
                    S_max_y[S_pending_base] <= C_work_max_y_next;
                    S_busy <= 1'b0;
                    S_merge_phase <= 0;
                    if(S_pending_row_end) begin
                        S_cleanup_active <= 1'b1;
                        S_cleanup_final_row <= S_pending_final_row;
                        S_cleanup_index <= 1;
                        if(S_pending_final_row && (S_best_area == 0)) begin
                            O_bbox_valid <= 1'b0;
                            O_bbox_x <= 0;
                            O_bbox_y <= 0;
                            O_bbox_width <= 0;
                            O_bbox_height <= 0;
                            O_component_area <= 0;
                        end
                    end
                end else begin
                    S_merge_phase <= S_merge_phase + 1'b1;
                end
            end else if(I_sample_valid && !S_cleanup_active) begin
                // Preserve the old up label before overwriting this column.
                S_up_left_label <= I_row_end ? 0 : S_label_line[I_col];
                S_left_label <= 0;
                S_merged_label_0 <= 0;
                S_merged_label_1 <= 0;
                if(I_row_end)
                    S_cleanup_index <= 1;
                if(I_row_end)
                    S_first_row <= 1'b0;

                if(I_mask && (C_assigned_label != 0)) begin
                    S_label_line[I_col] <= C_assigned_label;
                    if(!I_row_end)
                        S_left_label <= C_assigned_label;
                    S_row_root_active[C_assigned_label] <= 1'b1;
                    S_row_label_present[C_assigned_label] <= 1'b1;
                    S_pending_base <= C_assigned_label;
                    S_pending_root_left <= C_root_left;
                    S_pending_root_up_left <= C_root_up_left;
                    S_pending_root_up <= C_root_up;
                    S_pending_root_up_right <= C_root_up_right;
                    S_pending_x <= I_x;
                    S_pending_y <= I_y;
                    S_pending_row_end <= I_row_end;
                    S_pending_final_row <= I_final_row;
                    S_busy <= 1'b1;
                    S_merge_phase <= 0;
                end else begin
                    S_label_line[I_col] <= 0;
                    if(I_mask && (C_assigned_label == 0))
                        O_label_overflow <= 1'b1;
                    if(I_row_end) begin
                        S_cleanup_active <= 1'b1;
                        S_cleanup_final_row <= I_final_row;
                        if(I_final_row && (S_best_area == 0)) begin
                            O_bbox_valid <= 1'b0;
                            O_bbox_x <= 0;
                            O_bbox_y <= 0;
                            O_bbox_width <= 0;
                            O_bbox_height <= 0;
                            O_component_area <= 0;
                        end
                    end
                end
            end
        end
    end
endmodule
