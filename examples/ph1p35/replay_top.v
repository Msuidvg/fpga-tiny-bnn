`timescale 1ns/1ps

// Replay three published masks on the PH1P35 receiver board.
// CORE_ONLY selects 32x32 masks and omits CCL and normalization.
module replay_top #(
    parameter CORE_ONLY = 0,
    parameter CLOCK_HZ = 50000000,
    parameter BAUD = 115200
) (
    input wire I_sys_clk,
    input wire I_reset_n,
    output wire O_uart_tx
);
    localparam CASE_BYTES = CORE_ONLY ? 128 : 4608;
    localparam ROM_BYTES = 3 * CASE_BYTES;
    localparam ADDR_BITS = $clog2(ROM_BYTES);
    localparam LOAD_REQ = 0, LOAD_BYTE = 1, START = 2, STREAM = 3,
               ROW_GAP = 4, WAIT_RESULT = 5, SEND = 6, DRAIN = 7;

    reg [1:0] reset_sync;
    always @(posedge I_sys_clk or negedge I_reset_n) begin
        if (!I_reset_n) reset_sync <= 2'b00;
        else reset_sync <= {reset_sync[0], 1'b1};
    end
    wire reset_n = reset_sync[1];

    reg [7:0] samples [0:ROM_BYTES-1];
    initial $readmemh("samples.mem", samples);
    reg [ADDR_BITS-1:0] rom_addr;
    reg [7:0] rom_data;
    always @(posedge I_sys_clk) rom_data <= samples[rom_addr];

    reg [2:0] state;
    reg [1:0] sample_id;
    reg [7:0] col, row;
    reg [1:0] pixel_phase;
    reg [5:0] gap_left;
    reg [6:0] load_index;
    reg [1023:0] core_mask;
    reg [31:0] cycle_count, start_cycle;
    wire [31:0] elapsed = cycle_count - start_cycle;
    wire sample_valid = state == STREAM && pixel_phase == 0;
    wire row_end = sample_valid && col == 8'd255;
    wire result_valid, overflow;
    wire [2:0] gesture;
    wire [7:0] margin;
    wire signed [31:0] score1, score2, score3, score4, score5;

    generate if (CORE_ONLY) begin : g_core
        assign overflow = 1'b0;
        gesture_bnn_core u_core (
            .I_clk(I_sys_clk), .I_rst_n(reset_n), .I_start(state == START),
            .I_mask(core_mask), .O_result_valid(result_valid),
            .O_gesture(gesture), .O_confidence(margin), .O_class_score_1(score1),
            .O_class_score_2(score2), .O_class_score_3(score3),
            .O_class_score_4(score4), .O_class_score_5(score5)
        );
    end else begin : g_pipeline
        gesture_bnn_pipeline u_pipeline (
            .I_clk(I_sys_clk), .I_rst_n(reset_n), .I_frame_start(state == START),
            .I_sample_valid(sample_valid), .I_row_end(row_end),
            .I_final_row(row_end && row == 8'd143), .I_col(col), .I_row(row),
            .I_mask(rom_data[col[2:0]]), .O_label_overflow(overflow),
            .O_result_valid(result_valid), .O_gesture(gesture), .O_confidence(margin),
            .O_class_score_1(score1), .O_class_score_2(score2),
            .O_class_score_3(score3), .O_class_score_4(score4), .O_class_score_5(score5)
        );
    end endgenerate

    reg [239:0] packet;
    reg [4:0] byte_index;
    wire tx_busy;
    wire tx_start = state == SEND && !tx_busy;
    replay_uart_tx #(.CLOCK_HZ(CLOCK_HZ), .BAUD(BAUD)) u_uart (
        .clk(I_sys_clk), .reset_n(reset_n), .start(tx_start),
        .data(packet[7:0]), .tx(O_uart_tx), .busy(tx_busy)
    );

    always @(posedge I_sys_clk) begin
        if (!reset_n) begin
            state <= CORE_ONLY ? LOAD_REQ : START;
            rom_addr <= 0;
            sample_id <= 0;
            col <= 0;
            row <= 0;
            pixel_phase <= 0;
            gap_left <= 0;
            load_index <= 0;
            cycle_count <= 0;
            start_cycle <= 0;
            packet <= 0;
            byte_index <= 0;
        end else begin
            cycle_count <= cycle_count + 1'b1;
            if ((state == LOAD_BYTE) || (sample_valid && col[2:0] == 3'd7))
                rom_addr <= (rom_addr == ROM_BYTES-1) ? 0 : rom_addr + 1'b1;

            case (state)
                LOAD_REQ: state <= LOAD_BYTE;
                LOAD_BYTE: begin
                    core_mask[load_index*8 +: 8] <= rom_data;
                    load_index <= load_index + 1'b1;
                    state <= (load_index == 7'd127) ? START : LOAD_REQ;
                end
                START: begin
                    start_cycle <= cycle_count;
                    row <= 0;
                    col <= 0;
                    pixel_phase <= 0;
                    state <= CORE_ONLY ? WAIT_RESULT : STREAM;
                end
                STREAM: begin
                    pixel_phase <= pixel_phase + 1'b1;
                    if (pixel_phase == 3) begin
                        col <= col + 1'b1;
                        if (col == 8'd255) begin
                            row <= row + 1'b1;
                            gap_left <= 40;
                            state <= (row == 8'd143) ? WAIT_RESULT : ROW_GAP;
                        end
                    end
                end
                ROW_GAP: begin
                    gap_left <= gap_left - 1'b1;
                    if (gap_left == 1) state <= STREAM;
                end
                WAIT_RESULT: begin
                    if (result_valid) begin
                        packet <= {elapsed, score5, score4, score3, score2, score1,
                                   margin, 5'b0, gesture, 7'b0, overflow, 6'b0, sample_id, 16'h4e42};
                        byte_index <= 0;
                        state <= SEND;
                    end else if (elapsed == 32'd350000) begin
                        packet <= {elapsed, 176'b0, 8'd2, 6'b0, sample_id, 16'h4e42};
                        byte_index <= 0;
                        state <= SEND;
                    end
                end
                SEND: if (tx_start) begin
                    packet <= packet >> 8;
                    byte_index <= byte_index + 1'b1;
                    if (byte_index == 29) state <= DRAIN;
                end
                DRAIN: if (!tx_busy) begin
                    sample_id <= (sample_id == 2) ? 0 : sample_id + 1'b1;
                    state <= CORE_ONLY ? LOAD_REQ : START;
                end
                default: state <= CORE_ONLY ? LOAD_REQ : START;
            endcase
        end
    end
endmodule

module replay_uart_tx #(
    parameter CLOCK_HZ = 50000000,
    parameter BAUD = 115200
) (
    input wire clk, reset_n, start,
    input wire [7:0] data,
    output wire tx, busy
);
    localparam TICKS = (CLOCK_HZ + BAUD/2) / BAUD;
    localparam DIV_BITS = (TICKS > 1) ? $clog2(TICKS) : 1;
    reg [DIV_BITS-1:0] divider;
    reg [9:0] bits;
    reg [3:0] remaining;
    assign busy = remaining != 0;
    assign tx = busy ? bits[0] : 1'b1;

    always @(posedge clk) begin
        if (!reset_n) begin
            remaining <= 0;
            divider <= 0;
            bits <= 10'h3ff;
        end else if (!busy) begin
            if (start) begin
                bits <= {1'b1, data, 1'b0};
                remaining <= 10;
                divider <= TICKS-1;
            end
        end else if (divider == 0) begin
            bits <= {1'b1, bits[9:1]};
            remaining <= remaining - 1'b1;
            divider <= TICKS-1;
        end else begin
            divider <= divider - 1'b1;
        end
    end
endmodule
