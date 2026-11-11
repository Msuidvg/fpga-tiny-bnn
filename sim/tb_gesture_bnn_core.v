`timescale 1ns/1ps

`include "bnn_test_vectors.vh"

`ifndef BNN_INPUT_BITS
`define BNN_INPUT_BITS 1024
`define BNN_HIDDEN1_UNITS 96
`define BNN_HIDDEN2_UNITS 48
`define BNN_OUTPUT_UNITS 5
`define BNN_L1_POPCOUNT_BITS 11
`define BNN_L2_POPCOUNT_BITS 7
`define BNN_L3_POPCOUNT_BITS 6
`define BNN_INFERENCE_CYCLES 103153
`endif

module tb_gesture_bnn_core;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg start = 1'b0;
    reg [`BNN_INPUT_BITS-1:0] mask = 0;
    wire busy;
    wire result_valid;
    wire [2:0] gesture;
    wire [7:0] confidence;
    wire signed [31:0] class_score_1;
    wire signed [31:0] class_score_2;
    wire signed [31:0] class_score_3;
    wire signed [31:0] class_score_4;
    wire signed [31:0] class_score_5;

    reg [`BNN_INPUT_BITS-1:0] vectors [0:`BNN_VECTOR_COUNT-1];
    reg [2:0] expected_labels [0:`BNN_VECTOR_COUNT-1];
    reg [7:0] expected_confidence [0:`BNN_VECTOR_COUNT-1];
    reg [`BNN_HIDDEN1_UNITS-1:0] expected_h1 [0:`BNN_VECTOR_COUNT-1];
    reg [`BNN_HIDDEN2_UNITS-1:0] expected_h2 [0:`BNN_VECTOR_COUNT-1];
    reg [`BNN_HIDDEN1_UNITS*`BNN_L1_POPCOUNT_BITS-1:0]
        expected_l1_popcounts [0:`BNN_VECTOR_COUNT-1];
    reg [`BNN_HIDDEN2_UNITS*`BNN_L2_POPCOUNT_BITS-1:0]
        expected_l2_popcounts [0:`BNN_VECTOR_COUNT-1];
    reg [`BNN_OUTPUT_UNITS*`BNN_L3_POPCOUNT_BITS-1:0]
        expected_l3_popcounts [0:`BNN_VECTOR_COUNT-1];
    reg [159:0] expected_scores [0:`BNN_VECTOR_COUNT-1];

    integer vector_index;
    integer neuron;
    integer cycles;
    integer reference_cycles;

    always #5 clk = ~clk;

    gesture_bnn_core #(
        .H1_UNITS(`BNN_HIDDEN1_UNITS),
        .H2_UNITS(`BNN_HIDDEN2_UNITS),
        .W1_MEM_FILE(`BNN_W1_FILE),
        .W2_MEM_FILE(`BNN_W2_FILE),
        .W3_MEM_FILE(`BNN_W3_FILE),
        .T1_MEM_FILE(`BNN_T1_FILE),
        .T2_MEM_FILE(`BNN_T2_FILE),
        .B3_MEM_FILE(`BNN_B3_FILE)
    ) dut (
        .I_clk(clk),
        .I_rst_n(rst_n),
        .I_start(start),
        .I_mask(mask),
        .O_busy(busy),
        .O_result_valid(result_valid),
        .O_gesture(gesture),
        .O_confidence(confidence),
        .O_class_score_1(class_score_1),
        .O_class_score_2(class_score_2),
        .O_class_score_3(class_score_3),
        .O_class_score_4(class_score_4),
        .O_class_score_5(class_score_5)
    );

    task check_current_vector;
        begin
            if(gesture !== expected_labels[vector_index])
                $fatal(1,
                    "vector %0d class mismatch expected=%0d actual=%0d",
                    vector_index, expected_labels[vector_index], gesture);
            if(confidence !== expected_confidence[vector_index])
                $fatal(1,
                    "vector %0d confidence mismatch expected=%0d actual=%0d",
                    vector_index, expected_confidence[vector_index], confidence);
            if(class_score_1 !== $signed(expected_scores[vector_index][31:0]))
                $fatal(1, "vector %0d class-1 score mismatch", vector_index);
            if(class_score_2 !== $signed(expected_scores[vector_index][63:32]))
                $fatal(1, "vector %0d class-2 score mismatch", vector_index);
            if(class_score_3 !== $signed(expected_scores[vector_index][95:64]))
                $fatal(1, "vector %0d class-3 score mismatch", vector_index);
            if(class_score_4 !== $signed(expected_scores[vector_index][127:96]))
                $fatal(1, "vector %0d class-4 score mismatch", vector_index);
            if(class_score_5 !== $signed(expected_scores[vector_index][159:128]))
                $fatal(1, "vector %0d class-5 score mismatch", vector_index);
            if(dut.S_h1_activation !== expected_h1[vector_index])
                $fatal(1, "vector %0d hidden-1 activation mismatch", vector_index);
            if(dut.S_h2_activation !== expected_h2[vector_index])
                $fatal(1, "vector %0d hidden-2 activation mismatch", vector_index);
            for(neuron = 0; neuron < `BNN_HIDDEN1_UNITS; neuron = neuron + 1)
                if(dut.S_debug_l1_popcount[neuron] !==
                   expected_l1_popcounts[vector_index]
                       [neuron*`BNN_L1_POPCOUNT_BITS +: `BNN_L1_POPCOUNT_BITS])
                    $fatal(1,
                        "vector %0d layer-1 popcount mismatch neuron=%0d",
                        vector_index, neuron);
            for(neuron = 0; neuron < `BNN_HIDDEN2_UNITS; neuron = neuron + 1)
                if(dut.S_debug_l2_popcount[neuron] !==
                   expected_l2_popcounts[vector_index]
                       [neuron*`BNN_L2_POPCOUNT_BITS +: `BNN_L2_POPCOUNT_BITS])
                    $fatal(1,
                        "vector %0d layer-2 popcount mismatch neuron=%0d",
                        vector_index, neuron);
            for(neuron = 0; neuron < `BNN_OUTPUT_UNITS; neuron = neuron + 1)
                if(dut.S_debug_l3_popcount[neuron] !==
                   expected_l3_popcounts[vector_index]
                       [neuron*`BNN_L3_POPCOUNT_BITS +: `BNN_L3_POPCOUNT_BITS])
                    $fatal(1,
                        "vector %0d layer-3 popcount mismatch neuron=%0d",
                        vector_index, neuron);
        end
    endtask

    initial begin
        $readmemh(`BNN_VECTOR_INPUTS_FILE, vectors);
        $readmemh(`BNN_VECTOR_LABELS_FILE, expected_labels);
        $readmemh(`BNN_VECTOR_CONFIDENCE_FILE, expected_confidence);
        $readmemh(`BNN_VECTOR_H1_FILE, expected_h1);
        $readmemh(`BNN_VECTOR_H2_FILE, expected_h2);
        $readmemh(`BNN_VECTOR_L1_POPCOUNTS_FILE, expected_l1_popcounts);
        $readmemh(`BNN_VECTOR_L2_POPCOUNTS_FILE, expected_l2_popcounts);
        $readmemh(`BNN_VECTOR_L3_POPCOUNTS_FILE, expected_l3_popcounts);
        $readmemh(`BNN_VECTOR_SCORES_FILE, expected_scores);

        reference_cycles = -1;
        repeat(4) @(posedge clk);
        rst_n <= 1'b1;

        for(vector_index = 0; vector_index < `BNN_VECTOR_COUNT;
            vector_index = vector_index + 1) begin
            @(negedge clk);
            mask <= vectors[vector_index];
            start <= 1'b1;
            @(negedge clk);
            start <= 1'b0;
            cycles = 0;
            while(result_valid !== 1'b1 && cycles < `BNN_INFERENCE_CYCLES + 100) begin
                @(negedge clk);
                cycles = cycles + 1;
            end
            if(result_valid !== 1'b1)
                $fatal(1, "vector %0d timed out", vector_index);
            if(reference_cycles < 0)
                reference_cycles = cycles;
            else if(cycles != reference_cycles)
                $fatal(1,
                    "vector %0d latency changed expected=%0d actual=%0d",
                    vector_index, reference_cycles, cycles);
            check_current_vector();
            if((vector_index + 1) % 8 == 0)
                $display("checked %0d/%0d vectors", vector_index + 1,
                    `BNN_VECTOR_COUNT);
            @(negedge clk);
        end

        $display(
            "PASS tb_gesture_bnn_core vectors=%0d latency_clocks=%0d",
            `BNN_VECTOR_COUNT, reference_cycles);
        $finish;
    end
endmodule
