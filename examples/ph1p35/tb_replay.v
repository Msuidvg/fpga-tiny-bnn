`timescale 1ns/1ps

module tb_replay #(parameter CORE_ONLY = 0);
    localparam TICKS = 10;
    reg clk = 0;
    reg reset_n = 1;
    wire tx;
    always #5 clk = ~clk;
    replay_top #(.CORE_ONLY(CORE_ONLY), .CLOCK_HZ(115200*TICKS)) dut (
        .I_sys_clk(clk), .I_reset_n(reset_n), .O_uart_tx(tx)
    );

    reg [7:0] expected [0:89];
    reg [7:0] received;
    reg [31:0] cycles;
    integer packet_number, byte_number, fixture;

    task receive_byte;
        output reg [7:0] value;
        integer i;
        begin
            @(negedge tx);
            repeat (TICKS/2) @(posedge clk);
            #1;
            if (tx !== 1'b0) $fatal(1, "invalid UART start bit");
            for (i = 0; i < 8; i = i + 1) begin
                repeat (TICKS) @(posedge clk);
                #1;
                value[i] = tx;
            end
            repeat (TICKS) @(posedge clk);
            #1;
            if (tx !== 1'b1) $fatal(1, "invalid UART stop bit");
        end
    endtask

    initial begin
        $readmemh("expected_packets.mem", expected);
        #2 reset_n = 0;
        repeat (5) @(negedge clk);
        reset_n = 1;
        for (packet_number = 0; packet_number < 4; packet_number = packet_number + 1) begin
            fixture = packet_number % 3;
            for (byte_number = 0; byte_number < 30; byte_number = byte_number + 1) begin
                receive_byte(received);
                if (received !== expected[fixture*30 + byte_number])
                    $fatal(1, "UART mismatch case=%0d byte=%0d got=%02x expected=%02x",
                           fixture, byte_number, received, expected[fixture*30 + byte_number]);
                if (byte_number >= 26)
                    cycles[(byte_number-26)*8 +: 8] = received;
            end
            $display("UART case=%0d cycles=%0d", fixture, cycles);
        end
        $display("PASS tb_replay CORE_ONLY=%0d packets=4", CORE_ONLY);
        $finish;
    end
    initial begin
        #20000000;
        $fatal(1, "timeout waiting for replay packets");
    end
endmodule

module tb_replay_core;
    tb_replay #(.CORE_ONLY(1)) test();
endmodule
