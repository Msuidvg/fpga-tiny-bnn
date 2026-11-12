create_clock -name {sys_clk_50m} -period 20.000 [get_ports {I_sys_clk}]
# K3 asserts reset asynchronously. Deassertion is synchronized in replay_top.
set_false_path -from [get_ports {I_reset_n}]
# UART TX is sampled by the USB adapter on its own timebase.
set_false_path -to [get_ports {O_uart_tx}]
