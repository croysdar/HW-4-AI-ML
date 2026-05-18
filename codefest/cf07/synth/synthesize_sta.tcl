# Yosys synthesis + STA — compute_core targeting Sky130 HD, 300 MHz (3.33 ns)

set LIB "/Users/rebeccagilbert-croysdale/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
set PERIOD_PS 3333

yosys read_liberty -lib $LIB
yosys read_verilog -sv hdl/synth_top.sv
yosys synth -top compute_core -flatten
yosys dfflibmap -liberty $LIB
yosys abc -liberty $LIB -D $PERIOD_PS
yosys clean

# Area statistics
yosys tee -o synth/stat_report.txt stat -liberty $LIB

# Built-in STA (produces worst-case path delay per clock domain)
yosys tee -o synth/sta_report.txt sta

# Time estimate
yosys tee -o synth/timeest_report.txt timeest

yosys write_verilog -noattr synth/compute_core_netlist.v
