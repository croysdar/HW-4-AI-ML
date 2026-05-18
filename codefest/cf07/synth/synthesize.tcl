# Yosys synthesis script — compute_core targeting Sky130 HD, 300 MHz (3.33 ns)
# Run from codefest/cf07/ with: yosys synth/synthesize.tcl

set LIB "/Users/rebeccagilbert-croysdale/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
set PERIOD_PS 3333

# Read Liberty model for synthesis
yosys read_liberty -lib $LIB

# Read design (SystemVerilog)
yosys read_verilog -sv hdl/synth_top.sv

# Hierarchical synthesis
yosys synth -top compute_core -flatten

# Map flip-flops to library cells
yosys dfflibmap -liberty $LIB

# Technology map with ABC — -D <delay_ps> constrains to 300 MHz
yosys abc -liberty $LIB -D $PERIOD_PS

# Clean up
yosys clean

# Reports
yosys stat -liberty $LIB

# Write outputs
yosys write_verilog -noattr synth/compute_core_netlist.v
yosys write_json synth/compute_core_netlist.json

# Timing report via ABC's internal STA
yosys tee -o synth/yosys_synthesis.log stat -liberty $LIB
