# Yosys synthesis script — bnn_top (integrated) targeting Sky130 HD, 300 MHz
# Run from project/m3/ with: yosys synth/synthesize.tcl

set LIB "/Users/rebeccagilbert-croysdale/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
set PERIOD_PS 3333

yosys read_liberty -lib $LIB

# Read all three source files: top + both M2 submodules
yosys read_verilog -sv rtl/top.sv
yosys read_verilog -sv ../../project/m2/rtl/interface.sv
yosys read_verilog -sv ../../project/m2/rtl/compute_core.sv

# Synthesise with bnn_top as the root, flatten hierarchy for timing analysis
yosys synth -top bnn_top -flatten

yosys dfflibmap -liberty $LIB
yosys abc -liberty $LIB -D $PERIOD_PS
yosys clean

yosys stat -liberty $LIB

yosys write_verilog -noattr synth/bnn_top_netlist.v
yosys write_json   synth/bnn_top_netlist.json

yosys tee -o synth/yosys_synth_raw.log stat -liberty $LIB
