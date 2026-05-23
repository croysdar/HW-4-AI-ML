# Yosys synthesis script — compute_core (pipelined) targeting Sky130 HD, 300 MHz
# Synthesises only the pipelined compute_core to verify the timing fix.
# The full bnn_top synthesis is infeasible with Yosys because the 1512×256-bit
# weight memory (387K FFs) exhausts ABC. See synthesis_notes.md for rationale.

set LIB "/Users/rebeccagilbert-croysdale/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
set PERIOD_PS 3333

yosys read_liberty -lib $LIB
yosys read_verilog -sv ../rtl/compute_core.sv

yosys synth -top compute_core -flatten
yosys dfflibmap -liberty $LIB
yosys abc -liberty $LIB -D $PERIOD_PS
yosys clean

yosys stat -liberty $LIB

yosys write_verilog -noattr synth/compute_core_pipelined_netlist.v
yosys tee -o synth/yosys_synth_raw.log stat -liberty $LIB
