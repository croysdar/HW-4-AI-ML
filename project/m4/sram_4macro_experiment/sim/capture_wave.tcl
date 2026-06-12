# Auto-capture waveform screenshot for 4-SRAM bnn_top simulation.
# Run via: gtkwave --script capture_wave.tcl cosim_run.vcd

set end_time [gtkwave::getEndTime]

# Zoom to show the full simulation with a small margin
gtkwave::setZoomFactor -32
gtkwave::setMarker -1

# Show from time 0 to end
gtkwave::nop

set outfile [file normalize "project/m4/sram_4macro_experiment/sim/final_waveform.png"]
gtkwave::hardCopyFile $outfile

puts "Waveform written to $outfile"
gtkwave::quit
