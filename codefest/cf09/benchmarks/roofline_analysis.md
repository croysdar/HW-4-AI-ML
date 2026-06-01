# CF09 CLLM -- Roofline Analysis

ECE 510 Spring 2026 | Rebecca Gilbert-Croysdale

---

The accelerator operating point is plotted as **projected** at AI = 7.8 FLOP/byte and
74.9 GOPS -- placing it in the memory-bound region of the sky130 roofline, below the
ridge point of 16 FLOP/byte. The dominant uncertainty in this projection is the
assumed 100% utilization of the `compute_core` pipeline at 300 MHz. In reality, the
M3 post-P&R standalone synthesis violates timing by 2.4 ns (WNS), meaning the actual
achievable clock may be substantially lower unless in-context integration within
`bnn_top` closes the timing as expected. A 150 MHz clock, for example, would halve
attainable performance to ~37 GOPS, pushing the point further into the memory-bound
region. The second major uncertainty is the AXI4-Stream handshaking overhead: the
projection assumes zero pipeline stalls between tiles, but real-world backpressure from
weight SRAM read latency (1-cycle registered read) introduces at minimum a 1-cycle
bubble per tile, reducing effective throughput by up to 50% on short tiles. Converting
the projected point to a measurement requires: (1) running a full end-to-end cocotb
simulation of an entire conv3 layer and recording cycle count, and (2) completing M4
in-context P&R to obtain a verified post-route clock frequency before computing the
final GOPS figure.
