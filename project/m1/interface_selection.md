# Interface Selection and Bandwidth Analysis

ECE 510 Spring 2026 — M1 Deliverable

---

## 1. Selected Interface

**Protocol:** AXI4-Stream (AMBA 4.0)
**Configuration:** 256-bit data width, clocked at 300 MHz.

**Justification for Wildlife Camera Edge Context:**
The accelerator targets an embedded edge SoC suitable for a trail camera. AXI4-Stream is ideal for this application because it facilitates high-speed, unidirectional data transfers without the overhead of issuing memory addresses for every transaction. This maximizes throughput and minimizes power consumption when streaming continuous image tensors between the CPU host and the hardware accelerator.

## 2. Required Bandwidth Calculation

Based on the hardware partition defined in `partition_rationale.md` and the roofline plot, the custom BNN accelerator targets the following operating point:

- **Target Performance:** 1200 GFLOP/s
- **Hardware Arithmetic Intensity:** 150 FLOP/byte

The minimum memory bandwidth required to sustain this throughput is:
`Required Bandwidth = Target Performance / Arithmetic Intensity`
`Required Bandwidth = 1200 GFLOP/s / 150 FLOP/byte = 8.0 GB/s`

## 3. Rated Interface Bandwidth Calculation

The theoretical maximum (rated) bandwidth of the chosen AXI4-Stream interface is calculated as follows:

- **Bus Width:** 256 bits (32 bytes)
- **Clock Frequency:** 300 MHz

`Rated Bandwidth = Bus Width (bytes) × Clock Frequency`
`Rated Bandwidth = 32 bytes/cycle × 300 MHz = 9,600 MB/s = 9.6 GB/s`

## 4. Conclusion

The chosen 256-bit AXI4-Stream interface provides a rated bandwidth of **9.6 GB/s**, which exceeds the required bandwidth of **8.0 GB/s**. This 20% margin ensures the accelerator will remain compute-bound rather than interface-bound at its target performance of 1200 GFLOP/s.
