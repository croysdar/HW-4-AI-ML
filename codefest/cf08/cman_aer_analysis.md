# CMAN — AER Bandwidth Analysis

N = 1024 output neurons, f = 50 Hz mean firing rate, packets = 20 bits each (10-bit address + 6-bit timestamp + 4-bit framing/parity). Firing is independent (Poisson-like). Packets emitted asynchronously.

## 1. Mean Aggregate Spike Rate

R = N × f

R = 1024 × 50 = **51,200 spikes/second**

## 2. Mean AER Bandwidth

B = R × 20

B = 51,200 × 20 = 1,024,000 bits/second = **1.024 Mbit/s**

## 3. Interface Comparison

| Interface  | Limit        | Sustains Mean (1.024 Mbit/s)? |
|------------|--------------|-------------------------------|
| I²C        | ≤3.4 Mbit/s  | **Y**                         |
| SPI        | ≤50 Mbit/s   | Y                             |
| AXI4-Lite  | ~100 Mbit/s  | Y                             |

**Lowest-complexity interface that suffices: I²C**

## 4. Burst Peak Bandwidth

25% of 1024 neurons fire within a 1 ms window:

- Burst packet count: 0.25 × 1024 = **256 packets**
- Burst bits: 256 × 20 = 5,120 bits in 1 ms
- Peak bandwidth: 5,120 bits / 0.001 s = **5.12 Mbit/s**

Burst-to-mean ratio: 5.12 / 1.024 = **5:1**

I²C (≤3.4 Mbit/s) cannot absorb the burst. **Buffering is required.**

During the 1 ms burst, I²C can drain: 3.4 Mbit/s × 0.001 s = 3,400 bits. Excess: 5,120 − 3,400 = **1,720 bits = 86 packets**. A buffer holding at least 86 packets (1,720 bits) is required to absorb the burst without dropping events.

## 5. Frame-Based Comparison

A conventional readout samples all 1024 neurons every 1 ms (1000 frames/sec), sending 1 bit per neuron per sample:

B_frame = 1024 × 1 × 1000 = 1,024,000 bits/s = **1.024 Mbit/s**

AER-to-frame ratio at f = 50 Hz: 1.024 / 1.024 = **1:1**

**Crossover firing rate f_crossover:** Set AER bandwidth equal to frame bandwidth:

N × f × 20 = N × 1000

f_crossover = 1000 / 20 = **50 Hz**

AER and frame-based bandwidth are equal at exactly f = 50 Hz; below this rate AER uses less bandwidth than frame-based readout, making AER the right choice for sparse, low-firing-rate networks where most neurons are silent most of the time.
