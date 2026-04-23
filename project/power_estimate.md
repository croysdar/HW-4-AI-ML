# Power Estimate — Wildlife Camera BNN SoC

ECE 510 Spring 2026

---

## System Component Estimates

| Component | Estimate | Notes |
|---|---|---|
| BNN accelerator chiplet | 1–20 mW | Based on YodaNN (1.5 TOPS @ 895µW, 65nm) and XNORBIN (95 TOPS/W). Our 1.2 TOPS target → ~12–20 mW. |
| Image sensor | 100–150 mW | Typical low-power CMOS (e.g. OmnivisionOV5640) at 30fps continuous streaming. |
| Host ARM CPU | 50–300 mW | Cortex-A class (Pi/Jetson style): ~300mW. Cortex-M class (STM32/ESP32-S3): ~50mW. |
| DDR4 DRAM | 200–400 mW | Standard edge DDR4. Reducible with LPDDR4 or on-chip SRAM where possible. |
| **Total system** | **~0.3–0.7 W** | Conservative target: stay under 1W. |

---

## Architecture Note — Hybrid Precision Partition

The final architecture uses hybrid precision: Conv1 (8-bit) runs on the host ARM CPU; Conv2–4 (1-bit) run on the chiplet via XNOR+Popcount. This keeps the chiplet purely binary — no additional hardware complexity — while improving accuracy from 73% to 81%+. Conv1 on the host CPU adds negligible load (one 8-bit conv layer) and is already captured in the ARM CPU power estimate above.

## Key Insight

The BNN accelerator chiplet is nearly negligible in the power budget. The sensor, CPU, and DRAM are the real bottlenecks. The engineering challenge for 24/7 operation is duty-cycling those components to match the accelerator's efficiency — not the AI compute itself.

---

## Solar Panel Sizing (24/7 Operation)

Assuming 1W continuous system draw (conservative upper bound):

```
Daily energy needed : 1W × 24h = 24 Wh
Peak sun hours/day  : ~4.5 h (accounting for clouds, winter, latitude)
Required generation : 24 Wh ÷ 4.5 h = 5.3 W
Efficiency losses   : ~20–30% (battery charging circuit, voltage regulators)
```

**Recommended panel: 10W**
- Covers a 1W system with margin for cloudy days and seasonal variation
- Roughly the size of a sheet of paper / small laptop solar fabric
- A 5W panel is the minimum; 10W guarantees reliable 24/7 operation year-round

---

## References

- **YodaNN** — BNN accelerator: 1.5 TOPS @ 895 µW on 65nm process
- **XNORBIN** — 95 TOPS/W efficiency; at 1.2 TOPS target → ~12.6 mW
- **OmniVision OV5640** — ~140 mW at 30fps active streaming
- Cross-validated with Gemini analysis, April 2026

---

## Notes for Milestone 4

When OpenLane 2 synthesis produces a power report, expect the accelerator number to be very small (single-digit mW range). This is correct — not an error. The synthesis report will provide the ground-truth number to replace the estimate in this document.
