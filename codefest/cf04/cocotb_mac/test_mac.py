import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly

# cocotb 2.0 pattern: drive inputs in active region (after RisingEdge),
# then await ReadOnly() to read the committed post-NBA output value.


@cocotb.test()
async def test_mac_basic(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    dut.rst.value = 1
    dut.a.value = 0
    dut.b.value = 0

    # Cycle 0: Reset — drive rst=1 before first edge
    await RisingEdge(dut.clk)           # DFF captures rst=1 → out:=0
    dut.rst.value = 0                    # drive inputs for cycle 1
    dut.a.value = 3
    dut.b.value = 4
    await ReadOnly()                     # out=0 (post-reset, not checked)

    # Cycles 1–3: accumulate a=3, b=4
    await RisingEdge(dut.clk)           # DFF captures rst=0, a=3, b=4 → out:=12
    dut.rst.value = 0; dut.a.value = 3; dut.b.value = 4
    await ReadOnly()
    assert dut.out.value.to_signed() == 12, f"Expected 12, got {dut.out.value.to_signed()}"

    await RisingEdge(dut.clk)           # → out:=24
    dut.rst.value = 0; dut.a.value = 3; dut.b.value = 4
    await ReadOnly()
    assert dut.out.value.to_signed() == 24, f"Expected 24, got {dut.out.value.to_signed()}"

    await RisingEdge(dut.clk)           # → out:=36
    dut.rst.value = 1; dut.a.value = 0; dut.b.value = 0   # assert reset for next cycle
    await ReadOnly()
    assert dut.out.value.to_signed() == 36, f"Expected 36, got {dut.out.value.to_signed()}"

    # Cycle 4: reset asserted
    await RisingEdge(dut.clk)           # DFF captures rst=1 → out:=0
    dut.rst.value = 0; dut.a.value = -5; dut.b.value = 2
    await ReadOnly()
    assert dut.out.value.to_signed() == 0, f"Expected 0 after reset, got {dut.out.value.to_signed()}"

    # Cycles 5–6: accumulate a=−5, b=2
    await RisingEdge(dut.clk)           # → out:=−10
    dut.rst.value = 0; dut.a.value = -5; dut.b.value = 2
    await ReadOnly()
    assert dut.out.value.to_signed() == -10, f"Expected -10, got {dut.out.value.to_signed()}"

    await RisingEdge(dut.clk)           # → out:=−20
    await ReadOnly()
    assert dut.out.value.to_signed() == -20, f"Expected -20, got {dut.out.value.to_signed()}"


@cocotb.test()
async def test_mac_overflow(dut):
    """Verifies accumulator wraps (2's complement) — no saturation logic present."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    dut.rst.value = 1
    dut.a.value = 0
    dut.b.value = 0
    await RisingEdge(dut.clk)           # reset cycle

    dut.rst.value = 0
    dut.a.value = 127
    dut.b.value = 127                    # 16129 per cycle

    INT32_MAX = 2**31 - 1
    steps = (INT32_MAX // 16129) + 2    # guarantees at least one wrap

    for _ in range(steps):
        await RisingEdge(dut.clk)

    await ReadOnly()
    result = dut.out.value.to_signed()
    dut._log.info(f"After {steps} cycles of a=127*b=127: out = {result}")
    dut._log.info("Design WRAPS (2's complement) — no saturation logic present")
    assert result < 0, f"Expected negative after INT32 overflow (wrap), got {result}"
