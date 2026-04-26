import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly


@cocotb.test()
async def test_bnn_reset(dut):
    """Drive reset and verify accumulator clears to zero."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    dut.rst.value = 1
    dut.s_axis_act.value = 0
    dut.s_axis_valid.value = 0
    dut.weight.value = 0

    await RisingEdge(dut.clk)
    await ReadOnly()
    assert dut.acc.value.to_signed() == 0, \
        f"Expected acc=0 after reset, got {dut.acc.value.to_signed()}"

    dut.rst.value = 0


@cocotb.test()
async def test_bnn_xnor_popcount(dut):
    """Apply one representative XNOR-popcount transaction and check accumulation."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    dut.rst.value = 1
    dut.s_axis_act.value = 0
    dut.s_axis_valid.value = 0
    dut.weight.value = 0

    await RisingEdge(dut.clk)      # reset cycle

    # All-ones activations and all-ones weights: XNOR = all-ones → popcount = 256
    # dot = 2*256 - 256 = 256
    dut.rst.value = 0
    dut.s_axis_act.value = (1 << 256) - 1   # all 1s
    dut.weight.value = (1 << 256) - 1        # all 1s
    dut.s_axis_valid.value = 1

    await RisingEdge(dut.clk)
    await ReadOnly()
    result = dut.acc.value.to_signed()
    dut._log.info(f"All-ones XNOR-popcount: acc = {result} (expected 256)")
    assert result == 256, f"Expected 256, got {result}"
