# MAC Code Review

## LLM Attribution

| File          | LLM               |
| ------------- | ----------------- |
| `mac_llm_A.v` | Claude Sonnet 4.6 |
| `mac_llm_B.v` | Gemini 3.1 (Pro)  |

---

## Compile Results

Note: `iverilog` defaults to Verilog-2001 and rejects SystemVerilog keywords (`always_ff`, `logic`) without the `-g2012` flag. This flag is required even though the assignment does not mention it explicitly.

**mac_llm_A.v** — `iverilog -g2012 mac_llm_A.v`

```
No errors or warnings.
```

**mac_llm_B.v** — `iverilog -g2012 mac_llm_B.v`

```
No errors or warnings.
```

---

## Simulation Results

**mac_llm_A.v** — `iverilog -g2012 -o mac_a_sim mac_llm_A.v mac_tb.v && vvp mac_a_sim`

```
Cycle 1: out=12 (expect 12)
Cycle 2: out=24 (expect 24)
Cycle 3: out=36 (expect 36)
After rst: out=0 (expect 0)
Cycle 5: out=-10 (expect -10)
Cycle 6: out=-20 (expect -20)
mac_tb.v:32: $finish called at 66000 (1ps)
```

All outputs match. ✓

**mac_llm_B.v** — `iverilog -g2012 -o mac_b_sim mac_llm_B.v mac_tb.v && vvp mac_b_sim`

```
Cycle 1: out=12 (expect 12)
Cycle 2: out=24 (expect 24)
Cycle 3: out=36 (expect 36)
After rst: out=0 (expect 0)
Cycle 5: out=-10 (expect -10)
Cycle 6: out=-20 (expect -20)
mac_tb.v:32: $finish called at 66000 (1ps)
```

All outputs match. ✓

---

## Code Review Issues

### Issue 1 — Missing explicit sign cast on both outputs (sign extension risk)

**Offending line (`mac_llm_A.v`, line 14):**

```systemverilog
out <= out + 32'(a * b);
```

**Offending line (`mac_llm_B.v`, line 15):**

```systemverilog
out <= out + (a * b);
```

**Why it's a problem:** Both files compute `a * b` without using `signed'()` to explicitly assert signed interpretation. This works correctly only because `a` and `b` happen to be declared `logic signed [7:0]`. If the `signed` keyword were ever dropped from the port declarations, the multiplication result would be treated as unsigned, causing zero-extension of negative products instead of sign-extension — a silent math error with no compile warning.

**Corrected version:**

```systemverilog
out <= out + 32'(signed'(a) * signed'(b));
```

`signed'()` enforces signed arithmetic regardless of how the ports are declared.

---

### Issue 2 — Missing explicit width cast in mac_llm_B.v

**Offending line (`mac_llm_B.v`, line 15):**

```systemverilog
out <= out + (a * b);
```

**Why it's a problem:** `a * b` where both operands are `signed [7:0]` produces a 16-bit signed result. There is no explicit width cast before adding it into the 32-bit accumulator `out`. SystemVerilog's implicit width-matching rules will sign-extend the 16-bit product to 32 bits in this context, so it happens to work — but the behavior is implicit and easy to misread. `mac_llm_A.v` uses `32'(a * b)`, which makes the extension explicit and unambiguous.

**Corrected version:**

```systemverilog
out <= out + 32'(signed'(a) * signed'(b));
```

---

### Issue 3 — Reset value style difference

**mac_llm_A.v (line 12):**

```systemverilog
out <= 32'sd0;
```

**mac_llm_B.v (line 13):**

```systemverilog
out <= '0;
```

**Why it matters:** Both are functionally correct — `'0` is a SystemVerilog fill constant that sets all bits to zero. However, `32'sd0` is more explicit: it states the exact width and signedness, making the intent clear and preventing any ambiguity if the port width is later changed. `'0` infers its width from context, which can introduce ambiguity and produce unexpected results in more complex expressions

---

## mac_correct.v — Simulation Log

`iverilog -g2012 -o mac_sim mac_correct.v mac_tb.v && vvp mac_sim`

```
Cycle 1: out=12 (expect 12)
Cycle 2: out=24 (expect 24)
Cycle 3: out=36 (expect 36)
After rst: out=0 (expect 0)
Cycle 5: out=-10 (expect -10)
Cycle 6: out=-20 (expect -20)
mac_tb.v:32: $finish called at 66000 (1ps)
```

All outputs match. ✓

---

## mac_correct.v — Yosys Synthesis

`yosys -p 'read_verilog -sv mac_correct.v; synth; stat'`

/----------------------------------------------------------------------------\
 | yosys -- Yosys Open SYnthesis Suite |
| Copyright (C) 2012 - 2026 Claire Xenia Wolf <claire@yosyshq.com> |
| Distributed under an ISC-like license, type "license" to see terms |
\----------------------------------------------------------------------------/
Yosys 0.64 (git sha1 d8dab5b32666564eca8e18f412973853ce006e61, clang++ 21.0.0 -fPIC -O3)

-- Running command `read_verilog -sv mac_correct.v; synth; stat' --

1. Executing Verilog-2005 frontend: mac_correct.v
   Parsing SystemVerilog input from `mac_correct.v' to AST representation.
Generating RTLIL representation for module `\mac'.
   Successfully finished Verilog frontend.

2. Executing SYNTH pass.

2.1. Executing HIERARCHY pass (managing design hierarchy).

2.2. Executing PROC pass (convert processes to netlists).

2.2.1. Executing PROC_CLEAN pass (remove empty switches from decision trees).
Cleaned up 0 empty switches.

2.2.2. Executing PROC_RMDEAD pass (remove dead branches from decision trees).
Marked 1 switch rules as full_case in process $proc$mac_correct.v:8$1 in module mac.
Removed a total of 0 dead cases.

2.2.3. Executing PROC_PRUNE pass (remove redundant assignments in processes).
Removed 1 redundant assignment.
Promoted 0 assignments to connections.

2.2.4. Executing PROC_INIT pass (extract init attributes).

2.2.5. Executing PROC_ARST pass (detect async resets in processes).

2.2.6. Executing PROC_ROM pass (convert switches to ROMs).
Converted 0 switches.
<suppressed ~1 debug messages>

2.2.7. Executing PROC_MUX pass (convert decision trees to multiplexers).
Creating decoders for process `\mac.$proc$mac_correct.v:8$1'.
1/1: $0\out[31:0]

2.2.8. Executing PROC_DLATCH pass (convert process syncs to latches).

2.2.9. Executing PROC_DFF pass (convert process syncs to FFs).
Creating register for signal `\mac.\out' using process `\mac.$proc$mac_correct.v:8$1'.
  created $dff cell `$procdff$7' with positive edge clock.

2.2.10. Executing PROC_MEMWR pass (convert process memory writes to cells).

2.2.11. Executing PROC_CLEAN pass (remove empty switches from decision trees).
Found and cleaned up 1 empty switch in `\mac.$proc$mac_correct.v:8$1'.
Removing empty process `mac.$proc$mac_correct.v:8$1'.
Cleaned up 1 empty switch.

2.2.12. Executing OPT_EXPR pass (perform const folding).
Optimizing module mac.

2.3. Executing OPT_EXPR pass (perform const folding).
Optimizing module mac.

2.4. Executing OPT_CLEAN pass (remove unused cells and wires).
Finding unused cells or wires in module \mac..
Removed 0 unused cells and 2 unused wires.
<suppressed ~1 debug messages>

2.5. Executing CHECK pass (checking for obvious problems).
Checking module mac...
Found and reported 0 problems.

2.6. Executing OPT pass (performing simple optimizations).

2.6.1. Executing OPT_EXPR pass (perform const folding).
Optimizing module mac.

2.6.2. Executing OPT_MERGE pass (detect identical cells).
Finding identical cells in module `\mac'.
Computing hashes of 4 cells of `\mac'.
Finding duplicate cells in `\mac'.
Removed a total of 0 cells.

2.6.3. Executing OPT_MUXTREE pass (detect dead branches in mux trees).
Running muxtree optimizer on module \mac..
Creating internal representation of mux trees.
Evaluating internal representation of mux trees.
Analyzing evaluation results.
Removed 0 multiplexer ports.
<suppressed ~2 debug messages>

2.6.4. Executing OPT*REDUCE pass (consolidate $\*mux and $reduce*\* inputs).
Optimizing cells in module \mac.
Performed a total of 0 changes.

2.6.5. Executing OPT_MERGE pass (detect identical cells).
Finding identical cells in module `\mac'.
Computing hashes of 4 cells of `\mac'.
Finding duplicate cells in `\mac'.
Removed a total of 0 cells.

2.6.6. Executing OPT_DFF pass (perform DFF optimizations).

2.6.7. Executing OPT_CLEAN pass (remove unused cells and wires).
Finding unused cells or wires in module \mac..

2.6.8. Executing OPT_EXPR pass (perform const folding).
Optimizing module mac.

2.6.9. Finished fast OPT passes. (There is nothing left to do.)

2.7. Executing FSM pass (extract and optimize FSM).

2.7.1. Executing FSM_DETECT pass (finding FSMs in design).

2.7.2. Executing FSM_EXTRACT pass (extracting FSM from design).

2.7.3. Executing FSM_OPT pass (simple optimizations of FSMs).

2.7.4. Executing OPT_CLEAN pass (remove unused cells and wires).
Finding unused cells or wires in module \mac..

2.7.5. Executing FSM_OPT pass (simple optimizations of FSMs).

2.7.6. Executing FSM_RECODE pass (re-assigning FSM state encoding).

2.7.7. Executing FSM_INFO pass (dumping all available information on FSM cells).

2.7.8. Executing FSM_MAP pass (mapping FSMs to basic logic).

2.8. Executing OPT pass (performing simple optimizations).

2.8.1. Executing OPT_EXPR pass (perform const folding).
Optimizing module mac.

2.8.2. Executing OPT_MERGE pass (detect identical cells).
Finding identical cells in module `\mac'.
Computing hashes of 4 cells of `\mac'.
Finding duplicate cells in `\mac'.
Removed a total of 0 cells.

2.8.3. Executing OPT_MUXTREE pass (detect dead branches in mux trees).
Running muxtree optimizer on module \mac..
Creating internal representation of mux trees.
Evaluating internal representation of mux trees.
Analyzing evaluation results.
Removed 0 multiplexer ports.
<suppressed ~2 debug messages>

2.8.4. Executing OPT*REDUCE pass (consolidate $\*mux and $reduce*\* inputs).
Optimizing cells in module \mac.
Performed a total of 0 changes.

2.8.5. Executing OPT_MERGE pass (detect identical cells).
Finding identical cells in module `\mac'.
Computing hashes of 4 cells of `\mac'.
Finding duplicate cells in `\mac'.
Removed a total of 0 cells.

2.8.6. Executing OPT_DFF pass (perform DFF optimizations).
Adding SRST signal on $procdff$7 ($dff) from module mac (D = $add$mac_correct.v:12$3_Y, Q = \out, rval = 0).

2.8.7. Executing OPT_CLEAN pass (remove unused cells and wires).
Finding unused cells or wires in module \mac..
Removed 1 unused cells and 1 unused wires.
<suppressed ~2 debug messages>

2.8.8. Executing OPT_EXPR pass (perform const folding).
Optimizing module mac.

2.8.9. Rerunning OPT passes. (Maybe there is more to do..)

2.8.10. Executing OPT_MUXTREE pass (detect dead branches in mux trees).
Running muxtree optimizer on module \mac..
Creating internal representation of mux trees.
No muxes found in this module.
Removed 0 multiplexer ports.

2.8.11. Executing OPT*REDUCE pass (consolidate $\*mux and $reduce*\* inputs).
Optimizing cells in module \mac.
Performed a total of 0 changes.

2.8.12. Executing OPT_MERGE pass (detect identical cells).
Finding identical cells in module `\mac'.
Computing hashes of 3 cells of `\mac'.
Finding duplicate cells in `\mac'.
Removed a total of 0 cells.

2.8.13. Executing OPT_DFF pass (perform DFF optimizations).

2.8.14. Executing OPT_CLEAN pass (remove unused cells and wires).
Finding unused cells or wires in module \mac..

2.8.15. Executing OPT_EXPR pass (perform const folding).
Optimizing module mac.

2.8.16. Finished fast OPT passes. (There is nothing left to do.)

2.9. Executing WREDUCE pass (reducing word size of cells).
Converting cell mac.$mul$mac_correct.v:12$2 ($mul) from unsigned to signed.
Removed top 24 bits (of 32) from port A of cell mac.$mul$mac_correct.v:12$2 ($mul).
Removed top 24 bits (of 32) from port B of cell mac.$mul$mac_correct.v:12$2 ($mul).
Removed top 16 bits (of 32) from port Y of cell mac.$mul$mac_correct.v:12$2 ($mul).
Removed top 16 bits (of 32) from port B of cell mac.$add$mac_correct.v:12$3 ($add).

2.10. Executing PEEPOPT pass (run peephole optimizers).

2.11. Executing OPT_CLEAN pass (remove unused cells and wires).
Finding unused cells or wires in module \mac..

2.12. Executing ALUMACC pass (create $alu and $macc cells).
Extracting $alu and $macc cells in module mac:
  creating $macc model for $add$mac_correct.v:12$3 ($add).
creating $macc model for $mul$mac_correct.v:12$2 ($mul).
merging $macc model for $mul$mac_correct.v:12$2 into $add$mac_correct.v:12$3.
  creating $macc cell for $add$mac_correct.v:12$3: $auto$alumacc.cc:382:replace_macc$9
created 0 $alu and 1 $macc cells.

2.13. Executing SHARE pass (SAT-based resource sharing).

2.14. Executing OPT pass (performing simple optimizations).

2.14.1. Executing OPT_EXPR pass (perform const folding).
Optimizing module mac.

2.14.2. Executing OPT_MERGE pass (detect identical cells).
Finding identical cells in module `\mac'.
Computing hashes of 3 cells of `\mac'.
Finding duplicate cells in `\mac'.
Removed a total of 0 cells.

2.14.3. Executing OPT_MUXTREE pass (detect dead branches in mux trees).
Running muxtree optimizer on module \mac..
Creating internal representation of mux trees.
No muxes found in this module.
Removed 0 multiplexer ports.

2.14.4. Executing OPT*REDUCE pass (consolidate $\*mux and $reduce*\* inputs).
Optimizing cells in module \mac.
Performed a total of 0 changes.

2.14.5. Executing OPT_MERGE pass (detect identical cells).
Finding identical cells in module `\mac'.
Computing hashes of 3 cells of `\mac'.
Finding duplicate cells in `\mac'.
Removed a total of 0 cells.

2.14.6. Executing OPT_DFF pass (perform DFF optimizations).

2.14.7. Executing OPT_CLEAN pass (remove unused cells and wires).
Finding unused cells or wires in module \mac..
Removed 1 unused cells and 1 unused wires.
<suppressed ~2 debug messages>

2.14.8. Executing OPT_EXPR pass (perform const folding).
Optimizing module mac.

2.14.9. Rerunning OPT passes. (Maybe there is more to do..)

2.14.10. Executing OPT_MUXTREE pass (detect dead branches in mux trees).
Running muxtree optimizer on module \mac..
Creating internal representation of mux trees.
No muxes found in this module.
Removed 0 multiplexer ports.

2.14.11. Executing OPT*REDUCE pass (consolidate $\*mux and $reduce*\* inputs).
Optimizing cells in module \mac.
Performed a total of 0 changes.

2.14.12. Executing OPT_MERGE pass (detect identical cells).
Finding identical cells in module `\mac'.
Computing hashes of 2 cells of `\mac'.
Finding duplicate cells in `\mac'.
Removed a total of 0 cells.

2.14.13. Executing OPT_DFF pass (perform DFF optimizations).

2.14.14. Executing OPT_CLEAN pass (remove unused cells and wires).
Finding unused cells or wires in module \mac..

2.14.15. Executing OPT_EXPR pass (perform const folding).
Optimizing module mac.

2.14.16. Finished fast OPT passes. (There is nothing left to do.)

2.15. Executing MEMORY pass.

2.15.1. Executing OPT_MEM pass (optimize memories).
Performed a total of 0 transformations.

2.15.2. Executing OPT_MEM_PRIORITY pass (removing unnecessary memory write priority relations).
Performed a total of 0 transformations.

2.15.3. Executing OPT_MEM_FEEDBACK pass (finding memory read-to-write feedback paths).

2.15.4. Executing MEMORY_BMUX2ROM pass (converting muxes to ROMs).

2.15.5. Executing MEMORY_DFF pass (merging $dff cells to $memrd).

2.15.6. Executing OPT_CLEAN pass (remove unused cells and wires).
Finding unused cells or wires in module \mac..

2.15.7. Executing MEMORY_SHARE pass (consolidating $memrd/$memwr cells).

2.15.8. Executing OPT_MEM_WIDEN pass (optimize memories where all ports are wide).
Performed a total of 0 transformations.

2.15.9. Executing OPT_CLEAN pass (remove unused cells and wires).
Finding unused cells or wires in module \mac..

2.15.10. Executing MEMORY_COLLECT pass (generating $mem cells).

2.16. Executing OPT_CLEAN pass (remove unused cells and wires).
Finding unused cells or wires in module \mac..

2.17. Executing OPT pass (performing simple optimizations).

2.17.1. Executing OPT_EXPR pass (perform const folding).
Optimizing module mac.

2.17.2. Executing OPT_MERGE pass (detect identical cells).
Finding identical cells in module `\mac'.
Computing hashes of 2 cells of `\mac'.
Finding duplicate cells in `\mac'.
Removed a total of 0 cells.

2.17.3. Executing OPT_DFF pass (perform DFF optimizations).

2.17.4. Executing OPT_CLEAN pass (remove unused cells and wires).
Finding unused cells or wires in module \mac..

2.17.5. Finished fast OPT passes.

2.18. Executing MEMORY_MAP pass (converting memories to logic and flip-flops).

2.19. Executing OPT pass (performing simple optimizations).

2.19.1. Executing OPT_EXPR pass (perform const folding).
Optimizing module mac.

2.19.2. Executing OPT_MERGE pass (detect identical cells).
Finding identical cells in module `\mac'.
Computing hashes of 2 cells of `\mac'.
Finding duplicate cells in `\mac'.
Removed a total of 0 cells.

2.19.3. Executing OPT_MUXTREE pass (detect dead branches in mux trees).
Running muxtree optimizer on module \mac..
Creating internal representation of mux trees.
No muxes found in this module.
Removed 0 multiplexer ports.

2.19.4. Executing OPT*REDUCE pass (consolidate $\*mux and $reduce*\* inputs).
Optimizing cells in module \mac.
Performed a total of 0 changes.

2.19.5. Executing OPT_MERGE pass (detect identical cells).
Finding identical cells in module `\mac'.
Computing hashes of 2 cells of `\mac'.
Finding duplicate cells in `\mac'.
Removed a total of 0 cells.

2.19.6. Executing OPT_SHARE pass.

2.19.7. Executing OPT_DFF pass (perform DFF optimizations).

2.19.8. Executing OPT_CLEAN pass (remove unused cells and wires).
Finding unused cells or wires in module \mac..

2.19.9. Executing OPT_EXPR pass (perform const folding).
Optimizing module mac.

2.19.10. Finished fast OPT passes. (There is nothing left to do.)

2.20. Executing TECHMAP pass (map to technology primitives).

2.20.1. Executing Verilog-2005 frontend: /opt/homebrew/bin/../share/yosys/techmap.v
Parsing Verilog input from `/opt/homebrew/bin/../share/yosys/techmap.v' to AST representation.
Generating RTLIL representation for module `\_90_simplemap_bool_ops'.
Generating RTLIL representation for module `\_90_simplemap_reduce_ops'.
Generating RTLIL representation for module `\_90_simplemap_logic_ops'.
Generating RTLIL representation for module `\_90_simplemap_compare_ops'.
Generating RTLIL representation for module `\_90_simplemap_various'.
Generating RTLIL representation for module `\_90_simplemap_registers'.
Generating RTLIL representation for module `\_90_shift_ops_shr_shl_sshl_sshr'.
Generating RTLIL representation for module `\_90_shift_shiftx'.
Generating RTLIL representation for module `\_90_fa'.
Generating RTLIL representation for module `\_90_lcu_brent_kung'.
Generating RTLIL representation for module `\_90_alu'.
Generating RTLIL representation for module `\_90_macc'.
Generating RTLIL representation for module `\_90_alumacc'.
Generating RTLIL representation for module `$__div_mod_u'.
Generating RTLIL representation for module `$__div_mod_trunc'.
Generating RTLIL representation for module `\_90_div'.
Generating RTLIL representation for module `\_90_mod'.
Generating RTLIL representation for module `$\_\_div_mod_floor'.
Generating RTLIL representation for module `\_90_divfloor'.
Generating RTLIL representation for module `\_90_modfloor'.
Generating RTLIL representation for module `\_90_pow'.
Generating RTLIL representation for module `\_90_pmux'.
Generating RTLIL representation for module `\_90_demux'.
Generating RTLIL representation for module `\_90_lut'.
Generating RTLIL representation for module `$connect'.
Generating RTLIL representation for module `$input_port'.
Successfully finished Verilog frontend.

2.20.2. Continuing TECHMAP pass.
Using extmapper maccmap for cells of type $macc_v2.
  add \a * \b (8x8 bits, signed)
  add \out (32 bits, signed)
Using extmapper simplemap for cells of type $sdff.
Using template $paramod\_90_fa\WIDTH=32'00000000000000000000000000100000 for cells of type $fa.
Using template $paramod$fbc7873bff55778c0b3173955b7e4bce1d9d6834_90_alu for cells of type $alu.
Using extmapper simplemap for cells of type $and.
Using extmapper simplemap for cells of type $not.
Using extmapper simplemap for cells of type $or.
Using extmapper simplemap for cells of type $xor.
Using template $paramod_90_lcu_brent_kung\WIDTH=32'00000000000000000000000000100000 for cells of type $lcu.
Using extmapper simplemap for cells of type $pos.
Using extmapper simplemap for cells of type $mux.
No more expansions possible.
<suppressed ~460 debug messages>

2.21. Executing OPT pass (performing simple optimizations).

2.21.1. Executing OPT_EXPR pass (perform const folding).
Optimizing module mac.
<suppressed ~423 debug messages>

2.21.2. Executing OPT_MERGE pass (detect identical cells).
Finding identical cells in module `\mac'.
Computing hashes of 1569 cells of `\mac'.
Finding duplicate cells in `\mac'.
Computing hashes of 1405 cells of `\mac'.
Finding duplicate cells in `\mac'.
Computing hashes of 1302 cells of `\mac'.
Finding duplicate cells in `\mac'.
Computing hashes of 1228 cells of `\mac'.
Finding duplicate cells in `\mac'.
Computing hashes of 1174 cells of `\mac'.
Finding duplicate cells in `\mac'.
Computing hashes of 1157 cells of `\mac'.
Finding duplicate cells in `\mac'.
Computing hashes of 1125 cells of `\mac'.
Finding duplicate cells in `\mac'.
Computing hashes of 1093 cells of `\mac'.
Finding duplicate cells in `\mac'.
Computing hashes of 1077 cells of `\mac'.
Finding duplicate cells in `\mac'.
Computing hashes of 1047 cells of `\mac'.
Finding duplicate cells in `\mac'.
<suppressed ~1566 debug messages>
Removed a total of 522 cells.

2.21.3. Executing OPT_DFF pass (perform DFF optimizations).

2.21.4. Executing OPT_CLEAN pass (remove unused cells and wires).
Finding unused cells or wires in module \mac..
Removed 57 unused cells and 146 unused wires.
<suppressed ~58 debug messages>

2.21.5. Finished fast OPT passes.

2.22. Executing ABC pass (technology mapping using ABC).

2.22.1. Extracting gate netlist of module `\mac' to `<abc-temp-dir>/input.blif'..

2.22.1.1. Executed ABC.
Extracted 958 gates and 1006 wires to a netlist network with 48 inputs and 32 outputs.
Running ABC script: <abc-temp-dir>/abc.script
ABC: ======== ABC command line "source <abc-temp-dir>/abc.script"
ABC: + read_blif <abc-temp-dir>/input.blif
ABC: + read_library <abc-temp-dir>/stdcells.genlib
ABC: + strash
ABC: + &get -n
ABC: + &fraig -x
ABC: + &put
ABC: + scorr
ABC: Warning: The network is combinational (run "fraig" or "fraig_sweep").
ABC: + dc2
ABC: + dretime
ABC: + strash
ABC: + &get -n
ABC: + &dch -f
ABC: + &nf
ABC: + &put
ABC: + write_blif <abc-temp-dir>/output.blif

2.22.1.2. Re-integrating ABC results.
ABC RESULTS: AND cells: 112
ABC RESULTS: ANDNOT cells: 17
ABC RESULTS: MUX cells: 3
ABC RESULTS: NAND cells: 355
ABC RESULTS: NOR cells: 2
ABC RESULTS: NOT cells: 1
ABC RESULTS: OR cells: 23
ABC RESULTS: ORNOT cells: 31
ABC RESULTS: XNOR cells: 98
ABC RESULTS: XOR cells: 211
ABC RESULTS: internal signals: 926
ABC RESULTS: input signals: 48
ABC RESULTS: output signals: 32
Removing temp directory.
Removing global temp directory.

2.23. Executing OPT pass (performing simple optimizations).

2.23.1. Executing OPT_EXPR pass (perform const folding).
Optimizing module mac.

2.23.2. Executing OPT_MERGE pass (detect identical cells).
Finding identical cells in module `\mac'.
Computing hashes of 885 cells of `\mac'.
Finding duplicate cells in `\mac'.
Removed a total of 0 cells.

2.23.3. Executing OPT_DFF pass (perform DFF optimizations).

2.23.4. Executing OPT_CLEAN pass (remove unused cells and wires).
Finding unused cells or wires in module \mac..
Removed 0 unused cells and 210 unused wires.
<suppressed ~1 debug messages>

2.23.5. Finished fast OPT passes.

2.24. Executing HIERARCHY pass (managing design hierarchy).

2.25. Printing statistics.

=== mac ===

        +----------Local Count, excluding submodules.
        |
      832 wires
     1063 wire bits
        5 public wires
       50 public wire bits
        5 ports
       50 port bits
      885 cells
       17   $_ANDNOT_
      112   $_AND_
        3   $_MUX_
      355   $_NAND_
        2   $_NOR_
        1   $_NOT_
       31   $_ORNOT_
       23   $_OR_
       32   $_SDFF_PP0_
       98   $_XNOR_
      211   $_XOR_

2.26. Executing CHECK pass (checking for obvious problems).
Checking module mac...
Found and reported 0 problems.

3. Printing statistics.

=== mac ===

        +----------Local Count, excluding submodules.
        |
      832 wires
     1063 wire bits
        5 public wires
       50 public wire bits
        5 ports
       50 port bits
      885 cells
       17   $_ANDNOT_
      112   $_AND_
        3   $_MUX_
      355   $_NAND_
        2   $_NOR_
        1   $_NOT_
       31   $_ORNOT_
       23   $_OR_
       32   $_SDFF_PP0_
       98   $_XNOR_
      211   $_XOR_

End of script. Logfile hash: e76fc3667e, time: 0.16s, user: 0.06s, system: 0.01s, MEM: 24.94 MB peak
Yosys 0.64 (git sha1 d8dab5b32666564eca8e18f412973853ce006e61, clang++ 21.0.0 -fPIC -O3)
Time spent: 67% 1x abc (0 sec), 8% 1x techmap (0 sec), ...
