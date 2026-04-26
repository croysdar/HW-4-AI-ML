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
