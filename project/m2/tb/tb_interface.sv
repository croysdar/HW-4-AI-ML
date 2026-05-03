// tb_interface.sv — Self-Checking Testbench for AXI4-Stream Skid Buffer
// ======================================================================

`timescale 1ns/1ps
`default_nettype none

module tb_interface;

    // ── Parameters ────────────────────────────────────────────────────────────
    localparam real CLK_PERIOD = 10.0;
    localparam int  N_WORDS    = 100;

    // ── DUT Ports ─────────────────────────────────────────────────────────────
    logic         aclk;
    logic         aresetn;
    logic         s_axis_tvalid;
    logic         s_axis_tready;
    logic [255:0] s_axis_tdata;
    logic         s_axis_tlast;

    logic         core_valid;
    logic         core_ready;
    logic [255:0] core_data;
    logic         frame_done;

    // ── DUT Instantiation ─────────────────────────────────────────────────────
    axis_interface dut (
        .aclk         (aclk),
        .aresetn      (aresetn),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .s_axis_tdata (s_axis_tdata),
        .s_axis_tlast (s_axis_tlast),
        .core_valid   (core_valid),
        .core_ready   (core_ready),
        .core_data    (core_data),
        .frame_done   (frame_done)
    );

    // ── Clock Generation ──────────────────────────────────────────────────────
    initial aclk = 0;
    always #(CLK_PERIOD/2.0) aclk = ~aclk;

    // ── Reference Model (Queues) and Tracking ─────────────────────────────────
    logic [255:0] ref_queue_data [$];
    logic         ref_queue_last [$];
    
    int words_sent     = 0;
    int words_received = 0;
    int fail_count     = 0;

    // ── Utility: Random 256-bit Vector ────────────────────────────────────────
    function automatic logic [255:0] rand256();
        logic [255:0] v;
        for (int w = 0; w < 8; w++) v[w*32 +: 32] = $urandom();
        return v;
    endfunction

    // ── Concurrent Monitor: Input side (Push to Queue) ────────────────────────
    always @(posedge aclk) begin
        if (aresetn && s_axis_tvalid && s_axis_tready) begin
            ref_queue_data.push_back(s_axis_tdata);
            ref_queue_last.push_back(s_axis_tlast);
            words_sent++;
        end
    end

    // ── Concurrent Monitor: Output side (Pop and Verify) ──────────────────────
    always @(posedge aclk) begin
        if (aresetn && core_valid && core_ready) begin
            logic [255:0] exp_data;
            logic         exp_last;
            
            if (ref_queue_data.size() == 0) begin
                $display("FAIL [Time %0t]: DUT produced data, but reference queue is empty!", $time);
                fail_count++;
            end else begin
                exp_data = ref_queue_data.pop_front();
                exp_last = ref_queue_last.pop_front();
                
                if (core_data !== exp_data || frame_done !== exp_last) begin
                    $display("FAIL [Time %0t]: Mismatch!", $time);
                    $display("  Expected: Data=%h, Last=%b", exp_data, exp_last);
                    $display("  Got     : Data=%h, Last=%b", core_data, frame_done);
                    fail_count++;
                end
            end
            words_received++;
        end
    end

    // ── Main Test Sequence (Driver) ───────────────────────────────────────────
    initial begin
        // Initialize
        aresetn       = 1'b0;
        s_axis_tvalid = 1'b0;
        s_axis_tdata  = '0;
        s_axis_tlast  = 1'b0;
        core_ready    = 1'b1;

        // Reset Pulse (Drive on negedge!)
        @(negedge aclk);
        aresetn = 1'b1;
        @(negedge aclk);

        $display("=================================================");
        $display(" STARTING AXI-STREAM INTERFACE VERIFICATION");
        $display("=================================================");

        // ── Phase 1: High-Speed Pass-Through (No Backpressure) ──────────────
        $display("Phase 1: 50 transactions, core_ready always HIGH");
        for (int i = 0; i < 50; i++) begin
            @(negedge aclk); // <--- FIX: Setup data on the falling edge
            s_axis_tvalid = 1'b1;
            s_axis_tdata  = rand256();
            s_axis_tlast  = (i == 49);
            
            // Wait for handshake to clear on the rising edge
            do begin
                @(posedge aclk);
            end while (!s_axis_tready);
        end
        @(negedge aclk);
        s_axis_tvalid = 1'b0;
        
        // Wait for pipeline to drain
        repeat(5) @(posedge aclk);

        // ── Phase 2: Heavy Backpressure (Testing Skid Buffer) ───────────────
        $display("Phase 2: 50 transactions, core_ready randomly stalls");
        
        fork
            begin : backpressure_driver
                for (int i = 0; i < 50; i++) begin
                    @(negedge aclk); // <--- FIX: Setup data on the falling edge
                    s_axis_tvalid = 1'b1;
                    s_axis_tdata  = rand256();
                    s_axis_tlast  = (i == 49);
                    
                    do begin
                        @(posedge aclk);
                    end while (!s_axis_tready);
                end
                @(negedge aclk);
                s_axis_tvalid = 1'b0;
            end
            
            begin : backpressure_staller
                while (words_sent < 100) begin 
                    @(negedge aclk); // <--- FIX: Toggle backpressure on falling edge
                    core_ready = ($urandom() % 2 == 0); 
                end
                @(negedge aclk);
                core_ready = 1'b1; 
            end
        join

        // Wait for pipeline to drain completely
        repeat(10) @(posedge aclk);

        // ── Final Verification ────────────────────────────────────────────────
        $display("=================================================");
        $display(" Sent    : %0d transactions", words_sent);
        $display(" Received: %0d transactions", words_received);
        
        if (words_sent !== words_received) begin
            $display("FAIL: Sent and Received counts do not match!");
            fail_count++;
        end

        if (ref_queue_data.size() !== 0) begin
            $display("FAIL: Reference queue is not empty at end of test!");
            fail_count++;
        end

        if (fail_count == 0 && words_received == N_WORDS) begin
            $display("VERIFIABLE PASS — All AXI protocols and data matched");
        end else begin
            $display("FAIL — %0d errors detected", fail_count);
        end
        $display("=================================================");

        $finish;
    end

endmodule

`default_nettype wire