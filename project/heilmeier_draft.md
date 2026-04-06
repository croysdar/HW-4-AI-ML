1. What are you trying to do? Articulate your objectives using absolutely no jargon.

I am designing a custom hardware chip designed specifically to speed up a Binary Neural Network (BNN). My objective is to build a dedicated physical co-processor that is explicitly tailored to the BNN's 1-bit constraints to make it run much faster than it would on a normal computer.

2. How is it done today, and what are the limits of current practice?

Right now, if you run a 1-bit BNN on a standard CPU or GPU, it runs entirely in software. The limit of this current practice is that CPUs and GPUs are general-purpose processors built for complex math. They don't have specialized hardware to handle 1-bit operations efficiently, so running a BNN in software on a normal processor leaves massive amounts of speed and energy savings on the table.

3. What is new in your approach and why do you think it will be successful?

What is new is that I am building the custom hardware chiplet from the ground up to match the BNN's unique constraints. Because the algorithm only uses 1s and 0s, I can build the hardware using simple, fast logic gates like XNOR instead of massive math processors. I think this will be successful because shifting the workload from a slow software baseline to a custom hardware accelerator will show a huge, measurable speedup.
