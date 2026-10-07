# 90-second demo script

1. **(10 s) The problem.** "Messy names → handles. I give four examples, no code." Show `examples/normalize_name.json` (note the double space in `Grace  Hopper` → `grace..hopper`: the examples define the behaviour).
2. **(20 s) Synthesize.**
   `traceforge synthesize examples/normalize_name.json --policy uniform --output out/demo`
   Point at: outcome `solved`, the 4-node program `Replace(Lower(Trim(Input(0))), " ", ".")`, candidate count, and the sentence that it is *example-consistent*, not proven correct.
3. **(15 s) Apply.**
   `traceforge apply out/demo/program.json --input examples/people.csv --columns full_name --output out/people_clean.csv`
   Show the new `traceforge_output` / `traceforge_error` columns and that the input file is untouched. Note the quoted `"Hopper, Grace"` row: it produces `hopper,.grace`, which illustrates ambiguity — the examples never showed a comma.
4. **(25 s) Why guidance.** Open `runs/dev/report.html`: solve-rate-vs-candidate-budget curves for uniform vs heuristic/tree/neural; the controls table (untrained/shuffled keep the overhead without the signal).
5. **(20 s) Honesty.** Show the paired-bootstrap table, the joint-solved latency table (guided search is not always faster in wall-clock), and the Limitations section.
