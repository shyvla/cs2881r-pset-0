(.venv) shylan@Shylas-Macbook-Air src % python analyze.py --dataset gsm8k
/Users/shylan/VSCode/cs2881r-pset-0/src/runs/gsm8k_n150_light.jsonl
cells present: ['cot_ablated', 'cot_intact', 'cot_random', 'direct_ablated', 'direct_intact', 'direct_random']
dataset: gsm8k (assumed from --dataset; records carry no stamp)   split rows 1319 (from /Users/shylan/VSCode/cs2881r-pset-0/src/runs/gsm8k_n150_light_pin.json)

cell                 n    acc  norm   composition
cot_intact         150  80.7%     0   correct=121  incomplete=24  incorrect=5
cot_ablated        150  74.7%     0   correct=112  incomplete=30  incorrect=8
cot_random         150  76.7%     0   correct=115  incomplete=28  incorrect=7
direct_intact      150  27.3%     0   incorrect=109  correct=41
direct_ablated     150  24.7%     0   incorrect=113  correct=37
direct_random      150  24.7%     0   incorrect=113  correct=37

COT ARM   150 paired problems
   ablated    74.7%   drop +6.0 pts   McNemar vs intact b=12  c=3  p=0.03515625
   random     76.7%   drop +4.0 pts   McNemar vs intact b=10  c=4  p=0.1795654296875
   SELECTIVITY = (intact-ablated) - (intact-random) = random - ablated
   point +2.0 pts   95% CI [-3.3, +7.3]   p=0.532   width 10.7 pts
   CI includes zero -- no selectivity detected

DIRECT ARM   150 paired problems
   ablated    24.7%   drop +2.7 pts   McNemar vs intact b=11  c=7  p=0.480682373046875
   random     24.7%   drop +2.7 pts   McNemar vs intact b=5  c=1  p=0.21875
   SELECTIVITY = (intact-ablated) - (intact-random) = random - ablated
   point +0.0 pts   95% CI [-5.3, +5.3]   p=1.000   width 10.7 pts
   CI includes zero -- no selectivity detected

INTERACTION   150 problems in both arms
   ablation        -3.3 pts   95% CI [-10.7, +4.0]   p=0.407   width 14.7
   random control  -1.3 pts   95% CI [-7.3, +4.7]   p=0.678   width 12.0
   A control interaction that is not ~0 means broad degradation, not
   a J-space effect. Read the two lines together or neither.

EXPLORATORY SUBSET (config.problem_ids(12), the set the band was chosen on)
   Reported apart, never pooled: selection on outcome data inflates whatever
   it selected for, and this subset is measurably not representative.
   cot_intact    explored    83% (12)    holdout    80% (138)
   cot_ablated   explored    83% (12)    holdout    74% (138)
   cot_random    explored    83% (12)    holdout    76% (138)
   direct_intact    explored    50% (12)    holdout    25% (138)
   direct_ablated   explored    33% (12)    holdout    24% (138)
   direct_random    explored    42% (12)    holdout    23% (138)

---

(.venv) shylan@Shylas-Macbook-Air src % python analyze.py --dataset math500
/Users/shylan/VSCode/cs2881r-pset-0/src/runs/math500_n150_light.jsonl
cells present: ['cot_intact', 'cot_random', 'direct_ablated', 'direct_intact', 'direct_random']
dataset: math500 (assumed from --dataset; records carry no stamp)   split rows 500 (from /Users/shylan/VSCode/cs2881r-pset-0/src/runs/math500_n150_light_pin.json)

cell                 n    acc  norm   composition
cot_intact         150  94.0%     0   correct=141  incomplete=6  incorrect=3
cot_random         150  96.0%     0   correct=144  incorrect=3  incomplete=3
direct_intact      150  34.0%     0   incorrect=99  correct=51
direct_ablated     150  30.7%     0   incorrect=104  correct=46
direct_random      150  28.7%     0   incorrect=107  correct=43

COT ARM   150 paired problems
   random     96.0%   drop -2.0 pts   McNemar vs intact b=2  c=5  p=0.453125

DIRECT ARM   150 paired problems
   ablated    30.7%   drop +3.3 pts   McNemar vs intact b=8  c=3  p=0.2265625
   random     28.7%   drop +5.3 pts   McNemar vs intact b=8  c=0  p=0.0078125
   SELECTIVITY = (intact-ablated) - (intact-random) = random - ablated
   point -2.0 pts   95% CI [-6.7, +2.7]   p=0.487   width 9.3 pts
   CI includes zero -- no selectivity detected

INTERACTION   150 problems in both arms
   random control  +7.3 pts   95% CI [+2.7, +12.7]   p=0.003   width 10.0
   A control interaction that is not ~0 means broad degradation, not
   a J-space effect. Read the two lines together or neither.

EXPLORATORY SUBSET: not applicable. The band was selected on gsm8k
   (config.EXPLORED_ON), so every math500 problem here is confirmatory and the
   whole sample above is the holdout. Note in the report that the band was
   nonetheless CHOSEN on gsm8k and transferred, which is an assumption,
   not a measurement on this dataset.

---

(.venv) shylan@Shylas-Macbook-Air src % python analyze.py --dataset aime24
/Users/shylan/VSCode/cs2881r-pset-0/src/runs/aime24_n30_light.jsonl
cells present: ['cot_ablated', 'cot_intact', 'cot_random', 'direct_ablated', 'direct_intact', 'direct_random']
dataset: aime24 (assumed from --dataset; records carry no stamp)   split rows 30 (from /Users/shylan/VSCode/cs2881r-pset-0/src/runs/aime24_n30_light_pin.json)

cell                 n    acc  norm   composition
cot_intact          30  66.7%     0   correct=20  incorrect=5  incomplete=5
cot_ablated         20  55.0%     0   correct=11  incomplete=8  incorrect=1
cot_random          30  73.3%     0   correct=22  incorrect=7  incomplete=1
direct_intact       30   0.0%     0   incorrect=30
direct_ablated      30   0.0%     0   incorrect=30
direct_random       30   0.0%     0   incorrect=30

COT ARM   20 paired problems
   ablated    55.0%   drop +20.0 pts   McNemar vs intact b=5  c=1  p=0.21875
   random     75.0%   drop +0.0 pts   McNemar vs intact b=1  c=1  p=1.0
   SELECTIVITY = (intact-ablated) - (intact-random) = random - ablated
   point +20.0 pts   95% CI [+0.0, +40.0]   p=0.115   width 40.0 pts
   CI includes zero -- no selectivity detected
   CI wider than 25 pts: this cannot distinguish 'no effect' from
   'not enough problems'. Do not read it as a null.

DIRECT ARM   30 paired problems
   ablated     0.0%   drop +0.0 pts   McNemar vs intact b=0  c=0  p=1.0
   random      0.0%   drop +0.0 pts   McNemar vs intact b=0  c=0  p=1.0
   SELECTIVITY = (intact-ablated) - (intact-random) = random - ablated
   point +0.0 pts   95% CI [+0.0, +0.0]   p=1.000   width 0.0 pts
   CI includes zero -- no selectivity detected

INTERACTION   20 problems in both arms
   ablation        -20.0 pts   95% CI [-40.0, +0.0]   p=0.113   width 40.0
   random control  +0.0 pts   95% CI [-15.0, +15.0]   p=1.000   width 30.0
   A control interaction that is not ~0 means broad degradation, not
   a J-space effect. Read the two lines together or neither.

EXPLORATORY SUBSET: not applicable. The band was selected on gsm8k
   (config.EXPLORED_ON), so every aime24 problem here is confirmatory and the
   whole sample above is the holdout. Note in the report that the band was
   nonetheless CHOSEN on gsm8k and transferred, which is an assumption,
   not a measurement on this dataset.

