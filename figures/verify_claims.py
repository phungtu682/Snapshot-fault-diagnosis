"""Consistency check: every number quoted in the new manuscript paragraph must match the data."""
import os
import sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'experiments'))
import placement as P
import simlib
from itertools import combinations

d = np.load(os.path.join(HERE, 'data', 'dmp34.npz'), allow_pickle=True)
BUS = [str(b) for b in d['buses']]
MARG, EV, SEL = d['marg'], d['events'], [int(i) for i in d['sel']]
COV = MARG >= 3
fd = simlib.Feeder(34)
FEAS = {b: sum(1 for c in P.CLASSES if len(fd.bus_phases[b]) >= {'SLG': 1, 'LL': 2, '3PH': 3}[c]) for b in fd.buses}
INFORM = np.array([FEAS[str(e[0])] >= 2 for e in EV])

k = next(j for j, e in enumerate(EV) if str(e[0]) == '836' and float(e[1]) == 50)
row = MARG[k]
covered = [BUS[i] for i in np.where(row >= 3)[0]]
uncovered = [BUS[i] for i in np.where(row < 3)[0]]

print('CLAIM "23 buses ... cover it":', len(covered) == 23, len(covered))
print('CLAIM "margin 12--103":', round(row[row >= 3].min(), 1), '--', round(row[row >= 3].max(), 1))
print('CLAIM "feeder head, stubs and laterals see 0--1.6":',
      round(row[row < 3].min(), 2), '--', round(row[row < 3].max(), 2))
print('CLAIM "272 design events":', len(EV) == 272, len(EV))
print('CLAIM "64 of 272 on single-phase laterals":', int((~INFORM).sum()) == 64, int((~INFORM).sum()))
print('CLAIM "208 class-informative":', int(INFORM.sum()) == 208, int(INFORM.sum()))
print('CLAIM "DMP covers 199":', int(COV[INFORM][:, SEL].any(1).sum()) == 199, int(COV[INFORM][:, SEL].any(1).sum()))
print('CLAIM "5984 three-bus subsets": C(34,3) =', len(list(combinations(range(34), 3))))
best = max(int(COV[INFORM][:, s].any(1).sum()) for s in combinations(range(34), 3))
print('CLAIM "no set covers more": exhaustive best =', best)
print('CLAIM "nine unreachable, all at 800 and stubs":', int((COV[INFORM].sum(1) == 0).sum()) == 9,
      sorted({str(EV[i][0]) for i in np.where(INFORM & (COV.sum(1) == 0))[0]}))
print('  994 = 5984?', len(list(combinations(range(34), 3))))
