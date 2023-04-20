import sys

sys.path.append("C:/Users/ftheilen/Source/ocelot")
sys.path.append("D:/Fachpraktikum_DESY/GitHub/ocelot-master")

from cheetah import accelerator
from ocelot import Drift, Monitor

# Drift
drift_arlisolg1 = Drift(l=0.19600000000000006, eid="Drift_ARLISOLG1")
drift_arliscrg1 = Drift(l=0.19600000000000006, eid="Drift_ARLISCRG1")
drift_bsc = Drift(l=0.196, eid="Drift_BSC")
drift_scr = Drift(l=0.196, eid="Drift_SCR")
drift_bscr = Drift(l=0.196, eid="Drift_BSCR")

# Monitors
arlibscl1 = Monitor(eid="ARLIBSCL1")
arshscre1 = Monitor(eid="ARSHSCRE1")
bsc = Monitor(eid="BSC")
scr = Monitor(eid="SCR")
nt = Monitor(eid="nt")

# lattice
cell = (drift_bsc, drift_scr, drift_bscr, bsc, scr, nt)

segment = accelerator.Segment.from_ocelot(cell)

print(segment)
