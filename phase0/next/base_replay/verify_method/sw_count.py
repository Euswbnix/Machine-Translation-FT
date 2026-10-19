import random, sys
import swanlab  # import first, as train.py does (import precedes set_seed)
from swanlab.data.run.metadata.hardware import get_hardware_info
from swanlab.data.run.metadata import get_metadata
def pos():
    return random.getstate()[1][-1]
random.seed(42); p0 = pos()
info, mon = get_hardware_info()
print("after get_hardware_info: MT index", pos(), "(624 = untouched after seed; n<624 = n words drawn)", "collectors", len(mon), [type(m).__name__ for m in mon])
random.seed(42)
md, mon2 = get_metadata()
print("after get_metadata: MT index", pos(), "monitor", md.get("swanlab", {}).get("_monitor"))
# second call of collectors (as a periodic monitor would): do collect() calls draw?
random.seed(42); before = pos()
for m in mon2:
    try:
        m()
    except Exception as e:
        pass
print("after one round of collector calls:", pos())
