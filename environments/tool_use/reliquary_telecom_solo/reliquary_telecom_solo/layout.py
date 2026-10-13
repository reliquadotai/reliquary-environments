"""How train is shared between SFT and RL, under the name reliquary-competitive-
code gives it, so that a corpus job reads either package the same way.

The layout itself lives in `corpus`, beside the split rule it extends.
"""

from reliquary_telecom_solo.corpus import (
    RL_ORDER,
    SFT_ROWS,
    SPLITS,
    USES,
    use_of,
    use_range,
)

__all__ = ["RL_ORDER", "SFT_ROWS", "SPLITS", "USES", "use_of", "use_range"]
