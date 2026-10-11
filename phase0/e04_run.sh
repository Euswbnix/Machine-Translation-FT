#!/bin/bash
# E0.4: fine-tune the two halves of the QE top band at lr_scale 1.0, 3 seeds each.
# Pre-registered in PROTOCOL.md, "E0.4". Roughly 1h45m on one RTX 5090.
# The body is phase0/probe_run.sh; the run of record used this script at commit 699e3a2,
# before the body was shared with E0.5 (same commands, same flags).
export PROBE=e04 CONDITIONS=ft_topk_div,ft_topk_rep
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/probe_run.sh"
