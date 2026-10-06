#!/usr/bin/env bash
# Run before and after every live step on the sandbox test box (<sandbox-test-host>).
# It only reads: never attach to, send keys to or kill tmux `tmaxval`, and never touch
# the main Docker daemon (a read-only `docker ps` names its vf-* containers).
# Exit non-zero = stop and tell the user.
#
#   GUARD_SINCE   marker file whose mtime is the reference time for TMax progress
#                 (default /opt/rq-plan2/guard.since, created on the first run)
#   GUARD_MIN_FREE_GB   minimum free GB on / (default 15)
set -euo pipefail
SINCE=${GUARD_SINCE:-/opt/rq-plan2/guard.since}
MIN_FREE_GB=${GUARD_MIN_FREE_GB:-15}
TMAX_OUT=${GUARD_TMAX_OUT:-/root/tmax/validation}
STALL_S=$((45 * 60))

flags=$(ip -o link show docker0 2>/dev/null) || { echo "GUARD: docker0 is MISSING"; exit 1; }
case "$flags" in *"<"*UP*">"*) echo "GUARD: docker0 present and UP" ;;
  *) echo "GUARD: docker0 is not UP: $flags"; exit 1 ;; esac
tmux has-session -t tmaxval 2>/dev/null || { echo "GUARD: tmaxval is GONE"; exit 1; }
echo "GUARD: tmaxval present"
avail=$(free -m | awk '/^Mem:/ {print $7}')
echo "GUARD: ${avail} MB available"
[ "$avail" -ge 2500 ] || { echo "GUARD: under 2500 MB available: wait"; exit 1; }
free_gb=$(df -BG --output=avail / | tail -n 1 | tr -dc 0-9)
echo "GUARD: ${free_gb} GB free on /"
[ "$free_gb" -ge "$MIN_FREE_GB" ] || { echo "GUARD: under ${MIN_FREE_GB} GB free on /"; exit 1; }
mkdir -p "$(dirname "$SINCE")"
[ -e "$SINCE" ] || touch "$SINCE"
total=$(find "$TMAX_OUT" -maxdepth 1 -type f | wc -l)
new=$(find "$TMAX_OUT" -maxdepth 1 -type f -newer "$SINCE" | wc -l)
newest=$(find "$TMAX_OUT" -maxdepth 1 -type f -printf '%T@\n' | sort -n | tail -n 1)
age=$(( $(date +%s) - ${newest%.*} ))
vf=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -c '^vf-' || true)
echo "GUARD: tmax files ${total} (${new} new since $(date -r "$SINCE" -u +%H:%M:%SZ)), newest ${age}s ago, ${vf} vf-* containers"
if [ "$age" -ge "$STALL_S" ] && [ "$vf" -eq 0 ]; then
  echo "GUARD: TMax stalled (no new file for ${age}s and no vf-* container)"; exit 1
fi
echo "GUARD: OK"
