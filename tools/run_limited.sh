#!/bin/sh
# Run a command (e.g. dubber --best) next to a desktop session without
# letting it starve the browser or get killed by the OOM killer.
#
# usage: tools/run_limited.sh [-n NEED_GB] [-H HIGH_GB] [-M MAX_GB] CMD ...
#
#   * waits until NEED_GB (default 12) of RAM is available before starting
#   * runs CMD in its own systemd user scope (no root needed):
#       MemoryHigh=HIGH_GB (default 18): above this the kernel reclaims the
#         job's own pages (slows it down) instead of anyone else's
#       MemoryMax=MAX_GB (default 24): a runaway job is stopped inside its
#         own scope, never a random desktop process
#       MemorySwapMax=0: the job never pushes the desktop into swap
#     and at nice 10, so the desktop stays responsive
#   * guard: if the whole system falls below 2 GB available (e.g. the
#     browser grows), the job is frozen (cgroup freezer, nothing lost)
#     and thawed again once 5 GB are available
#   * prints the scope's peak memory at the end
#
# A real RAM reservation (MemoryLow) only works when the parent cgroups
# have one too, which needs root:
#   sudo systemctl set-property user-1000.slice MemoryLow=16G
set -e

NEED=12 HIGH=18 MAX=24
while getopts n:H:M: o; do
    case $o in
    n) NEED=$OPTARG ;;
    H) HIGH=$OPTARG ;;
    M) MAX=$OPTARG ;;
    *) exit 2 ;;
    esac
done
shift $((OPTIND - 1))
[ $# -gt 0 ] || { sed -n '2,6p' "$0"; exit 2; }

avail_gb() { awk '/MemAvailable/ { print int($2 / 1048576) }' /proc/meminfo; }

while [ "$(avail_gb)" -lt "$NEED" ]; do
    echo "run_limited: $(avail_gb) GB available, waiting for $NEED GB" >&2
    sleep 30
done

UNIT=job-$$
FROZEN=0
guard() {
    while sleep 5; do
        a=$(avail_gb)
        if [ "$FROZEN" = 0 ] && [ "$a" -lt 2 ]; then
            systemctl --user freeze "$UNIT.scope" 2>/dev/null &&
                FROZEN=1 && echo "run_limited: only $a GB free, job paused" >&2
        elif [ "$FROZEN" = 1 ] && [ "$a" -ge 5 ]; then
            systemctl --user thaw "$UNIT.scope" 2>/dev/null
            FROZEN=0
            echo "run_limited: $a GB free, job resumed" >&2
        fi
    done
}
guard &
GUARD=$!
trap 'kill $GUARD 2>/dev/null' EXIT

echo "run_limited: scope $UNIT.scope, MemoryHigh=${HIGH}G MemoryMax=${MAX}G," \
     "no swap, $(avail_gb) GB available" >&2
# the scope's peak usage is printed from inside it, before it goes away
st=0
systemd-run --user --scope --quiet --unit="$UNIT" --expand-environment=no \
    -p MemoryHigh="${HIGH}G" -p MemoryMax="${MAX}G" -p MemorySwapMax=0 \
    nice -n 10 sh -c '"$@"; st=$?
        d=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)
        echo "run_limited: peak memory $(($(cat $d/memory.peak) / 1048576)) MB" >&2
        exit $st' sh "$@" || st=$?
exit $st
