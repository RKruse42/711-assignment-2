#!/bin/bash
# Runs the experiments listed in scripts/jobs.txt, one after the other.
#
#   scripts/jobs.sh start     start (or resume) in the background
#   scripts/jobs.sh pause     let the current job finish, then stop
#   scripts/jobs.sh stop      stop right now (the current job is lost and restarts later)
#   scripts/jobs.sh status    what is running, what is left
#
# Finished jobs are skipped, so it is always safe to start again.
cd "$(dirname "$0")/.." || exit 1
LOG=outputs/logs/run_jobs.log
JOBS=scripts/jobs.txt

# prints "marker|command" for every job that is not done yet
pending_jobs() {
    while IFS='|' read -r marker cmd; do
        marker=$(echo "$marker" | xargs)
        [[ -z "$marker" || "$marker" == \#* ]] && continue
        [[ -f "$marker" ]] && continue
        echo "$marker|$(echo "$cmd" | sed 's/^ *//')"
    done < "$JOBS"
}

run_all() {
    source venv/bin/activate
    rm -f outputs/PAUSE.done
    pending_jobs | while IFS='|' read -r marker cmd; do
        if [[ -f outputs/PAUSE ]]; then
            mv outputs/PAUSE outputs/PAUSE.done
            echo "$(date '+%F %T') PAUSED before: $cmd"
            exit 0
        fi
        name=$(basename "$marker" .json)
        [[ "$marker" == outputs/eval/* ]] && name="${name}_eval"
        echo "$(date '+%F %T') START $cmd"
        $cmd < /dev/null 2>&1 | grep --line-buffered -vE "^W2026|sysfs" > "outputs/logs/$name.log"
        if [[ ! -f "$marker" ]]; then
            echo "$(date '+%F %T') FAILED, no $marker (see outputs/logs/$name.log). Stopping."
            exit 1
        fi
        echo "$(date '+%F %T') DONE  $marker"
    done
    [[ -z "$(pending_jobs)" ]] && echo "$(date '+%F %T') ALL JOBS DONE"
}

case "$1" in
    start)
        mkdir -p outputs/logs
        nohup "$0" run >> "$LOG" 2>&1 &
        echo "started in the background. Follow it with: tail -f $LOG" ;;
    pause)
        touch outputs/PAUSE
        echo "will pause after the current job finishes" ;;
    stop)
        pkill -f "^/bin/bash scripts/jobs.sh run$"
        pkill -f "^python -m improved"
        echo "stopped" ;;
    status)
        if pgrep -f "^python -m improved" > /dev/null; then
            pgrep -fa "^python -m improved" | cut -c1-140
        else
            echo "nothing is running"
        fi
        echo "jobs left: $(pending_jobs | wc -l)"
        tail -n 5 "$LOG" 2>/dev/null ;;
    run)
        run_all ;;
    *)
        echo "usage: scripts/jobs.sh start|pause|stop|status" ;;
esac
