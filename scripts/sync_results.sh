#!/bin/bash
# Shares campaign results between machines through the `campaign-results`
# branch of this repository on GitHub.
#
# Every machine only ever pushes ITS OWN files
#   results/<campaign_<q>q | init_comparison>.<machine>.jsonl
# so pushes from different machines never conflict; the files of all the
# other machines are copied into results_paper/imported/, where the
# campaign (to skip jobs already done elsewhere), src.plot_campaign and
# src.progress pick them up.
#
# Usage (from the repository root):
#   scripts/sync_results.sh               # sync once
#   scripts/sync_results.sh --loop 600    # sync every 600 s, forever
# Machine name: $EQNN_MACHINE, default `hostname -s`.
# Local clone of the results branch: $EQNN_RESULTS_REPO, default ../eqnn-campaign-results
set -u
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
MACHINE="${EQNN_MACHINE:-$(hostname -s)}"
RESULTS_REPO="${EQNN_RESULTS_REPO:-$REPO_ROOT/../eqnn-campaign-results}"
REMOTE_URL="$(git remote get-url origin)"
BRANCH=campaign-results

sync_once() {
  if [ ! -d "$RESULTS_REPO/.git" ]; then
    git clone -q --branch "$BRANCH" --single-branch "$REMOTE_URL" "$RESULTS_REPO" || return 1
  fi
  git -C "$RESULTS_REPO" pull -q --rebase || return 1
  mkdir -p "$RESULTS_REPO/results" results_paper/imported
  for stem in campaign_8q campaign_10q campaign_12q init_comparison; do
    local_file="results_paper/${stem}.jsonl"
    [ -f "$local_file" ] || continue
    # Only complete (newline-terminated) lines: the job may be appending
    # to the file right now.
    head -n "$(wc -l < "$local_file")" "$local_file" > "$RESULTS_REPO/results/${stem}.${MACHINE}.jsonl"
  done
  git -C "$RESULTS_REPO" add results
  if ! git -C "$RESULTS_REPO" diff --cached --quiet; then
    git -C "$RESULTS_REPO" commit -q -m "results from ${MACHINE} ($(date '+%Y-%m-%d %H:%M'))"
    for attempt in 1 2 3; do
      git -C "$RESULTS_REPO" push -q && break
      git -C "$RESULTS_REPO" pull -q --rebase
    done
  fi
  for f in "$RESULTS_REPO"/results/*.*.jsonl; do
    [ -e "$f" ] || continue
    case "$(basename "$f")" in *".${MACHINE}.jsonl") continue ;; esac
    cp "$f" results_paper/imported/
  done
  echo "$(date '+%H:%M:%S') synced as ${MACHINE}; imported: $(ls results_paper/imported 2>/dev/null | tr '\n' ' ')"
}

if [ "${1:-}" = "--loop" ]; then
  while true; do sync_once || echo "$(date '+%H:%M:%S') sync failed, retrying later"; sleep "${2:-600}"; done
else
  sync_once
fi
