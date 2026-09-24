# campaign-results

Results of the paper campaign (see `CAMPAIGN.md` on branch
`feature/paper-campaign`), shared between machines by
`scripts/sync_results.sh`.

- `results/campaign_<q>q.<machine>.jsonl`: one JSON record per finished job,
  written only by the machine it is named after.
- `data/ising/`: the cached 2D Ising configurations (16x16, 32x32, 64x64)
  used by the campaign, so every machine uses exactly the same data.
