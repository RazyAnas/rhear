#!/usr/bin/env bash
# Live MUSAN download progress. Ctrl-C to stop watching (does NOT stop the download).
SP=/private/tmp/claude-501/-Users-mariyafatima-projectonboard/9740b1c3-cbf2-43bf-805d-45bfb359fbf4/scratchpad/e03
F=$SP/musan.tar.gz
TARGET=11080000000        # 10.32 GB
prev=0; prev_t=$(date +%s)
while true; do
  [ -f "$F" ] || { echo "no file yet"; sleep 5; continue; }
  cur=$(stat -f%z "$F"); now=$(date +%s)
  pct=$(echo "scale=1; 100*$cur/$TARGET" | bc)
  gb=$(echo "scale=2; $cur/1073741824" | bc)
  dt=$((now-prev_t))
  if [ "$dt" -gt 0 ] && [ "$prev" -gt 0 ]; then
    rate=$(echo "scale=2; ($cur-$prev)/$dt/1048576" | bc)
    left=$(echo "scale=0; ($TARGET-$cur)/1048576" | bc)
    if [ "$(echo "$rate > 0.05" | bc)" -eq 1 ]; then
      eta=$(echo "scale=0; $left/$rate/60" | bc); eta="${eta} min"
    else eta="stalled"; fi
  else rate="--"; eta="--"; fi
  bar=$(printf '%.0f' "$(echo "$pct/2.5" | bc -l)")
  filled=$(printf '%*s' "$bar" '' | tr ' ' '#')
  empty=$(printf '%*s' "$((40-bar))" '')
  pgrep -f "curl.*musan" >/dev/null && st="downloading" || st="NOT RUNNING"
  printf "\r  [%s%s] %5s%%  %5s GB / 10.32 GB   %s MB/s   ETA %-10s %s   " \
         "$filled" "$empty" "$pct" "$gb" "$rate" "$eta" "$st"
  prev=$cur; prev_t=$now
  sleep 5
done
