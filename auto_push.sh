#!/bin/bash

cd /home/maani/niloo/binn_gnn_repo_ready || exit

git add .

if ! git diff --cached --quiet; then
    git commit -m "Auto update $(date '+%Y-%m-%d %H:%M:%S')"
    git push origin master
fi



