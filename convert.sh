#!/bin/bash
# go to codelabs
cd codelabs
# get all files
# FILES=find . -name '*.md'
set -e
# convert md to codelabs format
for f in $(find . -name '*.md'); do
  echo "Processing $f file..."
  if [ -f "$f" ]; then
    # mac os, please download
    if [ "$(uname)" == "Darwin" ]; then
      ../bin/claat-darwin-amd64 export -ga G-0JXY6PQLWQ -o ../public $f || echo "next file..."
    # linux (included in the repo)
    elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
      ../bin/claat-linux-amd64 export -ga G-0JXY6PQLWQ -o ../public $f || echo "next file..."
    fi
  fi
done
