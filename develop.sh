#!/bin/bash
# 開発環境
# 1. srcの画面を開く(vi)
# 2. dockerでroscoreを回す
# 3. rosrunのlog画面(x2)
# 4 コマンド用(makeを叩いたりする用の場所)
set -Ceu

tmux select-pane -t 0
tmux attach-session -t 0

