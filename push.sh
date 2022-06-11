#!/usr/bin/sh
file=collection_data.py
comment=ロボットで実行する学習データ収集プログラム
git add $file
sleep 2
git commit -m "$comment"
sleep 2
git push
