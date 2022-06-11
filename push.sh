#!/usr/bin/sh
file=collection_data.py
comment=学習データ収集プログラム．ロボットで実行
git add $file
sleep 2
git commit -m "$comment"
sleep 2
git push
