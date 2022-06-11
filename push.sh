#!/usr/bin/sh
file=push.sh
comment=git push用のスクリプト
git add $file
sleep 2
git commit -m "$comment"
sleep 2
git push
