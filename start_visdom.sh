#!/usr/bin/env bash
echo nohup python -m visdom.server -port 8097 --hostname 10.60.1.71 >log-visdom.txt 2>&1 &
nohup python -m visdom.server -port 8097 --hostname 10.60.1.71 >log-visdom.txt 2>&1 &