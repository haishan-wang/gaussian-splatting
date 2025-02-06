#!/bin/env bash
echo '[End time]' $(date)
export end=$(date +%s)
export take=$(( end - start ))
((sec=take%60, take/=60, min=take%60, hrs=take/60))
timestamp=$(printf "%d:%02d:%02d" $hrs $min $sec)
echo Time taken to execute commands is [$timestamp] [${take} mins].