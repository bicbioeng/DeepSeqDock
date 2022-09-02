#!/usr/bin/bash

echo "Start FastQC for $filename"
echo "--------------------------------"

fastqfolder="$file"
fastqcfolder="${output-}/Quality Assessment/fastQC/$filename"

mkdir -p "$fastqcfolder"

fastqc $fastqfolder/*.fastq -t 2 -o "$fastqcfolder"
