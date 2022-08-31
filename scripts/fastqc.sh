#!/usr/bin/bash

echo "Start FastQC for $filename"
echo "--------------------------------"

fastqfolder="$file"
fastqcfolder="${output-}/Quality Assessment/fastQC/$filename"
fastqc="$(dirname "$script_dir")/tools/fastqc/fastqc"

mkdir -p "$fastqcfolder"

$fastqc $fastqfolder/*.fastq -t 2 -o "$fastqcfolder"
