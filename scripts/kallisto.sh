#!/usr/bin/env bash

echo
echo "Start Kallisto for $filename"
echo "------------------------------"

fastqfolder="$file"
kallistofolder="${output-}/Quality Assessment/kallisto/$filename"
kallisto="$(dirname "$script_dir")/tools/kallisto/kallisto"
indexfolder="$script_dir"
fastqfiles=`ls -d -1 "$fastqfolder/"*.fastq | paste -sd " " -`

mkdir -p "$kallistofolder"

if [ $(ls $fastqfolder | wc -l) == "1" ]; then
	$kallisto quant -i "$indexfolder/human_index.idx" -o "$kallistofolder" --single -l 200 -s 20 -t 6 $fastqfiles
elif [ $(ls $fastqfolder | wc -l) == "2" ]; then
	$kallisto quant -i "$indexfolder/human_index.idx" -o "$kallistofolder" -t 6 $fastqfiles
else
	echo "Could not locate fastq for accession $filename"
fi

echo "Kallisto quantification for $filename completed"
