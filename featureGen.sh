#!/bin/bash
#SBATCH -A p_csb_meiler
#SBATCH -p production
#SBATCH --mem=150G 
#SBATCH -t 48:00:00

WORKDIR="/dors/meilerlab/data/belle6/GlyPred"
cd $WORKDIR
./trainModel.py elm/PLMDall_clust.fasta elm/*.csv
