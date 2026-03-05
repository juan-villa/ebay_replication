#!/bin/bash
set -e
echo "========================================"
echo "eBay Paid Search Replication Package"
echo "========================================"
echo ""

echo "Step 1: Preprocessing and figures..."
python code/preprocess.py
echo " -> Figures saved to output/figures/"
echo ""

echo "Step 2: DID estimation and tables..."
python code/did_analysis.py
echo " -> Table saved to output/tables/"
echo ""

echo "Step 3: Compiling paper..."
cd paper && pdflatex paper.tex && pdflatex paper.tex && cd ..
echo " -> Paper compiled to paper/paper.pdf"

echo ""
echo "========================================"
echo "Done. Paper is at paper/paper.pdf"
echo "========================================"

