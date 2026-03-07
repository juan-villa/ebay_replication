If I edit code/preprocess.py, Make will rebuild output/figures/figure_5_2.png and output/figures/figure_5_3.png because they depend on preprocess.py. Since those figures are inputs into paper/paper.pdf, Make also rebuilds paper/paper.pdf. It will skip output/tables/did_table.tex because that file does not depend on preprocess.py.


If I edit code/did_analysis.py, Make rebuilds output/tables/did_table.tex because it depends on did_analysis.py. Since that table is an input into paper/paper.pdf, Make also rebuilds paper/paper.pdf. It will skip output/figures/figure_5_2.png and output/figures/figure_5_3.png because they do not depend on did_analysis.py.


If I edit paper/paper.tex, Make rebuilds paper/paper.pdf because the PDF depends directly on paper.tex. It will skip output/figures/figure_5_2.png, output/figures/figure_5_3.png, and output/tables/did_table.tex because none of those files depend on paper.tex.


The Makefile makes the dependency structure of the project explicit, while run_all.sh only specifies a sequence of commands. This makes it clear which output files depend on which input files and scripts, and Make automatically knows which steps need to rerun when something changes. In contrast, run_all.sh reruns everything every time, even if only one file was modified. For any new collaborator, the Makefile documents the workflow as a directed graph of dependencies, making it easier to understand how the files are related.

