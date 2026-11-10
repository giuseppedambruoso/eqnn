PYTHON ?= python

.PHONY: figures supplementary numbers paper arxiv test check dla dla-checks all

figures:          ## every figure and table of the paper, from results/ (about one minute)
	$(PYTHON) analysis/circuit_diagrams.py
	$(PYTHON) analysis/schematics.py
	$(PYTHON) analysis/paper_figures.py

supplementary:    ## every setting that the paper does not report in full (supplementary/README.md, tables, figures)
	$(PYTHON) analysis/supplementary.py

numbers:          ## every number quoted in the text that is not read from a table
	$(PYTHON) analysis/text_numbers.py > paper/numbers.txt

paper:            ## compile paper/paper.pdf
	cd paper && pdflatex -interaction=nonstopmode paper >/dev/null && bibtex paper >/dev/null && \
	pdflatex -interaction=nonstopmode paper >/dev/null && pdflatex -interaction=nonstopmode paper >/dev/null
	@echo paper/paper.pdf

arxiv: paper      ## paper/arxiv.tar.gz: the sources to upload to arXiv (with the compiled bibliography paper.bbl)
	cd paper && tar czf arxiv.tar.gz paper.tex paper.bbl figures
	@echo paper/arxiv.tar.gz

test:             ## checks of the statements of the paper
	$(PYTHON) -m pytest

check:            ## re-run a sample of stored runs and compare them with the stored records
	$(PYTHON) experiments/reproduce_check.py

dla:              ## dynamical Lie algebras and function classes (results/dla.json)
	$(PYTHON) -m eqnn.dla results/dla.json

dla-checks:       ## independent numerical checks of the dimensions (PennyLane lie_closure, numerical ranks; ~20 min)
	$(PYTHON) experiments/dla_checks.py > results/dla_checks.txt
	$(PYTHON) experiments/dla_checks_rest.py >> results/dla_checks.txt

all:              ## recompute every result from scratch (hours), then the figures and the paper
	bash experiments/run_all.sh
	$(MAKE) figures numbers paper
