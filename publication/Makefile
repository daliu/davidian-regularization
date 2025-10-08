# Makefile for Stability Bonus Davidian Regularization Publication
# 
# This Makefile provides convenient commands for reproducing all experimental
# results and compiling the publication-ready LaTeX paper.

# Use virtual environment Python
VENV_PATH = ../reg
PYTHON = $(VENV_PATH)/bin/python

.PHONY: all clean experiments figures paper validate help

# Default target
all: experiments figures paper

# Run all experiments and generate results
experiments:
	@echo "Running comprehensive experimental validation..."
	$(PYTHON) scripts/run_full_pipeline.py

# Generate all publication figures
figures:
	@echo "Generating publication-quality figures..."
	$(PYTHON) scripts/generate_figures.py

# Compile LaTeX paper
paper: davidian_regularization.pdf

davidian_regularization.pdf: davidian_regularization.tex references.bib figures
	@echo "Compiling LaTeX paper..."
	pdflatex davidian_regularization.tex
	bibtex davidian_regularization
	pdflatex davidian_regularization.tex
	pdflatex davidian_regularization.tex
	@echo "✅ Paper compiled successfully: davidian_regularization.pdf"

# Validate reproducibility
validate:
	@echo "Validating reproducibility for publication..."
	$(PYTHON) scripts/validate_reproducibility.py

# Quick test of core functionality
test:
	@echo "Running quick functionality test..."
	$(PYTHON) -c "from src.davidian_regularization import stability_bonus_regularization; print('Test:', stability_bonus_regularization(0.85, 0.83))"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -f *.aux *.bbl *.blg *.log *.out *.toc
	rm -f davidian_regularization.pdf
	rm -rf data/synthetic_results/*
	rm -rf data/real_dataset_results/*
	rm -f logs/*.log
	rm -f reproducibility_validation_report.json
	rm -f pipeline_execution_summary.json

# Clean everything including figures
clean-all: clean
	rm -f figures/*.png figures/*.pdf

# Install dependencies
install:
	@echo "Installing required packages..."
	$(VENV_PATH)/bin/pip install numpy pandas matplotlib seaborn scikit-learn

# Create directory structure
setup:
	@echo "Setting up directory structure..."
	mkdir -p data/synthetic_results data/real_dataset_results data/processed
	mkdir -p figures logs
	@echo "✅ Directory structure created"

# Full pipeline from scratch
full-pipeline: clean setup experiments figures paper validate
	@echo "🎉 Full publication pipeline completed successfully!"

# Help target
help:
	@echo "Available targets:"
	@echo "  all           - Run experiments, generate figures, compile paper"
	@echo "  experiments   - Run all experimental validation"
	@echo "  figures       - Generate publication figures"  
	@echo "  paper         - Compile LaTeX paper to PDF"
	@echo "  validate      - Validate reproducibility"
	@echo "  test          - Quick functionality test"
	@echo "  clean         - Clean temporary files"
	@echo "  clean-all     - Clean everything including figures"
	@echo "  install       - Install required packages"
	@echo "  setup         - Create directory structure"
	@echo "  full-pipeline - Complete pipeline from scratch"
	@echo "  help          - Show this help message"
