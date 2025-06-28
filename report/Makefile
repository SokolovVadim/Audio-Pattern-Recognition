# Set the main LaTeX file (without extension)
MAIN = report

# Default target
all: $(MAIN).pdf

# Rule to compile PDF
$(MAIN).pdf: $(MAIN).tex
	pdflatex $(MAIN).tex
	pdflatex $(MAIN).tex

# Clean auxiliary files
clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).bbl $(MAIN).blg

# Clean everything including PDF
cleanall: clean
	rm -f $(MAIN).pdf
