TEX := System_Model_and_Algorithm
BUILD_DIR := build

.PHONY: all clean

all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex
	@mkdir -p $(BUILD_DIR)
	@if command -v latexmk >/dev/null 2>&1; then \
		latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error -outdir=$(BUILD_DIR) $(TEX).tex; \
	elif command -v xelatex >/dev/null 2>&1; then \
		xelatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=$(BUILD_DIR) $(TEX).tex; \
		xelatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=$(BUILD_DIR) $(TEX).tex; \
	else \
		echo "No LaTeX compiler found. Install latexmk with XeLaTeX support or xelatex."; \
		exit 127; \
	fi
	@cp $(BUILD_DIR)/$(TEX).pdf $(TEX).pdf

clean:
	@rm -rf $(BUILD_DIR)
	@rm -f $(TEX).aux $(TEX).fdb_latexmk $(TEX).fls $(TEX).log $(TEX).out $(TEX).xdv
