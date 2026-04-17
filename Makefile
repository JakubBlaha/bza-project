ARCHIVE_NAME = xblaha36-bza
PAPER_DIR = docs
PAPER_SRC = $(PAPER_DIR)/docs.tex
PAPER_PDF = $(PAPER_DIR)/docs.pdf
PRES_DIR = presentation
PRES_SRC = $(PRES_DIR)/presentation.md
PRES_PDF = $(PRES_DIR)/presentation.pdf

.PHONY: all paper presentation pack clean

all: paper presentation

# Compile LaTeX paper
paper:
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode docs.tex
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode docs.tex

# Export Marp presentation to PDF
presentation:
	npx @marp-team/marp-cli $(PRES_SRC) --pdf -o $(PRES_PDF)

# Create submission archive
pack:
	rm -rf $(ARCHIVE_NAME) $(ARCHIVE_NAME).zip
	mkdir -p $(ARCHIVE_NAME)
	cp README.md $(ARCHIVE_NAME)/README.md
	cp -r bza_tool $(ARCHIVE_NAME)/
	cp -r res $(ARCHIVE_NAME)/
	cp -r scripts $(ARCHIVE_NAME)/
	cp -r results $(ARCHIVE_NAME)/
	cp -r notebooks $(ARCHIVE_NAME)/
	cp -r $(PAPER_DIR) $(ARCHIVE_NAME)/
	cp -r $(PRES_DIR) $(ARCHIVE_NAME)/
	# Copy paper PDF and video to root of archive
	cp $(PAPER_PDF) $(ARCHIVE_NAME)/docs.pdf
	cp presentation.mov $(ARCHIVE_NAME)/presentation.mov
	cp pyproject.toml $(ARCHIVE_NAME)/
	cp uv.lock $(ARCHIVE_NAME)/
	# Remove build artifacts and venvs from the archive
	find $(ARCHIVE_NAME) -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	find $(ARCHIVE_NAME) -name '.venv' -type d -exec rm -rf {} + 2>/dev/null || true
	find $(ARCHIVE_NAME) -name '*.aux' -delete 2>/dev/null || true
	find $(ARCHIVE_NAME) -name '*.log' -delete 2>/dev/null || true
	find $(ARCHIVE_NAME) -name '*.out' -delete 2>/dev/null || true
	find $(ARCHIVE_NAME) -name '*.synctex.gz' -delete 2>/dev/null || true
	zip -r $(ARCHIVE_NAME).zip $(ARCHIVE_NAME)
	rm -rf $(ARCHIVE_NAME)
	@echo "Created $(ARCHIVE_NAME).zip"

clean:
	rm -f $(PAPER_DIR)/*.aux $(PAPER_DIR)/*.log $(PAPER_DIR)/*.out $(PAPER_DIR)/*.synctex.gz
	rm -f $(PRES_PDF)
	rm -rf $(ARCHIVE_NAME) $(ARCHIVE_NAME).zip
