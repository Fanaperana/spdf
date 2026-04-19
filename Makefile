# spdf — convenience Makefile
#
# Common tasks:
#   make              # alias for `make build`
#   make build        # release build (self-contained binary with embedded pdfium)
#   make debug        # debug build
#   make install      # install to $(PREFIX)/bin (default: ~/.local/bin)
#   make uninstall    # remove installed binary
#   make test         # run workspace tests
#   make fmt          # cargo fmt
#   make lint         # cargo clippy -D warnings
#   make clean        # cargo clean + remove cached pdfium
#   make run ARGS="…" # run built binary with arguments
#
# Tuning:
#   PREFIX=/usr/local make install     # system-wide install
#   FEATURES="pdfium" make build       # disable bundling (needs system pdfium)
#   CARGO=cross make build             # use a different cargo driver

CARGO    ?= cargo
PREFIX   ?= $(HOME)/.local
BINDIR   ?= $(PREFIX)/bin
BIN      := spdf
TARGET_DIR := target

# Empty => default features (includes bundled-pdfium).
FEATURES ?=
ifeq ($(strip $(FEATURES)),)
  FEATURE_FLAGS :=
else
  FEATURE_FLAGS := --no-default-features --features "$(FEATURES)"
endif

RELEASE_BIN := $(TARGET_DIR)/release/$(BIN)
DEBUG_BIN   := $(TARGET_DIR)/debug/$(BIN)

.PHONY: all build build-ocr debug install install-ocr uninstall test fmt lint clean run help \
        pdfium-download deps-ocr benchmark benchmark-update

all: build

help:
	@awk 'BEGIN{FS=":.*##"} /^[a-zA-Z_-]+:.*##/{printf "  %-18s %s\n",$$1,$$2}' $(MAKEFILE_LIST)

build: ## Release build with embedded pdfium
	$(CARGO) build --release -p spdf-cli $(FEATURE_FLAGS)
	@echo "built $(RELEASE_BIN)"

build-ocr: ## Release build with embedded pdfium + Tesseract OCR (needs libtesseract + libleptonica)
	$(CARGO) build --release -p spdf-cli --features tesseract $(FEATURE_FLAGS)
	@echo "built $(RELEASE_BIN) (with tesseract)"

deps-ocr: ## Install Tesseract system deps on Debian/Ubuntu (needs sudo)
	sudo apt-get install -y libtesseract-dev libleptonica-dev clang \
	                        tesseract-ocr tesseract-ocr-eng

debug: ## Debug build
	$(CARGO) build -p spdf-cli $(FEATURE_FLAGS)
	@echo "built $(DEBUG_BIN)"

install: build ## Install to $(BINDIR) (default: ~/.local/bin)
	@install -d "$(BINDIR)"
	install -m 0755 "$(RELEASE_BIN)" "$(BINDIR)/$(BIN)"
	@echo "installed $(BINDIR)/$(BIN)"
	@case ":$$PATH:" in \
	  *":$(BINDIR):"*) ;; \
	  *) echo "note: $(BINDIR) is not on your PATH" ;; \
	esac

install-ocr: build-ocr ## Install Tesseract-enabled binary to $(BINDIR)
	@install -d "$(BINDIR)"
	install -m 0755 "$(RELEASE_BIN)" "$(BINDIR)/$(BIN)"
	@echo "installed $(BINDIR)/$(BIN) (with tesseract)"
	@case ":$$PATH:" in \
	  *":$(BINDIR):"*) ;; \
	  *) echo "note: $(BINDIR) is not on your PATH" ;; \
	esac

uninstall: ## Remove installed binary
	rm -f "$(BINDIR)/$(BIN)"
	@echo "removed $(BINDIR)/$(BIN)"

test: ## Run workspace tests
	$(CARGO) test --workspace $(FEATURE_FLAGS)

fmt: ## Format code
	$(CARGO) fmt --all

lint: ## Clippy with warnings as errors
	$(CARGO) clippy --workspace --all-targets $(FEATURE_FLAGS) -- -D warnings

clean: ## cargo clean and purge cached pdfium
	$(CARGO) clean
	rm -rf "$(HOME)/.cache/spdf" \
	       "$(HOME)/Library/Caches/spdf" 2>/dev/null || true

run: build ## Run release binary. Use ARGS="parse file.pdf --no-ocr"
	"$(RELEASE_BIN)" $(ARGS)

pdfium-download: ## Fetch pdfium via xtask (only needed when bundling is disabled)
	$(CARGO) run -p xtask -- pdfium-download

benchmark: ## Run full benchmark (spdf vs liteparse vs raw tesseract). Set LITEPARSE_DIR=
	@bash benchmark/run.sh $(LITEPARSE_DIR)

benchmark-update: benchmark ## Run benchmark and refresh README table
	@python3 benchmark/update_readme.py
