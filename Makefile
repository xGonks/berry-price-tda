# =============================================================================
#  Berry Price TDA — Makefile
# =============================================================================

.DEFAULT_GOAL := help

RAW_DATA = data/raw/WPUSI01102B.csv

# Help
.PHONY: help
help:
	@echo ""
	@echo "  Comandos disponibles:"
	@echo "    make data        → Descarga / actualiza el CSV desde FRED"
	@echo "    make data-fresh  → Borra el CSV y re-descarga todo desde cero"
	@echo "    make clean       → Elimina el CSV de datos crudos"
	@echo ""

# Data
.PHONY: data
data:
	python scripts/download_berry_data.py --output $(RAW_DATA)

.PHONY: data-fresh
data-fresh:
	rm -f $(RAW_DATA)
	$(MAKE) data

.PHONY: clean
clean:
	rm -f $(RAW_DATA)
