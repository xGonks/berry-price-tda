# =============================================================================
#  Berry Price TDA - Makefile
# =============================================================================

.DEFAULT_GOAL := help

RAW_BERRY       = data/raw/WPUSI01102B.csv
INTERIM_BERRY   = data/interim/berry_features.csv

# Help
.PHONY: help
help:
	@echo ""
	@echo "  Comandos disponibles:"
	@echo ""
	@echo "  --- Datos crudos (FRED) ---"
	@echo "    make data          - Descarga / actualiza WPUSI01102B desde FRED"
	@echo "    make data-fresh    - Borra el CSV crudo y re-descarga desde cero"
	@echo ""
	@echo "  --- Variables exógenas + dataset interim ---"
	@echo "    make exog          - Descarga exógenas y construye data/interim/berry_features.csv"
	@echo "    make exog-fresh    - Borra el interim y reconstruye desde cero"
	@echo ""
	@echo "  --- Flujo completo ---"
	@echo "    make all           - data + exog (pipeline completo)"
	@echo "    make all-fresh     - data-fresh + exog-fresh"
	@echo ""
	@echo "  --- Limpieza ---"
	@echo "    make clean         - Elimina CSV crudo"
	@echo "    make clean-interim - Elimina CSV interim"
	@echo "    make clean-all     - Elimina todo lo generado"
	@echo ""

# ---------------------------------------------------------------------------
# Raw data
# ---------------------------------------------------------------------------
.PHONY: data
data:
	python scripts/download_berry_data.py --output $(RAW_BERRY)

.PHONY: data-fresh
data-fresh:
	rm -f $(RAW_BERRY)
	$(MAKE) data

# ---------------------------------------------------------------------------
# Exógenas + interim  (depende de que exista el raw)
# ---------------------------------------------------------------------------
.PHONY: exog
exog: $(RAW_BERRY)
	python scripts/download_exogenous.py --raw $(RAW_BERRY) --output $(INTERIM_BERRY)

# Si el raw no existe aún, lo descarga primero automáticamente
$(RAW_BERRY):
	$(MAKE) data

.PHONY: exog-fresh
exog-fresh:
	rm -f $(INTERIM_BERRY)
	$(MAKE) exog

# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------
.PHONY: all
all: data exog

.PHONY: all-fresh
all-fresh:
	rm -f $(RAW_BERRY) $(INTERIM_BERRY)
	$(MAKE) all

# ---------------------------------------------------------------------------
# Limpieza
# ---------------------------------------------------------------------------
.PHONY: clean
clean:
	rm -f $(RAW_BERRY)

.PHONY: clean-interim
clean-interim:
	rm -f $(INTERIM_BERRY)

.PHONY: clean-all
clean-all: clean clean-interim
