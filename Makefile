# =============================================================================
#  Berry Price TDA - Makefile
# =============================================================================

.DEFAULT_GOAL := help

# --- Rutas ---
RAW_BERRY    = data/raw/WPUSI01102B.csv
INTERIM      = data/interim/berry_features.csv
PROCESSED    = data/processed
MODELS       = models

FACTORIAL    = $(PROCESSED)/factorial_long.csv
OPTUNA_TRIALS = $(PROCESSED)/optuna_trials.csv
BEST_MODEL   = $(MODELS)/best_model.joblib

# Subcarpetas separadas para las dos variantes de Optuna (no se pisan)
MODELS_NOEXOG = $(MODELS)/sin_exog
MODELS_EXOG   = $(MODELS)/con_exog
PROC_NOEXOG   = $(PROCESSED)/sin_exog
PROC_EXOG     = $(PROCESSED)/con_exog

# Parametros por defecto (sobreescribibles: make optuna TRIALS=1000)
TRIALS      ?= 500
METRIC      ?= mae
CV_SPLITS   ?= 4
JOBS        ?= -1

.PHONY: help
help:
	@echo ""
	@echo "  Comandos disponibles:"
	@echo ""
	@echo "  --- Datos ---"
	@echo "    make data            - Descarga / actualiza WPUSI01102B desde FRED"
	@echo "    make data-fresh      - Borra el CSV crudo y re-descarga"
	@echo "    make exog            - Descarga exogenas -> data/interim/berry_features.csv"
	@echo "    make exog-fresh      - Reconstruye el interim desde cero"
	@echo ""
	@echo "  --- Modelado ---"
	@echo "    make optuna          - Busca optimo SIN exogenas -> models/sin_exog/"
	@echo "    make optuna-exog     - Busca optimo CON exogenas -> models/con_exog/"
	@echo "    make experiment      - Barrido factorial COMPLETO + analisis estadistico (~20 min)"
	@echo "    make experiment-quick- Factorial reducido + analisis (rapido, para probar)"
	@echo "    make analyze         - Re-analiza un factorial ya generado (sin recomputar)"
	@echo ""
	@echo "  --- Flujos completos ---"
	@echo "    make all             - data + exog (solo datos)"
	@echo "    make pipeline        - data + exog + optuna (datos -> mejor modelo sin exog)"
	@echo "    make full            - data + exog + optuna + optuna-exog + experiment (TODO)"
	@echo ""
	@echo "  --- Parametros (ejemplos) ---"
	@echo "    make optuna TRIALS=1000 METRIC=rmse"
	@echo "    make experiment JOBS=4 CV_SPLITS=5"
	@echo ""
	@echo "  --- Limpieza ---"
	@echo "    make clean           - Elimina CSV crudo"
	@echo "    make clean-interim   - Elimina CSV interim"
	@echo "    make clean-results   - Elimina resultados (processed/)"
	@echo "    make clean-models    - Elimina modelos guardados"
	@echo "    make clean-all       - Elimina todo lo generado"
	@echo ""

# =============================================================================
#  Datos
# =============================================================================
.PHONY: data
data:
	python scripts/download_berry_data.py --output $(RAW_BERRY)

.PHONY: data-fresh
data-fresh:
	rm -f $(RAW_BERRY)
	$(MAKE) data

$(RAW_BERRY):
	$(MAKE) data

.PHONY: exog
exog: $(RAW_BERRY)
	python scripts/download_exogenous.py --raw $(RAW_BERRY) --output $(INTERIM)

$(INTERIM):
	$(MAKE) exog

.PHONY: exog-fresh
exog-fresh:
	rm -f $(INTERIM)
	$(MAKE) exog

# =============================================================================
#  Modelado
# =============================================================================

# Optuna: busca el optimo rapido y GUARDA el mejor modelo (variante SIN exogenas)
.PHONY: optuna
optuna: $(INTERIM)
	python scripts/run_optuna.py \
		--data $(INTERIM) \
		--trials $(TRIALS) \
		--metric $(METRIC) \
		--cv-splits $(CV_SPLITS) \
		--jobs $(JOBS) \
		--out-dir $(PROC_NOEXOG) \
		--model-dir $(MODELS_NOEXOG)

# Optuna permitiendo exogenas en la busqueda (variante CON exogenas)
.PHONY: optuna-exog
optuna-exog: $(INTERIM)
	python scripts/run_optuna.py \
		--data $(INTERIM) \
		--trials $(TRIALS) \
		--metric $(METRIC) \
		--cv-splits $(CV_SPLITS) \
		--jobs $(JOBS) \
		--with-exog \
		--out-dir $(PROC_EXOG) \
		--model-dir $(MODELS_EXOG)

# Barrido factorial COMPLETO + analisis estadistico (lento, ~20 min)
.PHONY: experiment
experiment: $(INTERIM)
	python scripts/run_experiment.py \
		--data $(INTERIM) \
		--metric $(METRIC) \
		--cv-splits $(CV_SPLITS) \
		--jobs $(JOBS) \
		--checkpoint $(FACTORIAL) \
		--out-dir $(PROCESSED)

# Version rapida del factorial (subconjunto) para probar
.PHONY: experiment-quick
experiment-quick: $(INTERIM)
	python scripts/run_experiment.py \
		--data $(INTERIM) \
		--metric $(METRIC) \
		--cv-splits $(CV_SPLITS) \
		--jobs $(JOBS) \
		--checkpoint $(FACTORIAL) \
		--out-dir $(PROCESSED) \
		--quick

# Re-analizar un factorial ya generado (sin recomputar el barrido)
.PHONY: analyze
analyze:
	python scripts/run_experiment.py \
		--metric $(METRIC) \
		--analyze-only $(FACTORIAL) \
		--out-dir $(PROCESSED)

# =============================================================================
#  Flujos completos
# =============================================================================
.PHONY: all
all: data exog

.PHONY: pipeline
pipeline: data exog optuna

.PHONY: full
full: data exog optuna optuna-exog experiment

# =============================================================================
#  Limpieza
# =============================================================================
.PHONY: clean
clean:
	rm -f $(RAW_BERRY)

.PHONY: clean-interim
clean-interim:
	rm -f $(INTERIM)

.PHONY: clean-results
clean-results:
	rm -f $(PROCESSED)/*.csv

.PHONY: clean-models
clean-models:
	rm -rf $(MODELS_NOEXOG) $(MODELS_EXOG)
	rm -f $(MODELS)/best_model.joblib $(MODELS)/best_config.json

.PHONY: clean-all
clean-all: clean clean-interim clean-results clean-models
