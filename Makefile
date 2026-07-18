.PHONY: help venv install install-data install-train install-eval install-quantize install-serve install-all smoke clean

help:
	@echo "llama-tools — common commands"
	@echo ""
	@echo "  make venv              Create Python venv (.venv/)"
	@echo "  make install           Install base deps (Week 1)"
	@echo "  make install-data      Install data curation deps (Week 3)"
	@echo "  make install-train     Install training deps (Weeks 2, 4, 6, 8)"
	@echo "  make install-eval      Install evaluation deps (Week 7)"
	@echo "  make install-quantize  Install quantization deps (Week 9)"
	@echo "  make install-serve     Install serving deps (Week 11)"
	@echo "  make install-all       Install all optional groups"
	@echo ""
	@echo "  make smoke             Run Week 1 smoke test (Llama-3.2-1B on CPU)"
	@echo ""
	@echo "  make clean             Remove .venv and Python caches"

venv:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip

install: venv
	.venv/bin/pip install -e .

install-data:
	.venv/bin/pip install -e ".[data]"

install-train:
	.venv/bin/pip install -e ".[train]"

install-eval:
	.venv/bin/pip install -e ".[eval]"

install-quantize:
	.venv/bin/pip install -e ".[quantize]"

install-serve:
	.venv/bin/pip install -e ".[serve]"

install-all:
	.venv/bin/pip install -e ".[data,train,eval,quantize,serve,dev]"

smoke:
	.venv/bin/python smoke.py

clean:
	rm -rf .venv __pycache__ .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
