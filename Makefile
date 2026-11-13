.PHONY: test run smoke clean

test:
	PYTHONPATH=src pytest -q

run:
	PYTHONPATH=src python -m cam.pipeline --n-candidates 400 --seed 42 --output-dir outputs

smoke:
	PYTHONPATH=src python -m cam.pipeline --n-candidates 120 --seed 42 --output-dir outputs-smoke

clean:
	rm -rf outputs outputs-smoke .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
