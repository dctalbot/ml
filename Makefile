.venv: requirements.txt
	python -m venv .venv
	. .venv/bin/activate; pip install -r requirements.txt; pip check
	touch .venv
