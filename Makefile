dev:
	pip install -r requirements.txt
	pip install -e .
	pre-commit install

check:
	isort -c bin/image
	isort -c bin/image_segmentation
	isort -c bin/text
	isort -c dauphin/
	isort -c preprocess/
	isort -c scripts/
	isort -c setup.py
	black bin/image --check
	black bin/image_segmentation --check
	black bin/text --check
	black dauphin/ --check
	black preprocess/ --check
	black scripts/ --check
	black setup.py --check
	flake8 bin/
	flake8 dauphin/
	flake8 preprocess/
	flake8 scripts/
	flake8 setup.py

clean:
	pip uninstall -y dauphin
	rm -rf dauphin.egg-info pip-wheel-metadata

format:
	isort bin/image
	isort bin/image_segmentation
	isort bin/text
	isort dauphin/
	isort preprocess/
	isort scripts/
	isort setup.py
	black bin/image
	black bin/image_segmentation
	black bin/text
	black dauphin/
	black preprocess/
	black scripts/
	black setup.py
	flake8 bin/
	flake8 dauphin/
	flake8 preprocess/
	flake8 scripts/
	flake8 setup.py
    
.PHONY: dev check clean