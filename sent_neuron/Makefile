TEST_CMD = python3 -m unittest
CHECKSTYLE_CMD = flake8

all: compile checkstyle  # test

compile:
	@echo "Nothing to compile for Python"

test:
	$(TEST_CMD) tests/*.py

checkstyle:
	$(CHECKSTYLE_CMD) *.py

clean:
	rm -f *.pyc
	rm -rf __pycache__
	rm -f tests/*.pyc
	rm -rf tests/__pycache__

