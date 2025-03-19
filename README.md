# VascularFlow

Simulation of microfluidic flow in deformable vessels.

## Coding conventions

We follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) coding
conventions. Use [`black`](https://github.com/psf/black) to format your code.

In particuler:
* `CamelCase` for classes
* `snake_case` for functions and variables

For file names we use:
* `CamelCase.py` for modules
* `snake_case.py` for scripts and tests

## Tests

Before being able to run tests, you need to execute
```python
pip install -e .[test] 
```
to editably install the code.