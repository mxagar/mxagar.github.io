---
layout: post
title:  "Production-Level Code in Python"
date:   2023-12-31 08:00:00 +0000
categories: python programming
permalink: /blog/production-code-python.html
comments: true
---

Contents, topics:

- Python environments: conda, venv, pip-tools, poetry; pin and track depencencies.
- Version control: basic git usage and *fork-change-merge* workflow in teams.
- Docstrings: functions, modules, folow PEP8.
- Exception handling: try-except, logging.
- Linting: pylint, autopep8; follow PEP8.
- [Type hints](https://docs.python.org/3/library/typing.html).
- Dealing with configuration constants: config files, environment variables.
- Beware of the basic data structures and their efficiency in basic operations: lists, tuples, sets, dicts.
- Master Python (*pythonic*) tools, tips and tricks: `zip()`, list comprehensions, scope, by-reference/value...
- Modularization and OOP: use classes, whenever possible; otherwise refactor in reusable functions.
- Executable modules: `"__main__"` + `argparse`.
- Packages: `__init__.py`, `setup.py` or `pyproject.toml`.
- Testing: pytest.
- REST APIs: Flask, FastAPI? Because everything are microservices nowadays.
- Containerization: Docker and Docker-Compose? Because you should be able to ship/deploy your app.
- CI/CD?
- Architecture: SOLID, MVC, DDD, Design Patterns.

Resources:

- [Software engineering tools for Data Scientists](https://github.com/mxagar/data_science_udacity/blob/main/02_SoftwareEngineering/DSND_SWEngineering.md)
- [Producing clean code](https://github.com/mxagar/mlops_udacity/blob/main/01_Clean_Code/MLOpsND_CleanCode.md)
- [Software development tool guides](https://github.com/mxagar/tool_guides)
- [Data Structures and Design Patterns in Python](https://github.com/mxagar/python_software_engineering)

General overview on style and focus:

- Brief introduction to subtopics: what, why.
- Examples for each subtopic + links to resources.
- Focus: Not for ML engineers / Data Scientits, but in general for python developers.
- This is minimum: what I would ask in an interview.

Other resources specific for Data Scientists or Machine Learning Engineers:

- [Guide for Data Scientists](https://github.com/mxagar/data_science_udacity)
- [MLOps Guide](https://github.com/mxagar/mlops_udacity)
- [Cheat Sheet for EDA, Feature Engineering and Basic Data Modeling](https://github.com/mxagar/eda_fe_summary)

<br>

{% if page.comments %} 
{% include disqus-comments.html %}
{% endif %}