# FAQ & Contributing

## Frequently asked questions

??? question "Do I need to install Learn-Quant as a package?"
    No. It is a collection of standalone folders. Clone the repo, install the
    requirements once, and run any module directly.

??? question "Which Python version should I use?"
    Python 3.9+ works for every module. CI runs on 3.11.

??? question "Some modules import `scipy` / `scikit-learn` — are those required?"
    They are listed in `requirements.txt`. A handful of modules use them for
    optimisation or ML; the README of each module notes any extra dependency.

??? question "Can I use this code in my own project?"
    Yes — it is MIT licensed. It is written for learning, so audit and adapt
    before putting anything near real capital.

??? question "How do I run the JavaScript modules?"
    Install [Node.js](https://nodejs.org) 18+ and run `node <file>.js` inside
    the module folder.

??? question "Is any of this investment advice?"
    No. Everything here is educational. Markets carry risk; do your own research.

## Contributing

Contributions are very welcome.

- :material-bug-outline: **Found a bug?** Open an
  [issue](https://github.com/MeridianAlgo/Learn-Quant/issues).
- :material-lightbulb-on-outline: **Have a new module or strategy?** Fork the
  repo and open a pull request.
- :material-book-edit-outline: **Improving the docs?** Edit the relevant
  module `README.md` — the site rebuilds automatically.

### Module conventions

A good module folder contains:

1. A `README.md` with **theory → formula → usage → pitfalls**.
2. A single-purpose `.py` implementation with type hints and a runnable
   `__main__` demonstration.
3. A matching test in `z_tests/` (`test_<module>.py`).

Before opening a PR:

```bash
ruff check .
ruff format .
pytest z_tests -q
python docs_builder.py   # regenerate the site
```

Thanks for helping more people learn quant. :material-heart:
