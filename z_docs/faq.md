# FAQ and contributing

### Do I need to install Learn-Quant as a package?

No. It is a folder of folders. Install the requirements once and run any
lesson directly.

### Which Python version?

3.9 or later. CI runs 3.11.

### Can I use the code in my own project?

Yes, it is MIT licensed. It is written to be read, not to be fast or battle
tested, so check it before it goes anywhere near real money.

### How do I run the JavaScript lessons?

Install Node.js 18 or later and run `node <file>.js` in the lesson's folder.
They have no dependencies.

### Is this investment advice?

No. It is teaching material.

## Contributing

Bug reports and pull requests are welcome on
[GitHub](https://github.com/MeridianAlgo/Learn-Quant). To change a page on
this site, edit the `README.md` in the lesson's folder. The site is rebuilt
from those on every push to `main`.

A new lesson should have:

1. A `README.md` that explains the idea, gives the formula, shows usage and
   lists the caveats.
2. One focused `.py` (or `.js`) file with a demo under `__main__`.
3. A test in `z_tests/` named `test_<module>.py`.

Before opening a pull request:

```bash
ruff check .
ruff format .
pytest z_tests -q
python docs_builder.py
```
