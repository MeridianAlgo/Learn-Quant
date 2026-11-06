# Interactive Python Learning Platform

<p class="lq-meta">Advanced <span>/</span> AI & Machine Learning <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Learning%20Platform">source</a></p>


## Overview
An all-in-one learning hub that delivers progressive Python lessons through both a guided CLI and a hostable Flask web interface. Lessons combine narrative walkthroughs, executable code examples, mini quizzes, and follow-up practice ideas geared toward aspiring quantitative developers.

## Quickstart (CLI)
```bash
python "UTILS - Learning Platform/learning_platform_cli.py"
```
Features of the CLI experience:
- Step-by-step sections with narration and code snippets
- Inline quizzes with instant explanations
- Practice prompts and follow-up resources at the end of each lesson

## Host the Web Experience
```bash
pip install -r requirements.txt
python "UTILS - Learning Platform/learning_platform_web.py"
```
Then visit `http://127.0.0.1:5000/` in your browser. The web UI mirrors the CLI lessons with collapsible sections and quiz answers surfaced for self-paced study. You can deploy the Flask app on any platform that supports WSGI (Heroku, Railway, Fly.io, Render, etc.).

### Deployment Tips
- Set `FLASK_APP="UTILS - Learning Platform/learning_platform_web.py"`
- Use `flask run --host=0.0.0.0 --port=$PORT` on hosting providers
- Pin dependencies using the root `requirements.txt`

## Lesson Library
Lessons live in `content.py` and can be extended by adding new `Lesson` entries. Each lesson includes:
- Title, difficulty, and estimated completion time
- Objectives, section-by-section prose, and example code
- Optional quizzes (`QuizQuestion`) with explanations
- Practice prompts and follow-up resources

## Related Modules
- `main.py` launcher option **5** runs the CLI directly
- Beginner utilities in `UTILS - Python Basics - Strings/` and `...Numbers/`
- Advanced finance walkthroughs in `Documentation/Programs/level3_financial.py` and `level4_advanced.py`

Happy teaching and learning!

---

## Running it

```bash
cd "Learning Platform"
python "content.py"
```

## More in AI & Machine Learning

- [AI Development](AI Development.md): Command-line chatbots for Google's Gemini API, implemented in both Python and Node.js.
- [Machine Learning - Cross Validation](Machine Learning - Cross Validation.md): A model scored on the data it was trained on is a student grading their own exam.
- [Machine Learning - Feature Engineering](Machine Learning - Feature Engineering.md): The dirty secret of quant machine learning: the model is rarely the bottleneck.
- [Machine Learning - Gradient Descent](Machine Learning - Gradient Descent.md): Gradient descent is the engine inside almost every model that learns.
- [Machine Learning - K Nearest Neighbors](Machine Learning - K Nearest Neighbors.md): Most models squeeze a training set down into a handful of parameters and then discard the data.
- [Machine Learning - K-Means Clustering](Machine Learning - K-Means Clustering.md): Given a few hundred stocks and their return characteristics, which ones behave alike?

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
