<p class="lq-badges"><span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-cat">AI & Machine Learning</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Learning Platform"
    python "content.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Learning%20Platform)

---
# Interactive Python Learning Platform

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

## Continue in AI & Machine Learning

<div class="grid cards" markdown>

-   :material-robot-outline: __[AI Development](AI Development.md)__

    Command-line chatbots for Google's Gemini API, implemented in both Python and Node.js. This module demonstrates how to integrate a hosted large language model into a simple interactive application.

-   :material-robot-outline: __[Machine Learning - Feature Engineering](Machine Learning - Feature Engineering.md)__

    The dirty secret of quant machine learning: the model is rarely the bottleneck.

-   :material-robot-outline: __[Machine Learning - Gradient Descent](Machine Learning - Gradient Descent.md)__

    Gradient descent is the engine inside almost every model that learns. The idea

-   :material-robot-outline: __[Machine Learning - K-Means Clustering](Machine Learning - K-Means Clustering.md)__

    Given a few hundred stocks and their return characteristics, which ones behave

-   :material-robot-outline: __[Machine Learning - Logistic Regression](Machine Learning - Logistic Regression.md)__

    Linear regression predicts a number. **Logistic regression** predicts a

-   :material-robot-outline: __[Machine Learning - Random Forest](Machine Learning - Random Forest.md)__

    This module provides a basic implementation of a Random Forest Predictor for quantitative finance. It uses scikit-learn's `RandomForestRegressor` to predict time series data or returns based on a set of features.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
