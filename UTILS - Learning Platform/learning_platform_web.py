"""Flask web interface for the interactive learning platform."""

from __future__ import annotations

from pathlib import Path
from typing import List

from flask import Flask, abort, render_template_string, url_for

from content import LESSONS, Lesson, LessonSection, QuizQuestion, get_lesson_by_slug, list_lessons

app = Flask(__name__)

BASE_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{{ title }}</title>
    <style>
      body { font-family: 'Segoe UI', sans-serif; margin: 2rem; color: #0f172a; }
      header { margin-bottom: 2rem; }
      h1, h2, h3 { color: #1d4ed8; }
      .lesson-card { border: 1px solid #cbd5f5; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; background: #f8fafc; }
      .metadata { font-size: 0.95rem; color: #475569; margin-bottom: 0.5rem; }
      pre { background: #0f172a; color: #f8fafc; padding: 1rem; border-radius: 6px; overflow-x: auto; }
      code { font-family: "Fira Code", monospace; }
      nav a { margin-right: 1rem; }
      .quiz { margin-top: 1.5rem; }
      .quiz ul { list-style: none; padding-left: 0; }
      .quiz li { margin-bottom: 0.5rem; }
      footer { margin-top: 2.5rem; font-size: 0.85rem; color: #64748b; }
      .badge { background: #1d4ed8; color: white; padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.8rem; }
      .objectives, .practice { padding-left: 1.25rem; }
      .objectives li, .practice li { margin-bottom: 0.4rem; }
    </style>
  </head>
  <body>
    <header>
      <h1>{{ header }}</h1>
      <nav>
        <a href="{{ url_for('home') }}">All Lessons</a>
      </nav>
    </header>
    {{ body|safe }}
    <footer>
      <p>Content source: <code>{{ source }}</code></p>
      <p>Run via CLI: <code>python learning_platform_cli.py</code></p>
    </footer>
  </body>
</html>
"""


def _render(template: str, **context: object) -> str:
    return render_template_string(
        BASE_TEMPLATE,
        body=template,
        source=str(Path(__file__).relative_to(Path(__file__).resolve().parent.parent)),
        **context,
    )


@app.get("/")
def home() -> str:
    cards: List[str] = []
    for lesson in list_lessons():
        cards.append(
            """
            <div class="lesson-card">
              <h2><a href="{url}">{title}</a></h2>
              <div class="metadata">
                <span class="badge">{difficulty}</span>
                <span>Estimated: {minutes} min</span>
              </div>
              <p>{summary}</p>
            </div>
            """.format(
                url=url_for("lesson_detail", slug=lesson["slug"]),
                title=lesson["title"],
                difficulty=lesson["difficulty"],
                minutes=lesson["estimated_minutes"],
                summary=lesson["summary"],
            )
        )

    template = """
      <section>
        <p>Select a lesson to view guided sections, quizzes, and practice prompts.</p>
        {cards}
      </section>
    """.format(cards="\n".join(cards))
    return _render(template, title="Learning Platform", header="Interactive Python Learning Platform")


@app.get("/lesson/<slug>")
def lesson_detail(slug: str) -> str:
    lesson = get_lesson_by_slug(slug)
    if lesson is None:
        abort(404)

    sections_html = "\n".join(_render_section(section) for section in lesson.sections)
    objectives_html = """<ul class="objectives">{items}</ul>""".format(
        items="".join(f"<li>{item}</li>" for item in lesson.objectives)
    )
    practice_html = """<ul class="practice">{items}</ul>""".format(
        items="".join(f"<li>{item}</li>" for item in lesson.practice)
    )
    follow_up_html = """<ul class="practice">{items}</ul>""".format(
        items="".join(f"<li>{item}</li>" for item in lesson.follow_up)
    )

    template = """
      <article>
        <h2>{title}</h2>
        <p class="metadata">
          <span class="badge">{difficulty}</span>
          <span>Estimated: {minutes} min</span>
        </p>
        <p>{summary}</p>
        <section>
          <h3>Objectives</h3>
          {objectives}
        </section>
        {sections}
        <section>
          <h3>Practice Prompts</h3>
          {practice}
        </section>
        <section>
          <h3>Suggested Follow-Up</h3>
          {follow_up}
        </section>
      </article>
    """.format(
        title=lesson.title,
        difficulty=lesson.difficulty,
        minutes=lesson.estimated_minutes,
        summary=lesson.summary,
        objectives=objectives_html,
        sections=sections_html,
        practice=practice_html,
        follow_up=follow_up_html,
    )

    return _render(template, title=f"Lesson – {lesson.title}", header=lesson.title)


def _render_section(section: LessonSection) -> str:
    code_block = ""
    if section.code_example:
        code_block = f"<pre><code>{section.code_example}</code></pre>"

    quiz_block = ""
    if section.quiz:
        quiz_items = []
        for idx, question in enumerate(section.quiz, start=1):
            options = "".join(f"<li>{choice}</li>" for choice in question.choices)
            quiz_items.append(
                """
                <div class="quiz">
                  <h4>Quiz {idx}: {prompt}</h4>
                  <ul>{options}</ul>
                  <p><strong>Answer:</strong> {answer}</p>
                  <p>{explanation}</p>
                </div>
                """.format(
                    idx=idx,
                    prompt=question.prompt,
                    options=options,
                    answer=question.choices[question.answer_index],
                    explanation=question.explanation,
                )
            )
        quiz_block = "".join(quiz_items)

    return """
      <section>
        <h3>{title}</h3>
        <p>{body}</p>
        {code}
        {quiz}
      </section>
    """.format(
        title=section.title,
        body=section.body,
        code=code_block,
        quiz=quiz_block,
    )


if __name__ == "__main__":
    app.run(debug=True)
