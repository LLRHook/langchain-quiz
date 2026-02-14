#!/usr/bin/env python3
"""Generate multiple-choice quizzes from text using LangChain ChatOllama."""

import argparse
import json
import sys
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class QuizOption(BaseModel):
    text: str = Field(description="The option text")


class QuizQuestion(BaseModel):
    question: str = Field(description="The quiz question")
    options: list[str] = Field(description="Four answer choices", min_length=4, max_length=4)
    correct_index: int = Field(description="Index (0-3) of the correct answer", ge=0, le=3)
    explanation: str = Field(description="Brief explanation of why the correct answer is right")
    sources: list[str] = Field(default_factory=list, description="URLs of web sources that informed this question")


class Quiz(BaseModel):
    questions: list[QuizQuestion] = Field(description="List of quiz questions")


class TopicList(BaseModel):
    topics: list[str] = Field(description="Key topics or claims to search for")


SYSTEM_PROMPT = """You are a quiz generator. Given text content, create multiple-choice questions that test understanding of the key concepts.

Rules:
- Focus on the substantive content. Ignore ads, navigation, promotional material, or boilerplate.
- Each question should have exactly 4 options with one correct answer.
- Questions should range from factual recall to conceptual understanding.
- Explanations should be concise but educational.
- Make wrong options plausible but clearly incorrect to someone who understood the material."""


def build_quiz_prompt(text: str, num_questions: int, difficulty: str) -> str:
    return (
        f"Generate {num_questions} multiple-choice questions at {difficulty} difficulty "
        f"based on the following text:\n\n{text}"
    )


def generate_quiz(
    text: str,
    model: str = "llama3",
    base_url: str = "http://localhost:11434",
    num_questions: int = 5,
    difficulty: str = "medium",
    temperature: float = 0.7,
) -> Quiz:
    llm = ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
    )

    structured_llm = llm.with_structured_output(Quiz)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_quiz_prompt(text, num_questions, difficulty)),
    ]

    return structured_llm.invoke(messages)


def run_interactive_quiz(quiz: Quiz) -> None:
    letters = ["A", "B", "C", "D"]
    score = 0
    total = len(quiz.questions)

    print(f"\n{'=' * 50}")
    print(f"  Quiz: {total} questions")
    print(f"{'=' * 50}\n")

    for i, q in enumerate(quiz.questions):
        print(f"Question {i + 1}/{total}")
        print(f"  {q.question}\n")

        for j, option in enumerate(q.options):
            print(f"    {letters[j]}) {option}")

        while True:
            answer = input("\nYour answer (A/B/C/D): ").strip().upper()
            if answer in letters:
                break
            print("  Invalid input. Enter A, B, C, or D.")

        chosen = letters.index(answer)
        correct = q.correct_index

        if chosen == correct:
            print(f"  Correct! {letters[correct]}) {q.options[correct]}")
            score += 1
        else:
            print(f"  Wrong. You chose {letters[chosen]}) {q.options[chosen]}")
            print(f"  Correct answer: {letters[correct]}) {q.options[correct]}")

        print(f"  Explanation: {q.explanation}\n")
        print("-" * 50)

    pct = round(score / total * 100)
    print(f"\n{'=' * 50}")
    print(f"  Score: {score}/{total} ({pct}%)")
    print(f"{'=' * 50}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate quizzes from text using Ollama")
    parser.add_argument("input", nargs="?", help="Text file to quiz on (reads stdin if omitted)")
    parser.add_argument("-m", "--model", default="llama3", help="Ollama model name (default: llama3)")
    parser.add_argument("-u", "--url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("-n", "--num-questions", type=int, default=5, help="Number of questions (default: 5)")
    parser.add_argument("-d", "--difficulty", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("-t", "--temperature", type=float, default=0.7, help="LLM temperature (default: 0.7)")
    parser.add_argument("--json", action="store_true", help="Output quiz as JSON instead of interactive mode")
    args = parser.parse_args()

    if args.input:
        text = Path(args.input).read_text()
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.error("Provide a text file or pipe text via stdin")

    text = text.strip()
    if len(text) < 50:
        parser.error("Input text is too short to generate a meaningful quiz")

    print(f"Generating {args.num_questions} questions with {args.model}...")

    try:
        quiz = generate_quiz(
            text=text,
            model=args.model,
            base_url=args.url,
            num_questions=args.num_questions,
            difficulty=args.difficulty,
            temperature=args.temperature,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(quiz.model_dump(), indent=2))
    else:
        run_interactive_quiz(quiz)


if __name__ == "__main__":
    main()
