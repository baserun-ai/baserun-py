import os
import random
import sys

import openai
from openai import OpenAI

import baserun
from baserun.checks import check_includes

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

USER_NAMES = [
    "epeterson",
    "ezhang",
    "cwallace",
    "tsuchanek",
    # Rando names
    "tnguyen",
    "alee",
    "mdavis",
    "jmira",
]

# Requiring research by an agent
AGENT_QUESTIONS = [
    "What are the current top box office movies?",
    "What is the weather forecast for New York City this weekend?",
    "Who is the current CEO of Microsoft?",
    "What are the latest developments in electric car technology?",
    "What was the closing value of the NASDAQ yesterday?",
    "Who are the nominees for the most recent Best Picture Oscar?",
    "What are the latest advancements in quantum computing?",
    "When is the next solar eclipse visible in North America?",
    "What is the current exchange rate between the Euro and the US Dollar?",
]

# Things GPT-4-turbo should know with the 2023 knowledge cutoff
GPT_QUESTIONS = {
    "What was the highest grossing movie of 1997?": "titanic",
    "Has it ever snowed in San Francisco?": "1887",
    "Who is the CEO of Microsoft?": "nadella",
    "What are the latest developments in electric car technology as of March, 2023?": "battery",
    "How much did the NASDAQ index gain over the year 2022?": "composite",
    "Who were the nominees for the 2022 Best Picture Oscar?": "coda",
    "What are the latest advancements in quantum computing as of March, 2023?": "qubits",
    "When was the first solar eclipse recorded?": "2134",
    "Why didn't Britain adopt the Euro currency?": "pound",
}


def main():
    client = OpenAI()
    for i in range(10):
        user = random.choice(USER_NAMES)
        with baserun.start_trace(name="Question and Answer"):
            with baserun.with_session(user_identifier=user):
                for j in range(3):
                    question = random.choice(list(GPT_QUESTIONS.keys()))
                    print(f"{user} asks: {question}")

                    completion = client.chat.completions.create(
                        model="gpt-4-1106-preview",
                        messages=[
                            {
                                "role": "system",
                                "content": "Provide a concise answer to the user's question",
                            },
                            {"role": "user", "content": question},
                        ],
                    )
                    content = completion.choices[0].message.content

                    answer_check = GPT_QUESTIONS[question]
                    check_includes(
                        "openai_chat.content",
                        content,
                        [answer_check],
                        metadata={"answer": content, "question": question},
                    )
                    baserun.log(
                        "OpenAI Chat Results",
                        payload={"answer": content, "question": question},
                    )

                    print(content)
                    print("-----------")


if __name__ == "__main__":
    from dotenv import load_dotenv

    from baserun import Baserun

    load_dotenv()
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    Baserun.init()

    main()
