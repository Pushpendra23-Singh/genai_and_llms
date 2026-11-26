# app.py
from flask import Flask, render_template, request
import os

# NEW import (correct one)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)

# 🔐 API key (DON'T hardcode in real project)
os.environ["OPENAI_API_KEY"] = ""

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.2
)

@app.route("/", methods=["GET", "POST"])
def home():
    response = ""

    if request.method == "POST":
        role = request.form.get("role")
        topic = request.form.get("topic")

        prompt = ChatPromptTemplate.from_template(
            "You are a {role}. Explain the concept of {topic} in one paragraph."
        )

        messages = prompt.format_prompt(role=role, topic=topic).to_messages()

        result = model.invoke(messages)
        response = result.content

    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(debug=True)