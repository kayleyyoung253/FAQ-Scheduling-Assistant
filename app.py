import json
import openai
import gradio as gr
import os
from groq import Groq

# Load the data from data.json
with open("data.json", "r") as f:
    doctor_data = json.load(f)

# Set your OpenAI API key
openai.api_key = os.getenv("GROQ_API_KEY")  # Ensure this environment variable is set

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
def query_with_groq_and_ai(question):
    """
    Combines Groq API and local data to retrieve relevant information and formats it with AI.
    """
    try:
        # Query Groq API for relevant information
        groq_results = groq_client.query(
            index="doctor_schedule",  # Replace with the correct index if needed
            query={"query": question},
            top_k=3  # Number of relevant results to fetch
        )

        # Combine Groq results into a readable format
        groq_data = "\n".join([f"Result {i + 1}: {item}" for i, item in enumerate(groq_results["results"])])

        # Query local JSON data
        question_lower = question.lower()
        local_responses = []
        for doctor, details in doctor_data["doctors"].items():
            for category, info in details.items():
                if category.lower() in question_lower or any(word in question_lower for word in str(info).lower().split()):
                    local_responses.append(f"Doctor: {doctor}, {category.capitalize()}: {info}")

        local_data = "\n\n".join(local_responses)

        # Combine Groq and local data
        combined_data = f"From Groq:\n{groq_data}\n\nFrom Local Data:\n{local_data}"

        # Format the response using OpenAI
        ai_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant that combines Groq and local data for a user-friendly response."},
                {"role": "user", "content": f"Format this data: {combined_data}"}
            ]
        )
        return ai_response["choices"][0]["message"]["content"]

    except Exception as e:
        return f"An error occurred: {str(e)}"

# Gradio interface
iface = gr.Interface(
    fn=query_with_groq_and_ai,
    inputs="text",
    outputs="text",
    title="Doctor Scheduling Assistant with Groq and AI",
    description="Ask questions about doctor scheduling preferences. Combines Groq API and local data for better answers.",
    examples=[
        "What are the telemedicine hours for Bill?",
        "Can Anna accept new patients?",
        "What are the well check-up ages for Megan?"
    ]
)

# Launch the app
iface.launch()