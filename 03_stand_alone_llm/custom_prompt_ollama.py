import os
import streamlit as st
from langchain_community.llms import Ollama

def get_project_directory():
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

def generate_response(query):
    context = (
        "Dummy Survey System (DSS) Overview:\n"
        "- Surveys collect feedback on learning and teaching at University X.\n"
        "- Conducted every semester for each course.\n"
        "- Students receive email links to surveys.\n"
        "- Feedback is anonymous and vital for university improvements.\n\n"
        "Feedback Aspects:\n"
        "- Learning materials, assessments, collaboration, technology, overall satisfaction.\n"
        "- Students comment on best aspects and improvement areas.\n\n"
        "Response Rate Dashboard:\n"
        "- Tracks the number of responses during the survey.\n"
        "- Monitors response rates with filtering options.\n\n"
        "Usage Instructions:\n"
        "1. Access via provided links.\n"
        "2. Log in with your credentials.\n"
        "3. Navigate through filters to view results.\n\n"
        "Key Features:\n"
        "- Displays responses, surveys issued, and response rates.\n"
        "- Updated regularly.\n\n"
        "Dummy Links:\n"
        "- Survey Schedule: [Dummy Survey Schedule](https://example.com/survey-schedule)\n"
        "- Response Rate Dashboard: [Dummy Response Dashboard](https://example.com/response-dashboard)\n"
        "- Survey Dashboard: [Dummy Survey Dashboard](https://example.com/survey-dashboard)\n"
        "- Update Details: [Dummy Update Form](https://example.com/update-form)\n"
        "- Result Reports Request: [Dummy Request Form](https://example.com/request-form)\n\n"
        "Queries:\n"
        "- For any queries, complete this form: [Dummy Query Form](https://example.com/query-form)\n"
    )

    prompt = (
        f"As an AI assistant, your task is to provide a helpful response to the user's query based on the relevant context provided below. Follow these instructions:\n"
        f"1. Carefully review the user's query and understand the specific information they are seeking.\n"
        f"2. Examine the relevant context and identify the information that directly addresses the user's query.\n"
        f"3. Formulate a concise and targeted response that answers the user's query using only the information from the provided context.\n"
        f"4. If the user's query cannot be satisfactorily answered using the provided context, politely inform the user that the required information is not available in the given context.\n"
        f"\nUser Query: {query}\n"
        f"Context:\n{context}\n"
        f"Assistant's Response:"
    )

    print(f"Generated prompt:\n{prompt}")

    try:
        llm = Ollama(model="llama3")
        response = llm.invoke(prompt)
    except Exception as e:
        if "404" in str(e):
            # Try pulling the model
            os.system("ollama pull llama3")
            try:
                llm = Ollama(model="llama3")
                response = llm.invoke(prompt)
            except Exception as inner_e:
                print(f"An error occurred while generating the response after pulling the model: {inner_e}")
                return "An error occurred while generating the response. Please try again later."
        else:
            print(f"An error occurred while generating the response: {e}")
            return "An error occurred while generating the response. Please try again later."

    print(f"Generated response:\n{response}")

    return response

def main():
    st.set_page_config(page_title="Dummy Survey Assistant", page_icon=":memo:")
    st.title("Dummy Survey Assistant")

    project_directory = get_project_directory()
    image_path = os.path.join(project_directory, "00_data", "LOGO.png")
    st.sidebar.image(image_path, use_column_width=True)

    query = st.text_input("Enter your query:")

    if query:
        print(f"User query: {query}")
        response = generate_response(query)
        st.write("Assistant's Response:")
        st.write(response)

if __name__ == "__main__":
    main()

# Run the Streamlit app using the command:
# cd 03_stand_alone_llm
# streamlit run custom_prompt_ollama.py
# ollama run llama3
