!pip install openai pdfplumber
import openai
import pdfplumber
api_key = input("Enter your OpenAI API key: ")
os.environ["OPENAI_API_KEY"] = api_key
openai.api_key = api_key
#Load PDF and extract text
with pdfplumber.open("yourfile.pdf") as pdf:
    paper_text = ""
    for page in pdf.pages:
        paper_text += page.extract_text()

# Use the OpenAI API to generate a summary with system and user prompts
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are an expert technical writer specializing in summarizing complex research papers clearly and concisely."},
        {"role": "user", "content": "Summarize the key findings and methodology of the paper titled 'Advances in Vision Transformers for Multimodal AI'."}
    ],
    max_tokens=200,
    temperature=0.3,
)
response_text = response['choices'][0]['message']['content']
# Wrap the text to a maximum line width of 80 characters
wrapped_text = textwrap.fill(response_text, width=80)
# Print the nicely formatted paragraph
print(wrapped_text)
