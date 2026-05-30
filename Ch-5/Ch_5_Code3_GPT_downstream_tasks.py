!pip -q install transformers
from transformers import pipeline
translator = pipeline("translation_en_to_fr")
print(translator("I need more time",max_length=25)) 
[{'translation_text': "J'ai besoin de plus de temps"}]
print(translator("We are learning artificial intelligence",max_length=25)) 
[{'translation_text': "Nous apprenons l'intelligence artificielle"}]
print(translator("We are learning Large Language Models",max_length=25)) [{'translation_text': 'Nous apprenons les grands modèles linguistiques'}]
text_SUMZ=pipeline("summarization")
# Define the text to be summarized
text = """Now, when it comes to the hidden layers (which are like the posture, grip, and arm movement adjustments you make before the throw), it's not as direct. You can't see these layers' effects on the dartboard immediately like you can with the final throw. So, to help your friend correct their posture or grip, you'd need to watch their throws, see where the darts land, and then trace back through their arm movement, shoulder level, and grip to find what might be causing the inaccuracies.
In the neural network, the error gradients for the hidden layers are calculated by considering how much each hidden neuron contributed to the error in the output and then making a best guess at how to adjust those hidden neurons to get a better result. It's like saying, "Okay, your shoulder was a bit high, which is likely why your throw went too high. Let's try lowering your shoulder a bit and see if that helps. So, in essence, for the output layer, you're adjusting your aim based on where the dart hit, and for the hidden layers, you're adjusting your stance, grip, and throw based on understanding how those factors might have led to the error in where the dart ended up. """
summary = text_SUMZ(text, max_length=50, min_length=25, do_sample=True) print(summary)
!pip install openai==0.28
import openai, os, textwrap
api_key = input("Enter your OpenAI API key: ")
os.environ["OPENAI_API_KEY"] = api_key 
openai.api_key = api_key
response = openai.ChatCompletion.create(
model="gpt-3.5-turbo",
messages=[{"role": "user", "content": "Based on the text create an imaginary story for 10 year old kid: " + text}],
max_tokens=200,
temperature=0.3)
# Extract the summary from the response
summary = response['choices'][0]['message']['content'].strip()
# Format and print the summary as a paragraph
wrapped_summary = textwrap.fill(summary, width=80)
