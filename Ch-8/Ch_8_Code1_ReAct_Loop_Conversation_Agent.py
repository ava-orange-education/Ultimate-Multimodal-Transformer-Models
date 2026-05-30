!pip install python-dotenv
from dotenv import load_dotenv, find_dotenv
import os, openai, re, httpx
_ = load_dotenv(find_dotenv())  # Automatically finds and loads .env
from openai import OpenAI
client = OpenAI()
class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using GPT-3.5 for demo
            temperature=0,
            messages=self.messages
        )
        return completion.choices[0].message.content
prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer.
Use Thought to describe your thoughts about the question.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.
Your available actions are:
calculate: e.g. calculate: 8.5 * 6 + 12
average_electricity_cost: e.g. average_electricity_cost: Maharashtra
ingredient_quantity: e.g. ingredient_quantity: How much rice needed for 5 people?
Example: Question → Thought → Action → PAUSE → Observation → Answer
""".strip()
def calculate(what):
    try:
        return eval(what)
    except Exception as e:
        return f"Error in calculation: {e}"

def average_electricity_cost(state):
    state = state.strip().lower()
    cost_table = {
        "maharashtra": "₹9.50 per unit",
        "tamil nadu": "₹8.00 per unit",
        "kerala": "₹7.10 per unit",
        "delhi": "₹8.50 per unit",
        "karnataka": "₹8.90 per unit"
    }
    return f"Average electricity cost in {state.title()} is {cost_table.get(state, 
    '₹8.00 per unit')}"
def ingredient_quantity(question):
    import re
    question = question.lower()
    adults = re.search(r'(\d+)\s*adults?', question)
    kids = re.search(r'(\d+)\s*kids?', question)
    adults = int(adults.group(1)) if adults else 0
    kids = int(kids.group(1)) if kids else 0
    if adults + kids > 0:
        total_rice = adults * 90 + kids * 60
        return f"You need approximately {total_rice} grams of rice for {adults} 
        adults and {kids} kids."
    people = re.search(r'(\d+)\s*people', question)
    if people:
        people = int(people.group(1))
        return f"You need around {people * 90} grams of rice for {people} people."
    return "Could not determine people count. Try '2 adults and 3 kids'."
known_actions = {
    "calculate": calculate,
    "average_electricity_cost": average_electricity_cost,
    "ingredient_quantity": ingredient_quantity
}
# Automated ReAct loop executor
action_re = re.compile(r'^Action: (\w+): (.*)$')
def query(question, max_turns=5):
    i = 0
    bot = Agent(prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(result)
        actions = [action_re.match(a) for a in result.split('\n') if  
        action_re.match(a)]
        if actions:
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception(f"Unknown action: {action}: {action_input}")
            print(f" -- running {action} {action_input}")
            observation = known_actions[action](action_input)
            print("Observation:", observation)
            next_prompt = f"Observation: {observation}"
        else:
            return result
question = "I'm cooking lunch for 2 adults and 3 kids. How much rice should I prepare?"
query(question)
Thought: I should check the quantity of rice needed for 2 adults and 3 kids using ingredient_quantity.
Action: ingredient_quantity: How much rice needed for 2 adults and 3 kids?
 -- running ingredient_quantity How much rice needed for 2 adults and 3 kids?
Observation: You need approximately 360 grams of rice for 2 adults and 3 kids.
Answer: You need approximately 360 grams of rice to prepare lunch for 2 adults and 3 kids.
question = "I'm moving from Kerala to tamil nadu. Can you tell me how the electricity cost per unit compares between these two states?"
query(question)

Thought: I need to find the average electricity cost per unit for both Kerala and Tamil Nadu to compare them.
Action: average_electricity_cost: Kerala
PAUSE
 -- running average_electricity_cost Kerala
Observation: Average electricity cost in Kerala is ₹7.10 per unit
Thought: Now I need to find the average electricity cost per unit for Tamil Nadu to complete the comparison.
Action: average_electricity_cost: Tamil Nadu
PAUSE
 -- running average_electricity_cost Tamil Nadu
Observation: Average electricity cost in Tamil Nadu is ₹8.00 per unit
Answer: The average electricity cost in Kerala is ₹7.10 per unit, while in Tamil Nadu it is ₹8.00 per unit. Therefore, electricity is slightly more expensive in Tamil Nadu compared to Kerala.
