import os
KAGGLE_AGENT_PATH = "/kaggle_simulations/agent/"

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
# from unsloth import FastLanguageModel
import torch
from kaggle_secrets import UserSecretsClient
import sys
import shutil
import re
import csv
import random
import threading
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(False)

KAGGLE_AGENT_PATH = "/kaggle_simulations/agent/"
if os.path.exists(KAGGLE_AGENT_PATH):
    model_id_31 = os.path.join(KAGGLE_AGENT_PATH, "model")
    model_id_31_8bit = os.path.join(KAGGLE_AGENT_PATH, "model_31_8bit")
    model_id_v8 = KAGGLE_AGENT_PATH
    model_id_qwen = os.path.join(KAGGLE_AGENT_PATH, "model_qwen")
    keyword_path = os.path.join(KAGGLE_AGENT_PATH, "test.csv")
else:
    model_id = "/kaggle/input/llama-3/transformers/8b-chat-hf/1"
    model_id = "UCLA-AGI/Llama-3-Instruct-8B-SPPO-Iter3"
    model_id = "PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct"
    model_id = "MaziyarPanahi/Llama-3-8B-Instruct-v0.8"
    model_id = '/kaggle/input/maziyarv08/model'
    model_id_v8 = '/kaggle/input/llama-3-maziyarv08/model'
#     model_id_v10 = '/kaggle/input/llama-3-maziyarv10/model'
    model_id_31  = '/kaggle/input/llama-331/model'
    model_id_31_8bit = '/kaggle/input/llama-331-8-bit/model_31_8bit'
    model_id_vg_31  = '/kaggle/input/llama-vg331/model_vg31_4bit'
    model_id_vg_31_8bit = '/kaggle/input/llama-vg331-8-bit/model_vg31_8bit'
    model_id_qwen = '/kaggle/input/qwen-4bits/model_qwen'
    keyword_path = '/kaggle/input/test-data/test.csv'


import asyncio
USE_PIPE = False

def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id,
    #                                              max_length=8192,
    #                                              low_cpu_mem_usage=True,
                                                 device_map="auto",
                                                )
    return tokenizer, model

def generate_answer(model_xx, tokenizer_xx, id_eot_xx, template, max_new_tokens=200, temperature=0.8, **model_kwarg):
    inp_ids = tokenizer_xx(template, return_tensors="pt").to("cuda")
    out_ids = model_xx.generate(**inp_ids,max_new_tokens=max_new_tokens, temperature=temperature, **model_kwarg).squeeze()
    start_gen = inp_ids.input_ids.shape[1]
    out_ids = out_ids[start_gen:]
    if id_eot_xx in out_ids:
        stop = out_ids.tolist().index(id_eot_xx)
        out = tokenizer_xx.decode(out_ids[:stop])
    else:
        out = tokenizer_xx.decode(out_ids)
    out = out.replace("<|start_header_id|>assistant<|end_header_id|>", "")
    out = out.replace("<|start_header_id|>", "")
    out = out.replace("<|end_header_id|>", "")
    del inp_ids
    return out


def rag_info(model_xx, tokenizer_xx, id_eot_xx, keyword, ctg, re_rag_mode=False, addt_info='', **model_kwarg):

    template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an agent who provides structured information about a keyword and its category. The information must be highly relevant to its category and presented in a uniform bullet points structure (maximum 3 bullet points). Each bullet point should cover a specific aspect of the keyword, such as:

- Definition or description
- Common uses or functions
- Typical location or context where it is found
- Related or associated items
- Any interesting facts or notable characteristics<|eot_id|><|start_header_id|>user<|end_header_id|>

Give me information about keyword: [{keyword}] - category: [{ctg}].<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    if re_rag_mode:
        template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an agent who provides structured information about a keyword, its category and additional information. The information must be highly relevant to its category and presented in a uniform bullet points structure (maximum 3 bullet points). Each bullet point should cover a specific aspect of the keyword, such as:

- Definition or description
- Common uses or functions
- Typical location or context where it is found
- Related or associated items
- Any interesting facts or notable characteristics<|eot_id|><|start_header_id|>user<|end_header_id|>

Give me information about keyword: [{keyword}] - category: [{ctg}]
Fact about keyword(Additional information):
{addt_info}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

#     print('fix', template)
    inp_ids = tokenizer_xx(template, return_tensors="pt").to("cuda")
    out_ids = model_xx.generate(**inp_ids, max_new_tokens=250, temperature=0.5, **model_kwarg).squeeze()
    start_gen = inp_ids.input_ids.shape[1]
    out_ids = out_ids[start_gen:]
    if id_eot_xx in out_ids:
        stop = out_ids.tolist().index(id_eot_xx)
        out = tokenizer_xx.decode(out_ids[:stop])
    else:
        out = tokenizer_xx.decode(out_ids)
#     print('out',out)
#     print('out', out)
#     print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    # FIND CTGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG
    template_get_ctg = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a agent can detect primary indentify of keyword and it category with high accuracy
Rule:
1. Answer is only primary identify no reason or verbosity around
Example 1:
user: KEYWORD: [Candy] - Category: [thing]
information: The sweet world of candy! In the category of "thing", candy is a type of food that is typically made from sugar, corn syrup, and flavorings. It comes in many forms, such as gummies, sour sugary treats, and crunchy chocolates. Candy is often consumed as a snack or used as a reward, and its varieties can be endless!
assistant: [food]
Example 2:
user: KEYWORD: [nantes france] - Category: [place]
information: Nantes, France: As a city in western France, Nantes is known for its rich history, cultural attractions, and vibrant atmosphere. Located on the Loire River, it's famous for its 15th-century Castle of the Dukes of Brittany, the Machines de l'Île (mechanical marvels), and the Passage Pommeraye
assistant: [city]
<|eot_id|><|start_header_id|>user<|end_header_id|>KEYWORD: [{keyword}] - Category: [{ctg}]
information: {out}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    inp_ids = tokenizer_xx(template_get_ctg, return_tensors="pt").to("cuda")
    out_ids = model_xx.generate(**inp_ids,max_new_tokens=20, temperature=0.1, **model_kwarg).squeeze()
    start_gen = inp_ids.input_ids.shape[1]
    out_ids = out_ids[start_gen:]
    if id_eot_xx in out_ids:
        stop = out_ids.tolist().index(id_eot_xx)
        out_ctg = tokenizer_xx.decode(out_ids[:stop])
    else:
        out_ctg = tokenizer_xx.decode(out_ids)
    out_ctg = out_ctg.strip()
#     print('inden', out_ctg)
    end_format = f"""KEYWORD: [{keyword}] - Identify: {out_ctg} - Category: [{ctg}]
information:
{out.strip()}"""
    print('endxx\n', end_format)
    del inp_ids
    return end_format

def load_keyword():
    loaded_list = []
    with open(keyword_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            loaded_list.append(row[0])
    print(f'have {len(loaded_list)} keywords')
    return loaded_list

def filter_keywords_by_range(keywords, range_start, range_end):
    # Create a list of keywords that start with a letter within the range
    next_keywords = [
        keyword for keyword in keywords
        if range_start <= keyword[:len(range_start)].lower() <= range_end
    ]
    return next_keywords
import re

class Robot:
    def __init__(self):
        self.rag = ''
        self.raw_rag = ''
        self.is_re_rag = False
        self._syn_info = ''
        self.back_sync_info = ''
        self.start_list_str = ''
        self.new_rule = []
        self.guess_list = []
        self.alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.yes_counter = 0
        self.addt_info = []

        self.edible = False

        self.alpha_mode = False
        self.current_range = []
        self.start_with = ''
        self.keyword_list = load_keyword()
        self.non_stop = True
        self.non_stop_step = 0
        self.alphabet_n = list("abcdefghijklmnopqrstuvwxyz")
        self.start = 0
        self.end = len(self.alphabet_n) - 1
        self.mid = (self.start + self.end) // 2
        self.guess = None
        self.first_letter = ''
        self.second_lette = ''
        self.hidden_letter = None
        self.search_phase = 1  # To track if we're searching for the first or second letter

        self.model_31 = ''
        self.tokenizer_31 = ''
        self.id_eot_31 = ''

        self.model_31_8bit = ''
        self.tokenizer_31_8bit = ''
        self.id_eot_31_8bit = ''

        self.model_vg_31 = ''
        self.tokenizer_vg_31 = ''
        self.id_eot_vg_31 = ''

        self.model_vg_31_8bit = ''
        self.tokenizer_vg_31_8bit = ''
        self.id_eot_31_8bit = ''

        self.model_v8 = ''
        self.tokenizer_v8 = ''
        self.id_eot_v8 = ''

        self.model_qwen = ''
        self.tokenizer_qwen = ''
        self.id_eot_qwen = ''
        self.is_load_model_31 = False
        self.is_load_model_31_8bit = False
        self.is_load_model_v8 = False
        self.is_load_model_qwen = False
        self.add_itself = False

    def reset_xx(self):
        self.rag = ''
        self.raw_rag = ''
        self._syn_info = ''
        self.back_sync_info = ''
        self.guess_list = []
        self.add_itself = False
        self.start = 0
        self.current_range = []
        self.end = len(self.alphabet_n) - 1
        self.mid = (self.start + self.end) // 2
        self.guess = None
        self.hidden_letter = None
        self.search_phase = 1

    # @torch.inference_mode
    def syn_info(self, conv, back_sync_info=None, max_bp=6, block_guess=[], step=0, start_list_str='', use_rule=''):

        template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

As a highly professional and intelligent expert in synthetic information, you excel at synthetic essential information to solve problems from user conversation.
You adeptly transform this extracted information into a suitable format based on coversation to help the user guess the keyword in a game.
The keyword alway in domains "thing".
Based on the conversation history (not fully complete), synthesize important information that can help identify the hidden keyword.
Remember, don't deny without enough information; recommend re-checking information if the answers conflict.

Rules:
1. Synthesize all information by summarizing the history and main points.
2. You answer structure with bullet points(maximum {max_bp} most important bullet points).
3. Write a detailed, thorough, in-depth, and complex summary, while maintaining clarity and brevity.
4. Keep complete information of history in {max_bp} bullet points.
5. Combine main ideas and essential information, eliminate duplicate points, and focus on important aspects.
6. If a more specific detail is identified (e.g., the keyword is a type of fruit), eliminate broader, less relevant details (e.g., the keyword is not an animal).

<|eot_id|><|start_header_id|>user<|end_header_id|>

<Conversations-history>
{conv}
</Conversations-history>

Synthetic information:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        if step > 33:
            rule = f"""Rules:
1. Synthesize all information by summarizing the history and main points.
2. You answer structure with bullet points(maximum {max_bp} most important bullet points).
3. Write a detailed, thorough, in-depth, and complex summary, while maintaining clarity and brevity.
4. Keep complete information of history in {max_bp} bullet points.
5. Combine main ideas and essential information, eliminate duplicate points, and focus on important aspects.
6. If a more specific detail is identified (e.g., the keyword is a type of fruit), eliminate broader, less relevant details (e.g., the keyword is not an animal).
7. Ensure to include information about the first letter of the keyword."""
            if step > 44:
                rule = f"""Rules:
1. Synthesize all information by summarizing the history and main points.
2. Recommend some keyword broad category and 3 keywords for each broad category.
3. You answer structure with bullet points(maximum {max_bp} most important bullet points).
4. Write a detailed, thorough, in-depth, and complex summary, while maintaining clarity and brevity.
5. Keep complete information of history in {max_bp} bullet points.
6. Combine main ideas and essential information, eliminate duplicate points, and focus on important aspects.
7. If a more specific detail is identified (e.g., the keyword is a type of fruit), eliminate broader, less relevant details (e.g., the keyword is not an animal).
8. Ensure to include information about the first letter of the keyword."""
            if start_list_str != '':
                rule += f"""\n9. Recommend keyword MUST start with {start_list_str}."""
                if use_rule != []:
                    formatted_rules = "\n".join([f"{i + 1}. {rule}" for i, rule in enumerate(use_rule)])
                    rule += f"""\nReminder for recommend:
{formatted_rules}
"""
            template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
As a highly professional and intelligent expert in synthetic information, you excel at synthetic essential information to solve problems from user conversation.
You adeptly transform this extracted information into a suitable format based on coversation to help the user guess the keyword in a game.
The keyword alway in domain "things".
Based on the conversation history (not fully complete), synthesize important information that can help identify the hidden keyword.
Remember, don't deny without enough information.

{rule}

Output:

Summary of Information by bullet points:

Concise summary of the main points from the conversation history by use bullet points

Recommended Categories and Keywords:

[Category 1]:

Example Keyword 1
Example Keyword 2
Example Keyword 3
Example Keyword 4
Example Keyword 5
[Category 2]:

Example Keyword 1
Example Keyword 2
Example Keyword 3
Example Keyword 4
Example Keyword 5
[Category 3]:

Example Keyword 1
Example Keyword 2
Example Keyword 3
Example Keyword 4
Example Keyword 5

<|eot_id|><|start_header_id|>user<|end_header_id|>

<Conversations-history>
{conv}
</Conversations-history>

Synthetic information:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
#         print('aaaaaaaaaaaaaaaaaaaaaaaa')
#         print(template)
        inp_ids = self.tokenizer_31(template, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out_ids = self.model_31.generate(**inp_ids,max_new_tokens=300, temperature=0.8, do_sample=True, top_p=0.9).squeeze()
        start_gen = inp_ids.input_ids.shape[1]
        out_ids = out_ids[start_gen:]
        if self.id_eot_31 in out_ids:
            stop = out_ids.tolist().index(self.id_eot_31)
            out = self.tokenizer_31.decode(out_ids[:stop])
        else:
            out = self.tokenizer_31.decode(out_ids)

        print('syn', out)
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        return out

    def llm_rerank(self, questions_list):
        # Prepare the list of generated questions in the required format
        formatted_questions = "\n".join([f"{i + 1}. {question}" for i, question in enumerate(questions_list)])

        # Define the prompt for choosing the best question
        choose_best_question_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

As a highly intelligent and discerning assistant, your task is to select the best question from a given list. The best question should be the one that, based on the synthesized information provided, is most likely to help narrow down the keyword effectively. Here are the details:

Synthetic information:
{self._syn_info}

List of generated questions:
{formatted_questions}

Consider the following criteria when choosing the best question:
1. How well does the question align with the synthesized information?
2. How likely is the question to narrow down the keyword effectively?
3. Does the question avoid redundancy and provide a new angle of inquiry?

Provide your choice in the following specified format:
Chosen question: <chosen>Selected question</chosen>

Reminder:
- Chosen question MUST follow the specified format, start with <chosen> and end with </chosen>.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        inp_ids = self.tokenizer_31(choose_best_question_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out_ids = self.model_31.generate(**inp_ids,max_new_tokens=100, temperature=0.5, do_sample=True, top_p=0.9).squeeze()
        start_gen = inp_ids.input_ids.shape[1]
        out_ids = out_ids[start_gen:]
        if self.id_eot_31 in out_ids:
            stop = out_ids.tolist().index(self.id_eot_31)
            out = self.tokenizer_31.decode(out_ids[:stop])
        else:
            out = self.tokenizer_31.decode(out_ids)

        return out

    def on(self, mode, obs):
        assert mode in ["asking", "guessing", "answering"], "mode can only take one of these values: asking, answering, guessing"
        if mode == "asking":
            output = self.asker(obs)
        if mode == "answering":
            output = self.answerer(obs)
#             output = self.answerer_qwen(obs)
        if mode == "guessing":
            output = self.asker(obs)
        return output

    def asker(self, obs):
        if not self.is_load_model_31:
            self.is_load_model_31 = True
            self.tokenizer_31, self.model_31 = load_model(model_id=model_id_31)
            self.id_eot_31 = self.tokenizer_31.convert_tokens_to_ids(["<|eot_id|>"])[0]
        if not self.is_load_model_v8:
            self.is_load_model_v8 = True
            self.tokenizer_v8, self.model_v8 = load_model(model_id=model_id_v8)
            self.id_eot_v8 = self.tokenizer_v8.convert_tokens_to_ids(["<|eot_id|>"])[0]
        if obs.turnType =="ask":
            #q1
            if obs.step==0 and 0:
                return "Is it Agent Alpha?"
#             elif obs.step==0:
#                 self.non_stop = False
#                 self.non_stop_step = -3
            else:
                self.non_stop = False
                self.non_stop_step = -3

            if 1 or obs.answers[0].lower() == 'no':
                self.non_stop = False
                self.search_phase = -1
            elif obs.answers[0].lower() == 'yes':
                self.alpha_mode = True
                if self.start < self.end:
                    self.mid = (self.start + self.end) // 2
                    if self.search_phase == 1:
                        question = f'Does the keyword (in lowercase) precede "{self.alphabet_n[self.mid]}zzz" in alphabetical order?'
                    else:
                        question = f'Does the keyword (in lowercase) precede "{self.first_letter}{self.alphabet_n[self.mid]}zzz" in alphabetical order?'
                    return question
                else:
                    if self.search_phase == 1:
    #                     self.first_letter = self.guess
                        # Proceed to search for the second letter
                        self.search_phase = 2
                        print('TURN 2')
                        self.start = 0
                        self.end = len(self.alphabet_n) - 1
                        return self.asker(obs)  # Continue asking for the second letter
                    else:
                        if self.non_stop_step == 0:
                            self.non_stop_step = obs.step - 3
#                         print('xxxxxxxxx', self.non_stop_step)
                        self.search_phase = -1
                        self.non_stop = False


            ask_prompt = f"""As a highly professional and intelligent expert at asking questions, you excel at asking question to guess keyword from conversations history.
The user will think of a keyword that in domains things

Your role is ask question, try to find more information of the keyword to help guess it. Focus your questions to narrow down the search space within these options. Your questions must elicit only 'yes' or 'no' responses.
to help you, here's the examples of how it should work:

<Example 1 - keyword: "Tomato">
synthetic information from history:
Synthetic information:

* The keyword is not a man-made object, but a living organism.
* The keyword is not a vertebrate, indicating it might be an invertebrate or non-vertebrate animal.
* The keyword is a living organism, which can be a things.
* The keyword starts with the letter 'R', 'S', or 'T'.
* The keyword is not man-made, but a natural living thing.
* The keyword is a living organism that is not a man-made object.

the next question not explain:
is the keyword a type of fruit?
</Example 1 - keyword: "Tomato">

<Example 2 - keyword: "oxygen tank">
synthetic information from history:
Based on the provided conversations, I will synthesize the information to help identify the keyword. Here are the five most important bullet points:

• The keyword is a man-made structure, not a natural thing.
• Its typical use does not involve a single person at a time, suggesting it's often used in a group or shared setting.
• The keyword starts with the letters 'O', 'P', or 'Q'.
• It is not typically found in households, except possibly in exceptional circumstances or in a very specific context (e.g., a household with a unique or specialized use for the keyword).
• The keyword is not related to transportation, infrastructure, or industrial use, pointing to a more general or everyday context.

the next question not explain:
Is it related to the service industry?
</Example 2 - keyword: "oxygen tank">

<Example 3 - keyword: "Rat">
synthetic information from history:
Synthetic information:

**Key Points:**

* The keyword is a living organism.
* It is a vertebrate.
* The keyword is a mammal.
* It is not a type of mammal commonly found in domestication.
* It is not a man-made object.
* The keyword is a thing, not a place.
* The first letter of the keyword is 'R', 'S', or 'T'.

the next question not explain:
Is it a primate?
</Example 3 - keyword: "Rat">

Rules:
1. Give only the question, no explain, no verbosity around.
2. Your questions must elicit only 'yes' or 'no' responses.
3. If the last 4 questions received a 'no' answer, ask question to determining the first letter of the keyword.
4. does not create a negative question
5. Include conditional questions with examples when appropriate to help narrow down the keyword.
    Example conditional questions:
    - "Is the keyword a type of building or building itself?"
    - "Is the keyword an animal commonly found in households, such as a cat, dog, or hamster?"
    - "Is the keyword a type of vehicle or vehicle itself?"""

            five_shot_things = """
Input:
Question: Is the keyword a fruit or vegetable?
Answer: yes
Question: Is the keyword a fruit?
Answer: yes
Question: Is the keyword typically red or yellow?
Answer: yes
Question: Is the keyword typically eaten raw?
Answer: yes

the next question not explain:

Output:
Is the keyword a tropical fruit?

Input:
Question: Is the keyword the name of an alcoholic drink?
Answer: no
Question: Is the keyword a common beverage?
Answer: yes

the next question not explain:

Output:
Is the keyword consumed hot?

Input:
Question: Would the keyword be included in the broad category of Computer hardware?
Answer: no
Question: Would the keyword be considered a Home appliance?
Answer: no
Question: Is the thing related to a means of transportation?
Answer: no

the next question not explain:

Output:
Is it used for construction purposes?

Input:
Question: Would the keyword be included in the broad category of Flora?
Answer: no
Question: Would the keyword be included in the broad category of Chemical substances?
Answer: yes
Question: Would the keyword be included in the broad category of Medical technology?
Answer: no

the next question not explain:

Output:
Is the word associated with nature or natural resources?

Input:
Question: Would the keyword be included in the broad category of Architectural elements?
Answer: yes
Question: Would the keyword be included in the broad category of Health care?
Answer: no
Question: Is it used indoors?
Answer: no
Question: Would the keyword be included in the broad category of Electronics?
Answer: no
Question: Would the keyword be included in the broad category of Office equipment?
Answer: yes
Question: Would the keyword be included in the broad category of Writing media?
Answer: yes

the next question not explain:

Output:
Would the keyword be considered a Publication?
"""

            ask_prompt_his = f"""As a highly professional and intelligent expert at asking questions, you excel at asking question to guess keyword from syntheticed information.
The user will think of a keyword in domain things

Your role is ask question, try to find more information of the keyword to help guess it. Focus your questions to narrow down the search space within these options. Your questions must elicit only 'yes' or 'no' responses.
to help you, here's the examples of how it should work:
{five_shot_things}
Rules:
1. Give only the question, no explain, no verbosity around.
2. Your questions must elicit only 'yes' or 'no' responses.
3. does not create a negative question
4. Include conditional questions with examples when appropriate to help narrow down the keyword.
    Example conditional questions:
    - "Is the keyword a type of building, such as a house, skyscraper, or hut?"
    - "Is the keyword an animal commonly found in households, such as a cat, dog, or hamster?"
    - "Is the keyword a type of vehicle, such as a car, bicycle, or boat?
5. Keep question simple."""
            chat_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{ask_prompt}<|eot_id|>"""
            chat_template_his = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{ask_prompt_his}<|eot_id|>"""

            history = ''
            if len(obs.questions)>=1:
                for q, a in zip(obs.questions, obs.answers):
                    history += f"""Question: {q}\nAnswer: {a}\n"""
                    if q == "Is the keyword a man-made object or man-made product?" and a.lower() == 'yes' and self.add_itself == False:
                        print('active add itself')
                        self.add_itself = True
                    if q == "Is the keyword a product derived from producers (plants) or consumers (animals), rather than fungi?" and a.lower() == 'no' and self.add_itself == False:
                        print('active add itself')
                        self.add_itself = True
#                     chat_template += f"{q}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
#                     chat_template += f"{a}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            chat_template += f"<|start_header_id|>user<|end_header_id|>\n\nsynthetic information from history:\n{self._syn_info}\nthe next question not explain:<|eot_id|>"
            chat_template += "<|start_header_id|>assistant<|end_header_id|>"
            chat_template_his += f"<|start_header_id|>user<|end_header_id|>\n\nConversation history:\n{history}\nthe next question not explain:<|eot_id|>"
            chat_template_his += "<|start_header_id|>assistant<|end_header_id|>"
            try_ask_time = 5
            questions_list = []
            if obs.step<=18:
#                 print('curr')
#                 print(chat_template_his)
                for _ in range(try_ask_time):
                    output = generate_answer(self.model_31, self.tokenizer_31, self.id_eot_31, chat_template_his)
                    output = output.strip().lower()
                    if len(output)>=12:
                        questions_list.append(output)
            else:
                for _ in range(try_ask_time):
                    output = generate_answer(self.model_v8, self.tokenizer_v8, self.id_eot_v8, chat_template)
                    output = output.strip().lower()
                    if len(output)>=12:
                        questions_list.append(output)
            questions_list = list(set(questions_list))
            print('start_ql')
            for q in questions_list:
                print(q)
            print('end_ql')
            if len(questions_list) > 1:
                output = self.llm_rerank(questions_list)
                match = re.search(r'<chosen>(.*?)</chosen>', output)
                match_1 = re.search(r'<chosen>(.*?)<chosen>', output)
                if match:
                    output = match.group(1).strip()
                elif match_1:
                    output = match.group(1).strip()
                else:
                    print("can't catch rerank")
                    print(output)
                    output = questions_list[0]
            else:
                print('all questions error')

            output = output.lower()
            output = output.strip()
            if self.add_itself:
                tmp = output
                if 'specific type of ' in tmp:
                    print('refine question')
                    xx = tmp.rsplit('specific type of ', maxsplit=1)
                    x = xx[-1]
                    if x[-1] == '?':
                        x = x[:-1]
                        if len(x.split())<3:
#                             output = xx[0] + 'type of ' + f'{x} or {x} itself?'
                            output = xx[0] + f'{x} itself or specific type of {x}?'
                elif 'type of ' in tmp:
                    print('refine question')
                    xx = tmp.rsplit('type of ', maxsplit=1)
                    x = xx[-1]
                    if x[-1] == '?':
                        x = x[:-1]
                        if len(x.split())<3:
#                             output = xx[0] + 'type of ' + f'{x} or {x} itself?'
                            output = xx[0] + f'{x} itself or type of {x}?'
            return output

# # GUESSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
        elif obs.turnType == "guess":
            if obs.answers[0] == 'no':
                self.non_stop = False
            if obs.step == 2 and obs.answers[0] == 'yes' and self.alpha_mode:
                print('random choice first')
                return str(random.choice(self.keyword_list))
            if obs.step > 2 and self.search_phase == 1 and len(self.start_with)<2:
                if obs.answers[-1] == 'yes':
                    self.end = self.mid
                else:
                    self.start = self.mid + 1
#                 ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
                if self.start < self.end:
                    self.current_range = [self.alphabet_n[self.start], self.alphabet_n[self.end]]
#                     print('current', self.current_range)
                else:
                    self.current_range = [self.alphabet_n[self.start], self.alphabet_n[self.start]]
                    self.guess = self.alphabet_n[self.start]
                    self.first_letter = self.guess
                    self.start_with = self.guess
            elif obs.step > 2 and self.search_phase == 2 and len(self.start_with)<2:
                if obs.answers[-1] == 'yes':
                    self.end = self.mid
                else:
                    self.start = self.mid + 1

                if self.start < self.end:
                    self.current_range = [self.first_letter + self.alphabet_n[self.start], self.first_letter + self.alphabet_n[self.end]]
#                     print('current', self.current_range)
                else:
                    self.current_range = [self.first_letter + self.alphabet_n[self.start], self.first_letter + self.alphabet_n[self.start]]
                    self.second_letter = self.alphabet_n[self.start]
                    self.start_with = self.first_letter + self.second_letter
            print('start_with', self.start_with)
            if len(self.current_range) == 2 and self.non_stop == True:
                self.keyword_list = filter_keywords_by_range(self.keyword_list, self.current_range[0], self.current_range[1])
                if len(self.keyword_list) !=0:
                    print('current range', self.current_range)
                    print('fail to random choice')
                    return str(random.choice(self.keyword_list))
            conv = ""

            for i, (q, a) in enumerate(zip(obs.questions, obs.answers)):
                if "To make sure you understand requirement, answer below question" in q:
                    q = q.rsplit('answer below question:\n')[-1]
                if q == "Is it Agent Alpha?":
                    continue
                if "Does the keyword (in lowercase) precede" in q:
                    continue

                if q == "Does the keyword start with one of the letters 'B', 'C', 'P' or 'S'?" and a == 'yes':
                    self.start_list_str = "'B', 'C', 'P' or 'S'"
                if q == "Does the keyword start with one of the letters 'F', 'M', 'T' or 'W'?" and a == 'yes':
                    self.start_list_str = "'F', 'M', 'T' or 'W'"
                if q == "Does the keyword start with one of the letters 'B', 'C', 'P' or 'S'?" and "Does the keyword start with one of the letters 'F', 'M', 'T' or 'W'?" in obs.questions and obs.answers[i+1] == 'yes':
                    continue
                if obs.step>=17 and q == "Does the keyword start with one of the letters 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M' or 'N'?":
                    continue
                if obs.step>=20 and q == "Does the keyword start with one of the letters 'A', 'B', 'C', 'D', 'E', 'F' or 'G'?":
                    continue
                if obs.step>=20 and q == "Does the keyword start with one of the letters 'O', 'P', 'Q', 'R', 'S' or 'T'?":
                    continue

                conv += f"""Question: {q}\nAnswer: {a}\n"""
            if self.start_with != '':
                conv = f"""Question: Does the keyword start with '{self.start_with}?'\nAnswer: yes\n""" + conv
#             conv += f"""Question: {obs.questions[-1]}\nAnswer: {obs.answers[-1]}\n"""
            if obs.step < (9+self.non_stop_step):
                chat_template =  f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a smart AI assistant skilled at playing the 20 questions game. The user will think of a keyword in domain things.

Base on the history, can you guess the keyword, so make sure you understand information.

Rule:
1. Give only one the keyword, no explain, no verbosity around.
2. You can respond according to the information provided in the comment

Example:

<Conversations-history>
Question: keyword is thing. Is keyword man-made?
Answer: yes
Question: Is it typically found households?
Answer: yes
Question: Is the thing related to food or drink in any way?
Answer: yes
Question: Does the keyword start with one of the letters 'A', 'B', 'C', 'D', 'E', 'F' or 'G'?
Answer: yes
Question: is the keyword a type of food?
Answer: yes
Question: is the first letter of the keyword 'e'?
Answer: no
</Conversations-history>

The guess keyword not explain:
assistant:

Apple<|eot_id|><|start_header_id|>user<|end_header_id|>

<Conversations-history>
{conv}
</Conversations-history>

The guess keyword not explain:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
                output = generate_answer(self.model_v8, self.tokenizer_v8, self.id_eot_v8, chat_template, max_new_tokens=50)
                output = output.lower()
                output = output.strip()
                return output
            elif obs.step >= (9+self.non_stop_step):
                self.guess_list = obs.guesses
                self.guess_list = list(set(self.guess_list))
                if obs.step>40:
                    self._syn_info = self.syn_info(conv, self.back_sync_info, max_bp=12, block_guess=self.guess_list, step=obs.step, start_list_str=self.start_list_str, use_rule=self.new_rule)
                elif obs.step>30:
                    self._syn_info = self.syn_info(conv, self.back_sync_info, max_bp=10, block_guess=self.guess_list, step=obs.step, start_list_str=self.start_list_str, use_rule=self.new_rule)
                else:
                    self._syn_info = self.syn_info(conv, self.back_sync_info, max_bp=7, block_guess=self.guess_list, step=obs.step, start_list_str=self.start_list_str, use_rule=self.new_rule)
                self.back_sync_info = self._syn_info
                chat_template =  f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a smart AI assistant skilled at playing the 20 questions game. The user will think of a keyword in domain things

Base on the synthetic information, you can guess the keyword.

Rule:
1. Make sure you understand information before guess.
2. You can respond according to the information provided in the comment.
3. keyword must not in list: {', '.join(self.guess_list)}.
4. Give only one the keyword, no explain, no verbosity around.

Example:

<synthetic-information>
So far, the current state of the game is as synthetic information:

Based on the conversation history, here's the synthesized information:

• The keyword starts with one of the letters 'R', 'S', or 'T'.
• The keyword is a man-made, non-living, non-organic, small-sized object that can be held in one's hand.
• The object is designed to be held in one's hand for a direct interaction, such as grasping, cutting, or writing.
• The object is not a living organism, or natural resource component of the earth, nor is it a structure (natural or man-made).
• The object is a type of 'thing', in the domain of a product, possibly a tool or a consumer good, with a specific function, given the context of being held for a direct interaction.
</synthetic-information>

The guess keyword not explain:
assistant:

Scissors<|eot_id|><|start_header_id|>user<|end_header_id|>

So far, the current state of the game is as synthetic information:

<synthetic-information>
{self._syn_info}
Non-relevant keyword list: [{', '.join(self.guess_list)}].
</synthetic-information>

The guess keyword not explain:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n
"""
                for _ in range(5):

                    output = generate_answer(self.model_v8, self.tokenizer_v8, self.id_eot_v8, chat_template, max_new_tokens=50, do_sample=True)
                    output = output.lower()
                    output = output.strip()
                    if (output not in self.guess_list):
                        if len(output.split())<=4 and len(output)<35:
                            if (self.start_with != '' and len(output)>=len(self.start_with)):
                                if self.start_with == output[:len(self.start_with)]:
                                    return output
                            elif self.start_with == '':
                                return output
                        else:
                            print('Dup guess(>4 words or len>=35):', output)
                    else:
                        print('Dup guess(in guess list):', output)
                return output

    def generate_answer_thread(args):
        # Unpack arguments
        model_xx, tokenizer_xx, id_eot_xx, template, max_new_tokens, temperature = args

  # Your generate_answer logic here

# ANSWERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR
    def answerer(self, obs):

        question = obs.questions[-1]
        if obs.questions[-1] == 'Is it Agent Alpha?':
            self.alpha_mode = True
            return 'yes'

        if self.alpha_mode:
            match = re.search(r'keyword.*(?:come before|precede) \"([^\"]+)\" .+ order\?$', question)
            if match:
                testword = match.group(1)
                if testword is not None:
                    response = 'yes' if obs.keyword.lower() < testword else 'no'
                    return response

        if not self.is_load_model_31_8bit:
            self.is_load_model_31_8bit = True
            self.tokenizer_31_8bit, self.model_31_8bit = load_model(model_id=model_id_31_8bit)
            self.id_eot_31_8bit = self.tokenizer_31_8bit.convert_tokens_to_ids(["<|eot_id|>"])[0]

        if not self.rag:
#             self.rag = rag_info(self.model_31, self.tokenizer_31, self.id_eot_31, obs.keyword, obs.category)
            self.rag = rag_info(self.model_31_8bit, self.tokenizer_31_8bit, self.id_eot_31_8bit, obs.keyword, obs.category, do_sample=True, top_p=0.9)
            self.raw_rag = self.rag
        if self.yes_counter == 3 and self.is_re_rag == False:
            print('fail to RE-RAG')
            self.is_re_rag = True
            self.rag = rag_info(self.model_31_8bit, self.tokenizer_31_8bit, self.id_eot_31_8bit, obs.keyword, obs.category,
                                re_rag_mode=True, addt_info='\n'.join(self.addt_info[:3]), do_sample=True, top_p=0.9)
            self.raw_rag = self.rag
        self.rag = self.raw_rag
        print('KEYWORD:', obs.keyword)
        def extract_letters_from_question(question):
            # Updated regex to correctly exclude 's' only when preceded by an apostrophe in the word
            matches = re.findall(r"(?<=[ ,\[\('\"])([a-zA-Z])(?=[ ,\]\)'\"\?]|$)", question)
            # Filter out 's' when it is part of possessive forms or contractions
            filtered_matches = []
            for match in matches:
                # Check if the matched 's' is part of "'s" or standalone
                if match.upper() == 'S' and "'s" in question:
                    # Skip 's' if it's directly following an apostrophe indicating possessive form
                    apostrophe_index = question.find("'s")
                    if apostrophe_index > 0 and question[apostrophe_index - 1].isalpha():
                        continue
                filtered_matches.append(match.upper())

            return list(set(filtered_matches))
        first_letter = obs.keyword[0].upper()  # Convert to uppercase for consistency
        letters_list = extract_letters_from_question(obs.questions[-1])
        if len(letters_list) > 1:
            print('fall to auto check list letter for question:', obs.questions[-1])
            print('list letters:', letters_list)
            if first_letter in letters_list:
                return "yes"
            else:
                return 'no'
        self.yes_counter = 0
        for q, a in zip(obs.questions, obs.answers):
            if q == "Is it Agent Alpha?":
                continue
            if "Does the keyword (in lowercase) precede" in q:
                continue
            if a.lower() == 'yes' and "Does the keyword (in lowercase) precede" not in q:
                self.addt_info.append(f'{q} Answer: {a}.')
                self.yes_counter += 1

#         print('self info:', self.info)
        sys_prompt = f"""you are a helpful AI assistant, and your are very smart in playing 20 questions game,
User guess the keyword by asking you up to 20 questions, your answers to be valid must be a 'yes' or 'no'.
Know that the user will always guess a keyword in domain things.

Rules:
1. Your keyword is "{obs.keyword}"
2. Check each letter of keyword {obs.keyword}
3. Your answer must be based solely on the characteristics of the keyword itself, not on any related or indirect uses of the keyword.
4. Answer only **yes** or **no**
so make sure you understand the user's question and you understand the keyword you're playing on.
for now the word that the user should guess is: "{obs.keyword}", it is of category "{obs.category}"

Information about keyword:
- keyword start with: '{obs.keyword[0].lower()}'
- {self.rag}"""

        chat_template = f"""<|im_start|>system\n\n{sys_prompt}<|im_end|>"""
        chat_template += "<|im_start|>user\n\n"
        chat_template += f"Explaining your answer. Output the **yes** or **no** answer after explain reason you answer MUST have **yes** or **no**.\nQuestion:{obs.questions[-1]}<|im_end|>"
        chat_template += "<|im_start|>assistant\n\n"

        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        sys_prompt_xx = f"""you are a helpful AI assistant, and your are very smart in playing 20 questions game.
User guess the word by asking you up to 20 questions, your answers to be valid must be a 'yes' or 'no'.

Rules:
1. Your keyword is [{obs.keyword}].
2. Check each letter of keyword [{obs.keyword}].

so make sure you understand the user's question and you understand the keyword you're playing on.
for now the word that the user should guess is: "{obs.keyword}", it is of category "{obs.category}"
some basic knowledge:
- alphabetical order: {self.alphabet}.
some information about keyword:
- keyword start with: '{obs.keyword[0].upper()}'.
- {self.rag}"""

        chat_template_xx_llama = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|>"""
        chat_template_xx_llama += "<|start_header_id|>user<|end_header_id|>\n\n"
        chat_template_xx_llama += f"Output the **yes** or **no** answer after explain reason, you answer MUST have **yes** or **no**.\nQuestion:{obs.questions[-1]}<|eot_id|>"
        chat_template_xx_llama += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        chat_template_xx = f"""<|im_start|>system\n\n{sys_prompt_xx}<|im_end|>"""
        chat_template_xx += "<|im_start|>user\n\n"
        chat_template_xx += f"Output the 'yes' or 'no' answer after explain reason, you answer MUST have **yes** or **no**.\nQuestion:{obs.questions[-1]}<|im_end|>"
        chat_template_xx += "<|im_start|>assistant\n\n"

        x = []
        x_raw = []
        try_time = 5
        for _ in range(try_time):
            if 'or' in obs.questions[-1].split() and 0:
                print('direct to llama')
                output_raw = generate_answer(self.model_31, self.tokenizer_31, self.id_eot_31, chat_template_xx_llama, max_new_token=30)
            else:
#                 output = generate_answer(self.model_qwen, self.tokenizer_qwen, self.id_eot_qwen, chat_template_xx)
#                 output = generate_answer(self.model_31, self.tokenizer_31, self.id_eot_31, chat_template_xx_llama)
                output_raw = generate_answer(self.model_31_8bit, self.tokenizer_31_8bit, self.id_eot_31_8bit, chat_template_xx_llama, max_new_tokens=50, temperature=0.55, do_sample=True, top_p=0.9)
            x_raw.append(output_raw)
            output = output_raw.lower()
            output_words_list = re.sub(r'\W+', ' ', output)
            output_words_list = output_words_list.split()
#             print(output_words_list)
            if "**answer: yes**" in output:
                output = 'yes'
            elif "**answer: no**" in output:
                output = 'no'
            elif "answer is: yes" in output:
                output = 'yes'
            elif "answer is: no" in output:
                output = 'no'
            elif "answer is: 'yes'" in output:
                output = 'yes'
            elif "answer is: 'no'" in output:
                output = 'no'
            elif "**answer:** yes" in output:
                output = 'yes'
            elif "**answer:** no" in output:
                output = 'no'
            elif "**yes**" in output:
                output = "yes"
            elif "**no**" in output:
                output = "no"
            elif "Yes." in output_raw:
                output = 'yes'
            elif "No." in output_raw:
                output = 'no'
            elif "yes" in output_words_list:
                output = "yes"
            elif "no" in output_words_list or 'not' in output_words_list:
                output = "no"
            else:
                output = "no"
    #         print(chat_template)
            x.append(output)
        for i in x_raw:
            print(i)
        print(x)
        output = 'no' if sum(answer == 'no' for answer in x) > sum(answer == 'yes' for answer in x) else 'yes'
        return output

robot = Robot()

def agent(obs, cfg):
    if obs.turnType =="ask":
        response = robot.on(mode = "asking", obs = obs)

    elif obs.turnType =="guess":
        response = robot.on(mode = "guessing", obs = obs)

    elif obs.turnType =="answer":
        response = robot.on(mode = "answering", obs = obs)

    if response == None or len(response)<=1:
        response = "yes"
    print('STEPPPPPPPPPPPPPPPPPPPPPPPPPPPPP:', obs.step)
    if obs.turnType=='ask':
        print(obs.turnType,':\n', response.split('Answer below question:\n')[-1])
    else:
        print(obs.turnType,':\n', response)
    return response 

# env.render(mode="ipython", width=850, height=600)
