#!/usr/bin/python3

from langchain_core.prompts.prompt import PromptTemplate

def label_tanl_template(tokenizer):
  system_message = """please label entities (Material, Number, Operation, Amount-Unit, Condition-Unit, Material-Descriptor, Condition-Misc, Synthesis-Apparatus, Nonrecipe-Material, Brand) and relations (Recipe-target, Solvent-material, Atmospheric-material, Recipe-precursor, Participant-material, Apparatus-of, Condition-of, Descriptor-of, Number-of, Amount-of, Apparatus-attr-of, Brand-of, Coref-of, Next-operation) of the text as following examples. do not modify the original text, just add marks to the original text.

examples:

example1:

input:
The Baeyer-Villiger oxidation of ketones with Oxone(r) in the presence of ionic liquids as solvents Oxone(r) (4.0 mmol) was added to a solution of ketone (4.0 mmol) in ionic liquid (3.0 g) and stirred at 40 degC for 2.5-20 h (depending on the reaction rate).

output:
The Baeyer-Villiger oxidation of [ ketones | Material ] with Oxone(r) in the presence of ionic liquids as solvents [ Oxone(r) | Material | Recipe Precursor of = added ] [ (4.0 | Number | Number Of = mmol ] [ mmol) | Amount-Unit | Amount Of = Oxone(r) ] was [ added | Operation | Next Operation = stirred | | = Operation ] to a [ solution | Material-Descriptor | Descriptor Of = ketone ] of [ ketone | Material | Recipe Precursor of = added ] [ (4.0 | Number | Number Of = mmol ] [ mmol) | Amount-Unit | Amount Of = ketone ] in [ ionic liquid | Material | Solvent Material of = added ] [ (3.0 | Number | Number Of = g ] [ g) | Amount-Unit | Amount Of = ionic liquid ] and [ stirred | Operation | Next Operation = monitored | | = Operation ] at [ 40 | Number | Number Of = degC ] [ degC | Condition-Unit | Condition Of = stirred ] for [ 2.5-20 | Number | Number Of = h ] [ h | Condition-Unit | Condition Of = stirred ] (depending on the [ reaction rate). | Condition-Type ]

example2:

input:
The progress of the reaction was monitored by GC or HPLC. After this time, the post reaction mixture was dissolved in CH2Cl2 and filtered.

output:
The progress of the reaction was [ monitored | Operation | Next Operation = dissolved | | = Operation ] by [ GC | Characterization-Apparatus | Apparatus Of = monitored ] or [ HPLC. | Characterization-Apparatus | Apparatus Of = monitored ] After this time, the post reaction [ mixture | Material ] was [ dissolved | Operation | Next Operation = filtered | | = Operation ] in [ CH2Cl2 | Material | Solvent Material of = dissolved ] and [ filtered. | Operation | Next Operation = concentrated | | = Operation ]
"""
  messages = [
    {'role': 'system', 'content': system_message},
    {'role': 'user', 'content': '{text}'}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ['text'])
  return template

