#!/usr/bin/python3

from langchain_core.prompts.prompt import PromptTemplate

def label_tanl_template(tokenizer):
  system_message = """please label entities (Material, Number, Operation, Amount-Unit, Condition-Unit, Material-Descriptor, Condition-Misc, Synthesis-Apparatus, Nonrecipe-Material, Brand) and relations (Recipe-target, Solvent-material, Atmospheric-material, Recipe-precursor, Participant-material, Apparatus-of, Condition-of, Descriptor-of, Number-of, Amount-of, Apparatus-attr-of, Brand-of, Coref-of, Next-operation) of the text as following examples. do not modify the original text, just add marks to the original text.

examples:

example 1:

input:
The Baeyer-Villiger oxidation of ketones with Oxone(r) in the presence of ionic liquids as solvents Oxone(r) (4.0 mmol) was added to a solution of ketone (4.0 mmol) in ionic liquid (3.0 g) and stirred at 40 degC for 2.5-20 h (depending on the reaction rate).

output:
The Baeyer-Villiger oxidation of [ ketones | Material ] with Oxone ( r ) in the presence of ionic liquids as solvents [ Oxone(r) | Material | Recipe Precursor of = added ] ( [ 4.0 | Number | Number Of = mmol ] [ mmol | Amount-Unit | Amount Of = Oxone(r) ] ) was [ added | Operation | Next Operation = stirred | : = Operation ] to a [ solution | Material-Descriptor | Descriptor Of = ketone ] of [ ketone | Material | Recipe Precursor of = added ] ( [ 4.0 | Number | Number Of = mmol ] [ mmol | Amount-Unit | Amount Of = ketone ] ) in [ ionic liquid | Material | Solvent Material of = added ] ( [ 3.0 | Number | Number Of = g ] [ g | Amount-Unit | Amount Of = ionic liquid ] ) and [ stirred | Operation | Next Operation = monitored | : = Operation ] at [ 40 | Number | Number Of = degC ] [ degC | Condition-Unit | Condition Of = stirred ] for [ 2.5-20 | Number | Number Of = h ] [ h | Condition-Unit | Condition Of = stirred ] ( depending on the [ reaction rate | Condition-Type ] ) .

example 2:

input:
The progress of the reaction was monitored by GC or HPLC. After this time, the post reaction mixture was dissolved in CH2Cl2 and filtered.

output:
The progress of the reaction was [ monitored | Operation | Next Operation = dissolved | : = Operation ] by [ GC | Characterization-Apparatus | Apparatus Of = monitored ] or [ HPLC | Characterization-Apparatus | Apparatus Of = monitored ] . After this time, the post reaction [ mixture | Material ] was [ dissolved | Operation | Next Operation = filtered | : = Operation ] in [ CH2Cl2 | Material | Solvent Material of = dissolved ] and [ filtered | Operation | Next Operation = concentrated | : = Operation ] .

example 3:

input:
Next, the filtrate was concentrated and extracted with the appropriate solvent of ethyl acetate , diethyl or dibutyl ether ( 6x5 mL ) and concentrated .

output:
Next, the [ filtrate | Material ] was [ concentrated | Operation | Next Operation = extracted | : = Operation ] and [ extracted | Operation | Next Operation = concentrated | : = Operation ] with the appropriate [ solvent | Material-Descriptor | Descriptor Of = ethyl acetate | Descriptor Of = diethyl | Descriptor Of = dibutyl ether ] of [ ethyl acetate | Material | Solvent Material of = extracted ] , [ diethyl | Material | Solvent Material of = extracted ] or [ dibutyl ether | Material | Solvent Material of = extracted ] ( [ 6x5 | Number | Number Of = mL ] [ mL | Amount-Unit | Amount Of = ethyl acetate | Amount Of = diethyl | Amount Of = dibutyl ether ] ) and [ concentrated | Operation | Next Operation = purification | : = Operation ] .

example 4:

input:
The yields of lactones after the purification by column chromatography with hexane / ethyl acetate ( 4:1 ) as the eluent were 65-95 % .

output:
The [ yields | Property-Type ] of [ lactones | Material ] after the [ purification | Operation | Next Operation = distillation | : = Operation ] by [ column chromatography | Synthesis-Apparatus | Apparatus Of = purification ] with [ hexane | Nonrecipe-Material ] / [ ethyl acetate | Nonrecipe-Material ] ( [ 4:1 | Number ] ) as the [ eluent | Material-Descriptor | Descriptor Of = ethyl acetate | Descriptor Of = hexane ] were [ 65-95 | Number | Number Of = % ] [ % | Property-Unit | Property Of = lactones ] .

example 5:

input:
For HmimOAc ( bp 240 degC ; 70 degC / 1.2 mbar ) and H2mpyrOAc ( bp 255 degC ; 90 degC / 1.2 mbar ) distillation of the product or ionic liquid from the reaction mixture was performed .

output:
For [ HmimOAc | Material ] ( [ bp | Property-Type | Type Of = degC ] [ 240 | Number | Number Of = degC ] [ degC | Property-Unit | Property Of = HmimOAc ] ; [ 70 | Number | Number Of = degC ] [ degC | Property-Unit | Property Of = HmimOAc ] / [ 1.2 | Number | Number Of = mbar ] [ mbar | Property-Unit | Property Of = HmimOAc ] ) and [ H2mpyrOAc | Material ] ( [ bp | Property-Type | Type Of = degC ] [ 255 | Number | Number Of = degC ] [ degC | Property-Unit | Property Of = H2mpyrOAc ] ; [ 90 | Number | Number Of = degC ] [ degC | Property-Unit | Property Of = H2mpyrOAc ] / [ 1.2 | Number | Number Of = mbar ] [ mbar | Property-Unit | Property Of = H2mpyrOAc ] ) [ distillation | Operation | Next Operation = performed | : = Operation ] of the [ product | Material ] or [ ionic liquid | Material ] from the reaction [ mixture | Material ] was [ performed | Operation | Next Operation = purified | : = Operation ] .

example 6:

input:
ILs were purified for recycling tests .

output:
[ ILs | Material ] were [ purified | Operation | Next Operation = filtration | : = Operation ] for recycling tests .

example 7:

input:
After the filtration of post reaction mixture , and the extraction of the product with ethyl acetate ( bmimBF4 ) or dibutyl ether ( HmimOAc ) , ILs were concentrated , dried under vacuum ( 60 degC , 5 h ) and reused .

output:
After the [ filtration | Operation | Next Operation = extraction | : = Operation ] of post reaction [ mixture | Material ] , and the [ extraction | Operation | Next Operation = concentrated | : = Operation ] of the [ product | Material ] with [ ethyl acetate | Material ] ( [ bmimBF4 | Material-Descriptor | Descriptor Of = ethyl acetate ] ) or [ dibutyl ether | Material ] ( [ HmimOAc | Material-Descriptor | Descriptor Of = dibutyl ether ] ) , [ ILs | Material ] were [ concentrated | Operation | Next Operation = dried | : = Operation ] , [ dried | Operation | Next Operation = reused | : = Operation ] under [ vacuum | Condition-Misc | Condition Of = dried ] ( [ 60 | Number | Number Of = degC ] [ degC | Condition-Unit | Condition Of = dried ] , [ 5 | Number | Number Of = h ] [ h | Condition-Unit | Condition Of = dried ] ) and [ reused | Operation | Next Operation = stirred | : = Operation ] .
"""
  messages = [
    {'role': 'system', 'content': system_message},
    {'role': 'user', 'content': '{text}'}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ['text'])
  return template

