import argparse
import json
import os.path
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import transformers

template = '''Generate a python-like program that describes the reasoning steps required to verify the claim step-by-step. You can call three functions in the program: 1. Question() to answer a question; 2. Verify() to verify a simple claim; 3. Predict() to predict the veracity label. Several examples are given as follows.

# The claim is that In 1959, former Chilean boxer Alfredo Cornejo Cuevas (born June 6, 1933) won the gold medal in the welterweight division at the Pan American Games (held in Chicago, United States, from August 27 to September 7) in Chicago, United States, and the world amateur welterweight title in Mexico City.
def program():
    fact_1 = Verify("Alfredo Cornejo Cuevas was born in June 6, 1933.")
    fact_2 = Verify("Alfredo Cornejo Cuevas won the gold medal in the welterweight division at the Pan American Games in 1959.")
    fact_3 = Verify("The Pan American Games in 1959 was held in Chicago, United States, from August 27 to September 7.")
    fact_4 = Verify("Alfredo Cornejo Cuevas won the world amateur welterweight title in Mexico City.")
    label = Predict(fact_1 and fact_2 and fact_3 and fact_4)
    #end

# The claim is that The Footwork FA12, which was intended to start the season, finally debuted at the San Marino Grand Prix, a Formula One motor race held at Imola on 28 April 1991.
def program():
    fact_1 = Verify("The Footwork FA12, which was intended to start the season.")
    fact_2 = Verify("The Footwork FA12 finally debuted at the San Marino Grand Prix.")
    fact_3 = Verify("The San Marino Grand Prix was a Formula One motor race held at Imola on 28 April 1991.")
    label = Predict(fact_1 and fact_2 and fact_3)
    #end

# The claim is that SkyHigh Mount Dandenong (formerly Mount Dandenong Observatory) is a restaurant located on top of Mount Dandenong, Victoria, Australia.
def program():
    fact_1 = Verify("SkyHigh Mount Dandenong is a restaurant located on top of Mount Dandenong, Victoria, Australia.")
    fact_2 = Verify("SkyHigh Mount Dandenong is formerly known as Mount Dandenong Observatory.")
    label = Predict(fact_1 and fact_2)
    #end
    
# The claim is that In the 2001 Stanley Cup playoffs Eastern Conference Semifinals Devils' Elias scored and Maple Leafs' left Devils player Scott Neidermayer hurt.
def program():
    fact_1 = Verify("In the 2001 Stanley Cup playoffs Eastern Conference Semifinals Devils' Elias scored.")
    fact_2 = Verify("Maple Leafs' left Devils player Scott Neidermayer hurt.")
    label = Predict(fact_1 and fact_2)
    #end
    
# The claim is that Teldenia helena is a moth first described in 1967 by Wilkinson.
def program():
    fact_1 = Verify("Teldenia helena is a moth.")
    fact_2 = Verify("Teldenia helena was first described by Wilkinson in 1967.")
    label = Predict(fact_1 and fact_2)
    #end
    
# The claim is that Born December 30, 1974, William Frick was a dark horse candidate in the Maryland House of Delegates appointment process.
def program():
    fact_1 = Verify("William Frick was born in December 30, 1974.")
    fact_2 = Verify("William Frick was a dark horse candidate in the Maryland House of Delegates appointment process.")
    label = Predict(fact_1 and fact_2)
    #end

# The claim is that [[CLAIM]]
def program():'''

def generate_programs(dataset_json_path, output_path, num_examples, model, save_path):
    # create output dir
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # 加载新数据
    with open(dataset_json_path, 'r') as f:
        dataset = json.load(f)

    dataset_list = []
    for idx, (key, val) in enumerate(dataset.items()):
        if idx >= num_examples:
            break
        dataset_list.append(val)
    print(f'Loaded {num_examples} examples from {dataset_json_path}')

    results = []
    for idx, sample in enumerate(dataset_list):
        result = {
            'idx': idx,
            'id': sample['origin_id'],
            'claim': sample['generated_text_glm4'],
            'gold': 'refutes',
            'predicted_programs': []
        }
        results.append(result)


    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    for result in results:
        full_prompt = proprompt_construction(result['claim'].replace('\n', '').replace('\r', ''))
        sequences = pipeline(
            full_prompt,
            do_sample=True,
            top_k=10,
            temperature=0.1,
            top_p=0.95,
            num_return_sequences=1,
            max_length=2000,
            eos_token_id=1
        )
        result['predicted_programs'] = sequences[0]['generated_text'].split('def program():')[-1]

    print(f"Generated {len(results)} examples.")

    with open(os.path.join(save_path, f'CodeLlama_programs.json'),'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    return results


def proprompt_construction(claim):
    return template.replace('[[CLAIM]]', claim)

def parse_args():
    parser = argparse.ArgumentParser()
    # dataset args
    parser.add_argument('--dataset_json_path', type=str)
    parser.add_argument('--num_examples', default=100, type=int)
    parser.add_argument('--output_path', default = './results/programs', type=str)
    parser.add_argument('--model', default = 'meta-llama/CodeLlama-7b-hf', type=str)
    parser.add_argument('--save_path', default = '../outputs', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    generate_programs(args.dataset_json_path, args.output_path, args.num_examples, args.model, args.save_path)