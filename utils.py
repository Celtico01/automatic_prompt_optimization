import time
import openai
import string
import os
from salvar_saidas_gpt import salvar_resposta
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def parse_sectioned_prompt(s):
    result = {}
    current_header = None
    
    for line in s.split('\n'):
        line = line.strip()
        if line.startswith('# '):
            current_header = line[2:].strip().lower().split()[0]
            current_header = current_header.translate(str.maketrans('', '', string.punctuation))
            result[current_header] = ''
        elif current_header is not None:
            result[current_header] += line + '\n'
    
    return result

def chatgpt(prompt, n=1, top_p=1, stop=None, presence_penalty=0, frequency_penalty=0, logit_bias={}, timeout=80):
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = client.chat.completions.create(
            model=os.getenv("MODEL"),
            messages=messages,
            temperature=float(os.getenv("TEMPERATURE", 0.0)),
            n=n,
            top_p=top_p,
            stop=stop,
            max_tokens=int(os.getenv("MAX_TOKENS", 10000)),
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            timeout=timeout,
        )
        salvar_resposta(response)
        return [choice.message.content for choice in response.choices]
    except openai.APIError as e:
        print(f"Erro na API: {e}")
        return []

def instructGPT_logprobs(prompt, temperature=0.5):
    try:
        response = client.completions.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=temperature,
            max_tokens=1,
            logprobs=1,
            echo=True,
            timeout=80,
        )
        return response.choices
    except openai.APIError as e:
        print(f"Erro na API: {e}")
        return []
