import os
import sys
import time
import logging
from termcolor import colored
import openai
import tqdm
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict



# class for calling OpenAI API and handling cache
class GptApi:
    def __init__(self, verbose=False):
        self.verbose = verbose

        if "OPENAI_AZURE_ENDPOINT" in os.environ:
            assert "OPENAI_AZURE_KEY" in os.environ, "OPENAI_AZURE_KEY not found in environment"

            # Azure API access
            self.client = openai.AzureOpenAI(
                api_key=os.environ["OPENAI_AZURE_KEY"],
                azure_endpoint=os.environ["OPENAI_AZURE_ENDPOINT"],
                api_version="2023-07-01-preview"
            )
        elif "OPENAI_API_KEY" in os.environ:
            # OpenAI API access
            self.client = openai.OpenAI(
                api_key=os.environ["OPENAI_API_KEY"]
            )
        else:
            raise Exception("OPENAI_API_KEY or OPENAI_AZURE_KEY not found in environment")

        logging.getLogger().setLevel(logging.CRITICAL)  # in order to suppress all these HTTP INFO log messages

    # answer_id is used for determining if it was the top answer or how deep in the list it was
    def request(self, prompt, model, parse_response, temperature=0, answer_id=-1, cache=None, max_tokens=None):
        request = {"model": model, "temperature": temperature, "prompt": prompt}

        if request in cache and cache[request] is not None and len(cache[request]) > 0:
            answers = cache[request]
        else:
            answers = self.request_api(prompt, model, temperature, max_tokens)
            cache[request] = answers

        # there is no valid answer
        if len(answers) == 0:
            return [{
                    "temperature": temperature,
                    "answer_id": answer_id,
                    "answer": None,
                    "prompt": prompt,
                    "finish_reason": None,
                    "model": model,
                    }]

        parsed_answers = []
        for full_answer in answers:
            finish_reason = full_answer["finish_reason"]
            full_answer = full_answer["answer"]

            if finish_reason != "stop":
                print(f"No valid answer, giving score 0")
                errors = defaultdict(list)
                errors["critical"].append("Judge errored, giving answer score 0.")
                parsed_answers.append({
                    "temperature": temperature,
                    "answer_id": answer_id,
                    "answer": 0,
                    "errors": errors,
                    "prompt": prompt,
                    "finish_reason": finish_reason,
                    "model": model,
                })
                continue

            answer_id += 1
            answer = parse_response(full_answer)
            if isinstance(answer, tuple):
                answer, errors = answer
            else:
                errors = None
            if self.verbose or temperature > 0:
                print(f"Answer (t={temperature}): " + colored(answer, "yellow") + " (" + colored(full_answer, "blue") + ")", file=sys.stderr)
            if answer is None:
                continue
            parsed_answers.append({
                "temperature": temperature,
                "answer_id": answer_id,
                "answer": answer,
                "errors": errors,
                "prompt": prompt,
                "finish_reason": finish_reason,
                "model": model,
            })

        # there was no valid answer, increase temperature and try again
        if len(parsed_answers) == 0:
            print(f"No valid answer, increasing temperature to {temperature + 1} and trying again")
            return self.request(prompt, model, parse_response, temperature=temperature + 1, answer_id=answer_id, cache=cache)

        return parsed_answers

    def request_api(self, prompt, model, temperature=0, max_tokens=None):
        if temperature > 10:
            return [{"answer": None, "finish_reason": "error"}]
            
        # Add maximum token limit
        MAX_TOKENS_LIMIT = 4000  # Adjust this based on your model's context window
        if max_tokens and max_tokens > MAX_TOKENS_LIMIT:
            print(f"Reached maximum token limit of {MAX_TOKENS_LIMIT}", file=sys.stderr)
            return [{"answer": None, "finish_reason": "length"}]

        while True:
            try:
                response = self.call_api(prompt, model, temperature, max_tokens)
                break
            except Exception as e:
                # response was filtered
                if hasattr(e, 'code'):
                    if e.code == 'content_filter':
                        return [{"answer": None, "finish_reason": "filter"}]
                    print(e.code, file=sys.stderr)
                if hasattr(e, 'error') and e.error['code'] == 'invalid_model_output':
                    return [{"answer": None, "finish_reason": "invalid"}]

                # frequent error is reaching the API limit
                print(colored("Error, retrying...", "red"), file=sys.stderr)
                print(e, file=sys.stderr)
                time.sleep(1)

        answers = []
        for choice in response.choices:
            if choice.message.content is None:
                return [{"answer": None, "finish_reason": "invalid"}]
            if hasattr(choice, "message"):
                answer = choice.message.content.strip()
            else:
                answer = choice.text.strip()
                
            # one of the responses didn't finish, we need to request more tokens
            if choice.finish_reason != "stop":
                if self.verbose:
                    print(colored(f"Increasing max tokens to fit answers.", "red") + colored(answer, "blue"), file=sys.stderr)
                if max_tokens is None:
                    max_tokens = 500  # Set initial max_tokens if None
                new_max_tokens = max_tokens * 2
                print(f"Finish reason: {choice.finish_reason}, increasing max tokens to {new_max_tokens}", file=sys.stderr)
                if new_max_tokens > MAX_TOKENS_LIMIT:
                    print(f"Would exceed maximum token limit of {MAX_TOKENS_LIMIT}", file=sys.stderr)
                    return [{"answer": None, "finish_reason": choice.finish_reason}]
                return self.request_api(prompt, model, temperature=temperature, max_tokens=new_max_tokens)

            answers.append({
                "answer": answer,
                "finish_reason": choice.finish_reason,
            })

        if len(answers) > 1:
            # remove duplicate answers
            answers = [dict(t) for t in {tuple(d.items()) for d in answers}]

        return answers

    def call_api(self, prompt, model, temperature, max_tokens):
        parameters = {
            "temperature": temperature/10,
            "top_p": 1,
            "n": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
            "model": model
        }

        if max_tokens is not None:
            parameters["max_tokens"] = max_tokens

        if isinstance(prompt, list):
            # check that prompt contain list of dictionaries with role and content
            assert all(isinstance(p, dict) for p in prompt), "Prompts must be a list of dictionaries."
            assert all("role" in p and "content" in p for p in prompt), "Prompts must be a list of dictionaries with role and content."

            parameters["messages"] = prompt
        else:
            parameters["messages"] = [{
                "role": "user",
                "content": prompt,
            }]

        return self.client.chat.completions.create(**parameters)
    
    def bulk_request(self, df, model, parse_mqm_answer, cache, max_tokens=None):
        answers = []
        with ThreadPoolExecutor(100) as executor:
            futures = [
                executor.submit(self.request, row["prompt"], model, parse_mqm_answer, cache=cache, max_tokens=max_tokens)
                for _, row in df.iterrows()
            ]
            
            for future in tqdm.tqdm(futures, total=len(df), file=sys.stderr):
                answers += future.result()
                
        return answers
