import ipdb
import pandas as pd
import diskcache as dc
from gemba.gpt_api import GptApi
from gemba.gemba_mqm_utils import TEMPLATE_GEMBA_MQM, apply_template, parse_mqm_answer
from gemba.gemba_esa import TEMPLATE_GEMBA_ESA_ERROR_SPANS, TEMPLATE_GEMBA_ESA_RANKING
from gemba.prompt import prompts, validate_number


def get_gemba_scores(source, hypothesis, source_lang, target_lang, method="GEMBA-MQM_norm", model="gpt-4o", cache_dir=".cache"):
    """Get GEMBA scores for machine translation evaluation.
    
    This function evaluates machine translation quality using various GEMBA methods by leveraging
    large language models to analyze source and translated text pairs.

    Args:
        source (list): List of source language segments to be evaluated
        hypothesis (list): List of target language translations to be evaluated
        source_lang (str): Source language code (e.g. 'en' for English)
        target_lang (str): Target language code (e.g. 'de' for German) 
        method (str): Evaluation method to use. One of:
            - "GEMBA-MQM": MQM-style error annotation and scoring
            - "GEMBA-MQM_norm": MQM-style error annotation and scoring with normalization
            - "GEMBA-DA": Direct assessment scoring
            - "GEMBA-DA_ref": Direct assessment with reference
            - "GEMBA-SQM": Scalar quality metrics
            - "GEMBA-SQM_ref": Scalar quality metrics with reference
            - "GEMBA-stars": Star rating evaluation
            - "GEMBA-stars_ref": Star rating with reference
            - "GEMBA-classes": Classification-based evaluation
            - "GEMBA-classes_ref": Classification with reference
            - "GEMBA-ESA": Error span annotation and ranking
        model (str): Name of the LLM model to use for evaluation
        cache_dir (str): Directory to store the cache in

    Returns:
        list: List of scores/evaluations for each source-hypothesis pair. The format depends on the method:
            - For MQM: Negative scores where higher is better (max 0, min -25)
            - For MQM_norm: Normalized scores to 0-100 range
            - For DA/SQM: Numeric scores
            - For stars: 1-5 star ratings
            - For classes: Classification labels
            - For ESA: Numeric rankings based on error spans
        list: List of error classes for each source-hypothesis pair. Only returned for MQM methods.

    The function uses disk caching to store results and avoid redundant API calls. Cache is stored
    in a '{cache_dir}/{model}_{method}' directory.
    """
    
    df = pd.DataFrame({'source_seg': source, 'target_seg': hypothesis})
    df['source_lang'] = source_lang
    df['target_lang'] = target_lang

    cache = dc.Cache(f'{cache_dir}/{model}_{method}', expire=None, size_limit=int(10e10), cull_limit=0, eviction_policy='none')
    gptapi = GptApi()

    if method in ["GEMBA-MQM", "GEMBA-MQM_norm"]:
        df["prompt"] = df.apply(lambda x: apply_template(TEMPLATE_GEMBA_MQM, x), axis=1)
        parse_answer = lambda x: parse_mqm_answer(x, list_mqm_errors=True, full_desc=True, normalize=method == "GEMBA-MQM_norm")
        answers = gptapi.bulk_request(df, model, parse_answer, cache=cache, max_tokens=500)
    elif method in ["GEMBA-DA", "GEMBA-DA_ref", "GEMBA-SQM", "GEMBA-SQM_ref", "GEMBA-stars", "GEMBA-stars_ref", "GEMBA-classes", "GEMBA-classes_ref"]:
        df["prompt"] = df.apply(lambda x: apply_template(prompts[method]['prompt'], x), axis=1)
        parse_answer = prompts[method]["validate_answer"]
        answers = gptapi.bulk_request(df, model, parse_answer, cache=cache, max_tokens=500)
    elif method == "GEMBA-ESA":
        df["prompt"] = df.apply(lambda x: apply_template(TEMPLATE_GEMBA_ESA_ERROR_SPANS, x), axis=1)
        parse_answer = lambda x: x
        error_spans = gptapi.bulk_request(df, model, parse_answer, cache=cache)
        df['error_spans'] = pd.DataFrame(error_spans)['answer']

        df["prompt"] = df.apply(lambda x: apply_template(TEMPLATE_GEMBA_ESA_RANKING, x), axis=1)
        parse_answer = validate_number
        answers = gptapi.bulk_request(df, model, parse_answer, cache=cache)
    else:
        raise Exception(f"Method {method} not supported.")
    
    df = pd.DataFrame(answers)
    return df['answer'].tolist(), df['errors'].tolist()
