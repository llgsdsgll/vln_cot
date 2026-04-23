import os
import json
from tqdm import tqdm
import argparse
from openai import OpenAI
from ram.utils.openset_utils import openimages_rare_unseen

DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_TEXT_MODEL = "qwen3.6-plus"

parser = argparse.ArgumentParser(
    description='Generate LLM tag descriptions for RAM++ open-set recognition')
parser.add_argument('--api_key',
                    default=os.getenv('DASHSCOPE_API_KEY'))
parser.add_argument('--llm_model',
                    default=os.getenv('NAVGEN_LLM_MODEL', DEFAULT_TEXT_MODEL))
parser.add_argument('--llm_base_url',
                    default=os.getenv('DASHSCOPE_BASE_URL', DEFAULT_DASHSCOPE_BASE_URL))
parser.add_argument('--output_file_path',
                    help='save path of llm tag descriptions',
                    default='datasets/openimages_rare_200/openimages_rare_200_llm_tag_descriptions.json')


def analyze_tags(client, model, tag):
    # Generate LLM tag descriptions

    llm_prompts = [ f"Describe concisely what a(n) {tag} looks like:", \
                    f"How can you identify a(n) {tag} concisely?", \
                    f"What does a(n) {tag} look like concisely?",\
                    f"What are the identifying characteristics of a(n) {tag}:", \
                    f"Please provide a concise description of the visual characteristics of {tag}:"]

    results = {}
    result_lines = []

    result_lines.append(f"a photo of a {tag}.")

    for llm_prompt in tqdm(llm_prompts):

        # send message
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": llm_prompt}],
            max_tokens=77,
            temperature=0.99,
            n=10,
        )

        # parse the response
        for item in response.choices:
            result_lines.append(item.message.content.strip())
        results[tag] = result_lines
    return results


if __name__ == "__main__":

    args = parser.parse_args()

    if not args.api_key:
        raise RuntimeError("Missing API key. Set DASHSCOPE_API_KEY or pass --api_key.")

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.llm_base_url,
    )

    categories = openimages_rare_unseen

    tag_descriptions = []

    for tag in categories:
        result = analyze_tags(client, args.llm_model, tag)
        tag_descriptions.append(result)

    output_file_path = args.output_file_path

    with open(output_file_path, 'w') as w:
        json.dump(tag_descriptions, w, indent=3)
