# GPT API
import base64
import os

from openai import OpenAI


DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_TEXT_MODEL = "qwen3.6-plus"


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def prompt_make(prompt_path, ex_prompt):    # set prompt
    with open(prompt_path, "r", encoding='utf-8') as f:  # open the file
        txt = f.readlines()
        prompt_system = txt[1]
        prompt = txt[3]  # read the user line
        if len(txt) > 4:
            for i in range(4, len(txt)):
                prompt = prompt + txt[i]
        prompt = prompt + ex_prompt
        # print(prompt_system)
        # print(prompt)
        return prompt_system, prompt


def _resolve_api_key(args):
    api_key = getattr(args, "API_KEY", None) or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set DASHSCOPE_API_KEY in the environment or pass --API_KEY."
        )
    return api_key


def _resolve_base_url(args):
    return getattr(args, "llm_base_url", None) or os.getenv("DASHSCOPE_BASE_URL") or DEFAULT_DASHSCOPE_BASE_URL


def _resolve_text_model(args):
    return getattr(args, "llm_model", None) or os.getenv("NAVGEN_LLM_MODEL") or DEFAULT_TEXT_MODEL


def _resolve_vision_model(args):
    return getattr(args, "vlm_model", None) or os.getenv("NAVGEN_VLM_MODEL") or _resolve_text_model(args)


def _make_client(args):
    return OpenAI(
        api_key=_resolve_api_key(args),
        base_url=_resolve_base_url(args),
    )


def _extract_message_text(message):
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_list = []
        for item in content:
            if isinstance(item, str):
                text_list.append(item)
                continue
            if isinstance(item, dict) and item.get("text"):
                text_list.append(item["text"])
        return "\n".join(text_list).strip()
    return str(content)


def _chat_completion(args, model, messages):
    client = _make_client(args)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    if not response.choices:
        raise RuntimeError("The model returned no choices.")
    return _extract_message_text(response.choices[0].message)


def gpt4o(args, prompt_path, ex_prompt):   # It is gpt4-o
    prompt_system, prompt = prompt_make(prompt_path, ex_prompt)
    return _chat_completion(
        args,
        _resolve_text_model(args),
        [
            {
                "role": "system",
                "content": prompt_system
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
    )


def gpt4o_mini(args, prompt_path, ex_prompt):  # It is -o-mini
    prompt_system, prompt = prompt_make(prompt_path, ex_prompt)
    return _chat_completion(
        args,
        _resolve_text_model(args),
        [
            {
                "role": "system",
                "content": prompt_system
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
    )


def gpt4_vision(args, prompt_path, ex_prompt, img_path):  # VLM
    base64_image = encode_image(img_path)
    prompt_system, prompt = prompt_make(prompt_path, ex_prompt)
    return _chat_completion(
        args,
        _resolve_vision_model(args),
        [
            {"role": "system",
             "content": prompt_system
             },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
    )
