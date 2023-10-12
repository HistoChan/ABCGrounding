import torch
from tenacity import retry, stop_after_attempt, wait_fixed, wait_random
import time
from typing import Tuple, Union, List
from utils import join_path

# put you GPT API key here
GPTKEY = ""


class LocalLLM:
    def __init__(self, name: str, word_limit: int) -> None:
        self.name = name
        self.word_limit = word_limit
        can_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda") if can_cuda else torch.device("cpu")
        print(f"Cuda use is {can_cuda} and current device is {self.device}")


class ScaffoldLLM(LocalLLM):
    def __init__(self) -> None:
        super().__init__("Empty LLM", 0)

    def get_text(self, query: str = "") -> Tuple[str, float]:
        t = time.time()
        response = "answer"
        response_t = round(time.time() - t, 6)
        return response, response_t


class Dolly2(LocalLLM):
    def __init__(
        self, home_dir: str, model_name: str = "../", word_limit: int = 5
    ) -> None:
        from transformers import pipeline

        super().__init__(model_name, word_limit)
        self.pipeline = pipeline(
            model=join_path([home_dir, model_name]),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            max_new_tokens=word_limit,
        )

    def get_text(self, query: str = "") -> Tuple[str, float]:
        t = time.time()
        response = self.pipeline(query)
        response_t = round(time.time() - t, 6)
        return response[0]["generated_text"], response_t


class GPT:
    def __init__(self, model_type="gpt-3.5-turbo", word_limit: int = 5) -> None:
        assert model_type in ["gpt-3.5-turbo", "gpt-4"]
        self.name = model_type
        self.total_tokens = 0  # for record
        self.api_key = GPTKEY
        self.word_limit = word_limit

    @retry(wait=wait_fixed(60) + wait_random(0, 2), stop=stop_after_attempt(10))
    def get_text(self, query: str = "", ans_num=1) -> List[str]:
        import openai

        openai.api_key = self.api_key
        t = time.time()

        response = openai.ChatCompletion.create(
            model=self.name,
            messages=[
                {"role": "system", "content": "FOL assistant."},
                {"role": "user", "content": query},
            ],
            temperature=1,
            n=ans_num,
            max_tokens=self.word_limit,
        )
        self.total_tokens += response["usage"]["total_tokens"]
        return [choice["message"]["content"] for choice in response["choices"]], round(
            time.time() - t, 6
        )


class OpenLlama(LocalLLM):
    def __init__(
        self, home_dir: str, model_name: str = "../", word_limit: int = 5
    ) -> None:
        from transformers import LlamaTokenizer, LlamaForCausalLM

        model_dir = join_path([home_dir, model_name])
        self.tokenizer = LlamaTokenizer.from_pretrained(model_dir)
        self.model = LlamaForCausalLM.from_pretrained(
            model_dir, torch_dtype="auto", device_map="auto"
        )
        super().__init__(model_name, word_limit)

    @torch.no_grad()
    def get_text(self, query: str = "") -> str:
        query = "Q: " + query + "\nA:"
        t = time.time()
        input_ids = self.tokenizer(query, return_tensors="pt").input_ids.to(self.device)
        generation_output = self.model.generate(
            input_ids=input_ids, max_new_tokens=self.word_limit
        )
        response = self.tokenizer.decode(generation_output[0])
        response_t = round(time.time() - t, 6)
        # remove the question
        return response.split("\nA:")[1].strip(), response_t


class T5(LocalLLM):
    def __init__(
        self, home_dir: str, model_name: str = "../", word_limit: int = 5
    ) -> None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        model_dir = join_path([home_dir, model_name])
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_dir, torch_dtype="auto", device_map="auto"
        )
        super().__init__(model_name, word_limit)

    @torch.no_grad()
    def get_text(self, query: str = "") -> str:
        t = time.time()
        input_ids = self.tokenizer(query, return_tensors="pt").input_ids.to(self.device)
        generation_output = self.model.generate(
            input_ids=input_ids, max_new_tokens=self.word_limit
        )
        response = self.tokenizer.decode(generation_output[0], skip_special_tokens=True)
        response_t = round(time.time() - t, 6)
        return response, response_t


# load model
def setup_LLM(
    home_dir: str, model_name: str, word_limit: int
) -> Tuple[Union[Dolly2, OpenLlama, T5, GPT, ScaffoldLLM], float]:
    if model_name.startswith("dolly-v2"):
        t = time.time()
        model = Dolly2(home_dir, model_name, word_limit)
    elif model_name.startswith("open_llama"):
        t = time.time()
        model = OpenLlama(home_dir, model_name, word_limit)
    elif model_name.startswith("t5"):
        t = time.time()
        model = T5(home_dir, model_name, word_limit)
    elif model_name.startswith("gpt"):
        t = time.time()
        model = GPT(model_name, word_limit)
    else:
        t = time.time()
        model = ScaffoldLLM()
        print("Model setup error!")
    print(f"Now using: {model.name}")
    setup_time = round(time.time() - t, 6)
    return model, setup_time