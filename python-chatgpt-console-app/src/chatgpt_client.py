

from openai import OpenAI


MODEL = "meta-llama-3.1-8b-instruct"

class ChatGPTClient:

    def __init__(self, api_key=None):
        # LM Studio client initialization
        self.api_key = api_key or "lm-studio"
        self.client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key=self.api_key)


    def send_message(self, messages, stream=False, model=None):
        if model is None:
            model = MODEL
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream
        )
        return response

    def stream_response(self, messages, model=None):
        if model is None:
            model = MODEL
        response = self.send_message(messages, stream=True, model=model)
        collected_chunks = []
        for chunk in response:
            chunk_message = getattr(chunk.choices[0].delta, 'content', '')
            if chunk_message is None:
                chunk_message = ''
            collected_chunks.append(chunk_message)
        return ''.join([c if c is not None else '' for c in collected_chunks])

    def get_response(self, user_message, model=None):
        if model is None:
            model = MODEL
        messages = [{"role": "user", "content": user_message}]
        response = self.send_message(messages, model=model)



        return response.choices[0].message.content




