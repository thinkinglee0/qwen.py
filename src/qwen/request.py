import asyncio


class ModelRequest:
    def __init__(self, loop, input_ids: list[int], sampling, max_new_tokens):
        self.input_ids = input_ids
        self.sampling = sampling
        self.max_new_tokens = max_new_tokens
        self.loop = loop

        self.token_queue: asyncio.Queue[int | None | Exception] = asyncio.Queue()
        self.finished = False
