import tiktoken


class CostCalculator:
    def __init__(self):
        # OpenAI GPT-4 fiyatları ($ / 1K token)
        self.GPT4_INPUT_COST = 0.01
        self.GPT4_OUTPUT_COST = 0.03

        # Claude 3 Sonnet fiyatları ($ / 1K token)
        self.CLAUDE_INPUT_COST = 0.015
        self.CLAUDE_OUTPUT_COST = 0.075

        # Token sayacı başlat
        self.gpt4_encoder = tiktoken.encoding_for_model("gpt-4")

    def calculate_openai_cost(self, prompt_tokens, completion_tokens):
        """OpenAI GPT-4 maliyet hesaplaması"""
        input_cost = (prompt_tokens * self.GPT4_INPUT_COST) / 1000
        output_cost = (completion_tokens * self.GPT4_OUTPUT_COST) / 1000
        return input_cost + output_cost

    def calculate_claude_cost(self, input_tokens, output_tokens):
        """Claude 3 Sonnet maliyet hesaplaması"""
        input_cost = (input_tokens * self.CLAUDE_INPUT_COST) / 1000
        output_cost = (output_tokens * self.CLAUDE_OUTPUT_COST) / 1000
        return input_cost + output_cost

    def count_openai_tokens(self, text):
        """OpenAI için token sayısı hesapla"""
        return len(self.gpt4_encoder.encode(text))

    def estimate_cost(self, text, model="gpt-4"):
        """Verilen metin için tahmini maliyet hesapla"""
        if model == "gpt-4":
            tokens = self.count_openai_tokens(text)
            return self.calculate_openai_cost(tokens, 0)  # Sadece input maliyeti
        else:
            # Claude için anthropic.get_token_count() kullanılabilir
            # ancak bu fonksiyon henüz mevcut değil
            pass