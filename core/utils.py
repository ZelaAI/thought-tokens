import time

class TokensPerSecondTimer:
    def __init__(self, tokens_per_call: int = 1):
        self.last_time = time.time()
        self.call_count = 0
        self.running_average = -1
        self.tokens_per_call = tokens_per_call

    def __call__(self):
        gap = time.time() - self.last_time
        tokens_per_second = self.tokens_per_call / gap
        if self.call_count > 3:
            # Rolling average
            self.running_average = self.running_average * 0.9 + tokens_per_second * 0.1
        elif self.call_count == 3:
            # Set baseline
            self.running_average = tokens_per_second
        
        self.last_time = time.time()
        self.call_count += 1
        
        return self.running_average        
