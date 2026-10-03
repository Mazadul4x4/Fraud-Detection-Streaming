"""Load test for the scoring API (Locust). Not collected by pytest (name matches neither test_*.py nor *_test.py).

Each simulated user sends realistic transactions to POST /score as fast as possible.
Synthetic card numbers start with "LT" so real card histories in Redis are untouched;
remove them afterwards with the clean-up command in the README.

Single user (pure latency):
    locust -f tests/locustfile.py --host http://localhost:8000 --headless -u 1 -r 1 -t 60s --csv reports/load_1user
20 concurrent users (throughput under load):
    locust -f tests/locustfile.py --host http://localhost:8000 --headless -u 20 -r 5 -t 60s --csv reports/load_20users
"""

import random
from datetime import datetime, timedelta

from locust import HttpUser, between, task

CATEGORIES = [
    "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
    "health_fitness", "home", "kids_pets", "misc_net", "misc_pos",
    "personal_care", "shopping_net", "shopping_pos", "travel",
]
N_CARDS = 1000
START = datetime(2021, 1, 1)


class Cardholder(HttpUser):
    wait_time = between(0, 0.01)  # almost no pause: measure what the API can sustain

    def on_start(self):
        self.clock = START + timedelta(days=random.randint(0, 30))
        self.counter = 0

    @task
    def score(self):
        self.counter += 1
        self.clock += timedelta(seconds=random.randint(30, 3600))
        payload = {
            "transaction_id": f"lt-{id(self)}-{self.counter}",
            "cc_num": f"LT{random.randrange(N_CARDS):06d}",
            "amount": round(random.lognormvariate(3.8, 1.0), 2),
            "category": random.choice(CATEGORIES),
            "timestamp": self.clock.isoformat(timespec="seconds"),
            "customer_dob": "1980-05-17",
            "customer_lat": 40.71,
            "customer_long": -74.0,
            "merchant_lat": round(40.71 + random.uniform(-1, 1), 4),
            "merchant_long": round(-74.0 + random.uniform(-1, 1), 4),
            "city_pop": random.choice([1_000, 50_000, 1_000_000]),
        }
        with self.client.post("/score", json=payload, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
