"""Locust baseline (T021). Run: locust -f locustfile.py --host http://localhost:8000"""
import random

from locust import HttpUser, between, task

SAMPLES = ["المنتج ممتاز وسريع التوصيل", "سيء جدا ولا انصح به", "عادي مش وحش ومش حلو"]


class ReviewUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def predict(self):
        self.client.post("/predict", json={"text": random.choice(SAMPLES)})
