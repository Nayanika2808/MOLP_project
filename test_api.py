import unittest
import requests

BASE_URL = "http://127.0.0.1:5000"

class TestAPI(unittest.TestCase):
    def test_health(self):
        response = requests.get(f"{BASE_URL}/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "Healthy"})

    def test_predict_single(self):
        response = requests.post(f"{BASE_URL}/predict", json={"feature": 5.1})
        self.assertEqual(response.status_code, 200)
        self.assertIn("predictions", response.json())

    def test_predict_multiple(self):
        response = requests.post(f"{BASE_URL}/predict", json={"feature": [[5.1], [3.5]]})
        self.assertEqual(response.status_code, 200)
        self.assertIn("predictions", response.json())

if __name__ == "__main__":
    unittest.main()