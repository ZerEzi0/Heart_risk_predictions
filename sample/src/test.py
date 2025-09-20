import requests

def main():
    # Тест /health
    r = requests.get("http://127.0.0.1:8000/health")
    if r.status_code != 200:
        print(f"Error: {r.status_code}")
    else:
        print("Health check:", r.json())
    
    # Тест /predict_csv
    files = {'file': open('heart_test.csv', 'rb')}
    r = requests.post("http://127.0.0.1:8000/predict_csv", files=files)
    if r.status_code == 200:
        print("Predictions:", r.json())
    else:
        print(f"Error: {r.status_code}, {r.text}")

if __name__ == "__main__":
    main()