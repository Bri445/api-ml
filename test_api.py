"""
Simple test script for the emoji sentiment API
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n=== Testing /health endpoint ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_analyze_get():
    """Test GET request to /analyze_request (should fail with 405)"""
    print("\n=== Testing GET /analyze_request (should fail) ===")
    response = requests.get(f"{BASE_URL}/analyze_request")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 405

def test_analyze_post():
    """Test POST request to /analyze_request"""
    print("\n=== Testing POST /analyze_request ===")
    data = {
        "text": "I absolutely love this product! It's amazing and works great! 😊"
    }
    response = requests.post(
        f"{BASE_URL}/analyze_request",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_analyze_empty():
    """Test POST with empty text"""
    print("\n=== Testing POST /analyze_request with empty text ===")
    data = {
        "text": ""
    }
    response = requests.post(
        f"{BASE_URL}/analyze_request",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 400

def test_predict():
    """Test /predict endpoint"""
    print("\n=== Testing POST /predict ===")
    data = {
        "text": "This is terrible 😠"
    }
    response = requests.post(
        f"{BASE_URL}/predict",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

if __name__ == "__main__":
    print("Starting API tests...")
    print("Make sure the API is running at http://localhost:8000")
    
    tests = [
        ("Health Check", test_health),
        ("GET /analyze_request", test_analyze_get),
        ("POST /predict", test_predict),
        ("Empty text validation", test_analyze_empty),
        ("POST /analyze_request", test_analyze_post),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"ERROR: {e}")
            results.append((name, False))
    
    print("\n" + "="*50)
    print("TEST RESULTS:")
    print("="*50)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
