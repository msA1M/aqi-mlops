#!/usr/bin/env python3
"""
Test script for AQI prediction endpoint
Usage: python test_aqi_endpoint.py
"""

import requests
import json

API_URL = "http://127.0.0.1:8000"

# Load test data
with open("test_aqi_data.json", "r") as f:
    test_data = json.load(f)

print("🧪 Testing AQI Prediction Endpoint\n")
print("=" * 60)

for test_name, payload in test_data.items():
    print(f"\n📊 {test_name.upper().replace('_', ' ')}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(f"{API_URL}/predict/aqi", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Status: {response.status_code}")
            print(f"   Predicted AQI: {result['predicted_aqi']}")
            print(f"   Category: {result['category']}")
            print(f"   Alert Level: {result['alert_level']}")
        else:
            print(f"❌ Status: {response.status_code}")
            print(f"   Error: {response.text}")
    
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
    
    print("-" * 60)

print("\n✨ Testing complete!")

