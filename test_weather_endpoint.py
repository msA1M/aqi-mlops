#!/usr/bin/env python3
"""
Test script for Weather prediction endpoint
Usage: python test_weather_endpoint.py
"""

import requests
import json
import os

API_URL = "http://127.0.0.1:8000"

# Load test cities from JSON file if it exists, otherwise use default list
if os.path.exists("test_weather_data.json"):
    with open("test_weather_data.json", "r") as f:
        data = json.load(f)
        test_cities = data.get("cities", [])
else:
    # Default test cities
    test_cities = [
        "Brasilia",
        "Karachi",
        "Lahore",
        "Islamabad",
        "Delhi",
        "Mumbai",
        "New York",
        "London",
        "Paris",
        "Tokyo"
    ]

print("🌦️ Testing Weather Prediction Endpoint\n")
print("=" * 60)

for city in test_cities:
    print(f"\n📍 Testing City: {city}")
    
    try:
        response = requests.get(
            f"{API_URL}/predict/weather",
            params={"city": city}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Status: {response.status_code}")
            print(f"   City: {result['city']}")
            print(f"   Predictions:")
            for key, value in result['predictions'].items():
                print(f"     - {key}: {value:.2f}")
        else:
            print(f"❌ Status: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"   Error: {error_detail.get('detail', response.text)}")
            except:
                print(f"   Error: {response.text}")
    
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
    
    print("-" * 60)

print("\n✨ Testing complete!")

