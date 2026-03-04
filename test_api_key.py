import urllib.request
import json

# Testing with an available model 'gemini-2.5-flash'
url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=AIzaSyAh4zRGXRUy1NLuAkiMQ14_76Yyu5i3f6Y"

data = json.dumps({
    "contents": [{"parts": [{"text": "Say hello world"}]}]
}).encode('utf-8')

req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

try:
    with urllib.request.urlopen(req) as response:
        print("✅ SUCCESS: The API key is working with model gemini-2.5-flash.")
        print("Status code:", response.getcode())
        resp_data = json.loads(response.read().decode('utf-8'))
        
        # safely extract the text from the response structure
        try:
            text = resp_data['candidates'][0]['content']['parts'][0]['text']
            print("Model generated response:", text)
        except (KeyError, IndexError):
            print("Raw response JSON:", json.dumps(resp_data, indent=2))
            
except urllib.error.HTTPError as e:
    print("❌ ERROR: The API request failed.")
    print("HTTP Error code:", e.code)
    print("Reason:", e.reason)
    print("Response:", e.read().decode('utf-8'))
except Exception as e:
    print("An unexpected error occurred:")
    print(e)
