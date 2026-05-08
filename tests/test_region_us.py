import os
import time
import vertexai
from vertexai.generative_models import GenerativeModel

location = os.getenv("GOOGLE_CLOUD_LOCATION")
project = os.getenv("GOOGLE_CLOUD_PROJECT")

print(f"Testing Vertex AI in {location}...")
start = time.time()

try:
    vertexai.init(project=project, location=location)
    model = GenerativeModel("gemini-2.0-flash")
    response = model.generate_content("Reply: US_REGION_OK")

    elapsed = time.time() - start
    print(f"✅ Success! Response: {response.text.strip()}")
    print(f"⏱️  Round-trip time: {elapsed:.2f} seconds")

    if elapsed > 3.0:
        print("⚠️  Note: Higher latency expected for US region from HK")

except Exception as e:
    print(f"❌ Error: {e}")
