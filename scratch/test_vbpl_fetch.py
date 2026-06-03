import urllib.request
import json
import sys
import os

# Append backend path so we can import the parser
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))
from ingestion import LegalDocumentIngestion

def main():
    doc_id = "139877"
    url = f"https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/{doc_id}"
    print(f"Connecting to: {url}")
    
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            status_code = response.getcode()
            print(f"Response status code: {status_code}")
            
            payload_data = response.read().decode('utf-8')
            payload = json.loads(payload_data)
            
            print("Payload status:", payload.get("success"), payload.get("statusCode"))
            print("Payload message:", payload.get("message"))
            
            data = payload.get("data", {})
            print("Document Title:", data.get("title"))
            print("Document Type:", data.get("docType", {}).get("name"))
            print("Effective Status:", data.get("effStatus", {}).get("name"))
            print("Issue Date:", data.get("issueDate"))
            
            content_html = data.get("documentContent", {}).get("content", "")
            print(f"HTML Content length: {len(content_html)} characters")
            
            print("\nTriggering LegalDocumentIngestion.ingest_vbpl_payload()...")
            ingestor = LegalDocumentIngestion()
            total_items = ingestor.ingest_vbpl_payload(doc_id, payload)
            print(f"Ingestion succeeded! Processed {total_items} items.")
            
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
