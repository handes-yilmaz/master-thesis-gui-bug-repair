#!/usr/bin/env python
import os, json, argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROMPT_TEMPLATE = '''You are a senior frontend engineer. Given:
- Bug report: {bug_report}
- OCR text from screenshot: {ocr_text}
- (Optional) UI event sequence: {ui_events}
- (Optional) Code snippet: {code_snippet}

Task:
1) Identify the likely root cause in a React/Vue/Vanilla JS codebase.
2) Propose a minimal patch (diff or code block).
3) Explain how the patch fixes the user-visible issue.
4) Note any accessibility improvements (ARIA roles, labels, contrast).
Constraints:
- Keep changes minimal and safe.
- Follow best practices for state management and event handling.
'''

def call_openai(prompt, model, api_key):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return resp.choices[0].message.content

def call_huggingface(prompt, model, api_key):
    import requests
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": prompt}
    
    try:
        resp = requests.post(API_URL, headers=headers, json=payload)
        print(f"[DEBUG] Status code: {resp.status_code}")
        print(f"[DEBUG] Response headers: {resp.headers}")
        
        if resp.status_code != 200:
            print(f"[ERROR] API returned status {resp.status_code}: {resp.text}")
            return f"API Error: {resp.status_code} - {resp.text}"
        
        # Try to parse JSON response
        try:
            json_resp = resp.json()
            print(f"[DEBUG] JSON response: {json_resp}")
            
            if isinstance(json_resp, list) and len(json_resp) > 0:
                if "generated_text" in json_resp[0]:
                    return json_resp[0]["generated_text"]
                elif "text" in json_resp[0]:
                    return json_resp[0]["text"]
                else:
                    return str(json_resp[0])
            else:
                return str(json_resp)
                
        except json.JSONDecodeError as e:
            print(f"[DEBUG] Raw response text: {resp.text[:200]}...")
            return f"Response parsing error: {resp.text[:100]}..."
            
    except Exception as e:
        return f"Request error: {str(e)}"

def main(cfg_path, bug_report, ocr_text, ui_events, code_snippet):
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    prompt = PROMPT_TEMPLATE.format(
        bug_report=bug_report,
        ocr_text=ocr_text,
        ui_events=ui_events,
        code_snippet=code_snippet
    )

    provider = cfg["llm"].get("provider", "openai")
    model = cfg["llm"]["model"]
    api_key_env = cfg["llm"]["api_key_env"]
    api_key = os.getenv(api_key_env)

    if not api_key:
        print(f"[ERROR] Missing API key in env var: {api_key_env}")
        return

    print("=== PROMPT ===")
    print(prompt)
    print("\n=== RESPONSE ===")

    if provider == "openai":
        print(call_openai(prompt, model, api_key))
    elif provider == "huggingface":
        print(call_huggingface(prompt, model, api_key))
    else:
        print("[ERROR] Unknown provider")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.json")
    ap.add_argument("--bug_report", default="Example: Button not clickable on mobile.")
    ap.add_argument("--ocr_text", default="Login button visible; tooltip 'Submit form'.")
    ap.add_argument("--ui_events", default="tap(button#login) -> no navigation")
    ap.add_argument("--code_snippet", default="<button id='login' onClick={submit}>Login</button>")
    args = ap.parse_args()
    main(args.config, args.bug_report, args.ocr_text, args.ui_events, args.code_snippet)
