#!/usr/bin/env python3
"""
Debug script to see what the LLM is returning.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load keys silently
keys_path = Path.home() / ".devease" / "keys"
env_vars = {}
with open(keys_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' in line:
            key, value = line.split('=', 1)
            env_vars[key.strip()] = value.strip()

api_key = env_vars.get('OPENAI_API_KEY', '')
org_id = env_vars.get('OPENAI_ORG_ID', '')

print("Testing LLM response parsing...\n")

from openai import OpenAI
from src.growth.twitter.agents.content_agent import ContentAgent
from src.growth.twitter.core.types import TwitterKeys, TwitterConfig, TwitterTask, TaskType

# Initialize client
client = OpenAI(api_key=api_key, organization=org_id)

# Create task
task = TwitterTask(
    id="debug_test",
    type=TaskType.POST,
    idea="分享今天在广告投放中发现的一个有趣模式：提高ROAS的3个反直觉技巧",
    style="犀利吐槽，硬核数据"
)

# Initialize agent to get prompts
keys = TwitterKeys(openai_api_key=api_key, openai_org_id=org_id)
config = TwitterConfig(llm_model="gpt-4o")
agent = ContentAgent(keys, config)

# Build the prompt
golden_examples = agent._load_golden_examples()
prompt = agent._build_prompt(task, None, golden_examples)

print("=" * 80)
print("PROMPT SENT TO LLM:")
print("=" * 80)
print(prompt[:500] + "...")
print("\n" + "=" * 80)
print("LLM RESPONSE:")
print("=" * 80)

# Call API
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": agent._get_system_prompt()},
        {"role": "user", "content": prompt}
    ],
    temperature=0.8,
    max_tokens=1000
)

content = response.choices[0].message.content
print(content)
print("\n" + "=" * 80)
print(f"Response length: {len(content)} characters")
print("=" * 80)

# Try to parse it
try:
    import json
    parsed = json.loads(content)
    print("\n✓ JSON parsing successful!")
    print(f"Keys in JSON: {list(parsed.keys())}")
except json.JSONDecodeError as e:
    print(f"\n❌ JSON parsing failed: {e}")
    print(f"Error position: {e.pos if hasattr(e, 'pos') else 'unknown'}")

    # Show what's around the error position
    if hasattr(e, 'pos') and e.pos:
        start = max(0, e.pos - 50)
        end = min(len(content), e.pos + 50)
        print(f"Context around error: ...{content[start:end]}...")
