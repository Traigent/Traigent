"""
Minimal Bedrock 'hello world' with Titan Text Lite.
- Uses default AWS credentials/region (us-east-1).
- Prints the model output and token counts from response headers.
"""

import json
import os
import sys

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL_ID = "amazon.titan-text-lite-v1"  # cost-efficient, small context model

PROMPT = (
    "User: Say 'Hello, world! what is the capital of France?' and nothing else.\nBot:"
)


def main() -> int:
    try:
        client = boto3.client("bedrock-runtime", region_name=REGION)
    except Exception as e:
        print(f"Failed to create Bedrock Runtime client: {e}", file=sys.stderr)
        return 2

    payload = {
        "inputText": PROMPT,
        "textGenerationConfig": {"maxTokenCount": 64, "temperature": 0.0, "topP": 0.9},
    }

    try:
        resp = client.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )
    except NoCredentialsError:
        print(
            "No AWS credentials found. Run `aws sts get-caller-identity` first.",
            file=sys.stderr,
        )
        return 3
    except ClientError as e:
        print(f"AWS error: {e}", file=sys.stderr)
        return 4
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 5

    # Body
    body = json.loads(resp["body"].read())
    text = body["results"][0]["outputText"]
    print("=== Model output ===")
    print(text)

    # Token usage (Bedrock returns counts in HTTP headers)
    headers = resp.get("ResponseMetadata", {}).get("HTTPHeaders", {})
    prompt_tokens = int(headers.get("x-amzn-bedrock-input-token-count", 0))
    completion_tokens = int(headers.get("x-amzn-bedrock-output-token-count", 0))
    print("\n=== Usage ===")
    print(f"prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
