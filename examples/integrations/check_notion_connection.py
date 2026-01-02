import os
import sys
from typing import Any

# Try to load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # If dotenv is not installed, we rely on the user sourcing the .env file
    pass

try:
    from notion_client import Client
except ImportError:
    print("Error: 'notion-client' is not installed.")
    print("Run: pip install notion-client")
    sys.exit(1)


def get_page_title(page: dict[str, Any]) -> str:
    """Extract title from Notion page properties."""
    properties = page.get("properties", {})
    for prop in properties.values():
        if prop.get("type") == "title":
            title_list = prop.get("title", [])
            if title_list:
                return title_list[0].get("plain_text", "Untitled")
    return "Untitled"


def main():
    """List recent Notion pages to verify connection."""
    api_key = os.environ.get("NOTION_API_KEY")

    if not api_key:
        print("❌ Error: NOTION_API_KEY environment variable is not set.")
        print("Please ensure .env file exists or export the variable.")
        return

    print(f"🔄 Connecting to Notion with key: {api_key[:4]}...{api_key[-4:]}")

    try:
        client = Client(auth=api_key)
        # Search for pages (empty query returns recently accessed)
        response = client.search(
            filter={"value": "page", "property": "object"}, page_size=5
        )

        results = response.get("results", [])

        if not results:
            print("✅ Connected, but no pages found accessible to this integration.")
            print(
                "Make sure you have shared specific pages with your integration in Notion."
            )
            return

        print(f"✅ Successfully connected! Found {len(results)} accessible pages:")
        for page in results:
            title = get_page_title(page)
            print(f"   - {title} (ID: {page['id']})")

    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")


if __name__ == "__main__":
    main()
