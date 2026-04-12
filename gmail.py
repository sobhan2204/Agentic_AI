import os
import json
import base64
from email.mime.text import MIMEText
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("gmail")
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

def get_credentials() -> Credentials:
    creds = None
    
    # Strategy 1: token.json mounted as a volume at /app/token.json
    if os.path.exists("/app/token.json"):
        creds = Credentials.from_authorized_user_file("/app/token.json", SCOPES)
    
    # Strategy 2: token passed as env var (base64-encoded JSON string)
    elif os.getenv("GMAIL_TOKEN_JSON"):
        token_data = json.loads(
            base64.b64decode(os.getenv("GMAIL_TOKEN_JSON")).decode()
        )
        creds = Credentials.from_authorized_user_info(token_data, SCOPES)
    
    if not creds:
        raise RuntimeError(
            "Gmail credentials not found. Either:\n"
            "  1. Mount token.json: -v /path/to/token.json:/app/token.json\n"
            "  2. Set GMAIL_TOKEN_JSON env var with base64-encoded token"
        )
    
    # Refresh if expired — works headlessly, no browser needed
    if not creds.valid and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        # Persist refreshed token back to mounted file if possible
        if os.path.exists("/app/token.json"):
            with open("/app/token.json", "w") as f:
                f.write(creds.to_json())
    
    if not creds.valid:
        raise RuntimeError(
            "Gmail token is invalid and cannot be refreshed. "
            "Re-run the OAuth flow locally to get a fresh token.json, then remount it."
        )
    
    return creds


@mcp.tool()
def send_email(recipient: str, subject: str, body: str) -> str:
    """Sends an email using the user's Gmail account."""
    try:
        creds = get_credentials()
        service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        message = MIMEText(body)
        message["to"] = recipient
        message["subject"] = subject
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        sent = service.users().messages().send(
            userId="me", body={"raw": raw_message}
        ).execute()
        return f"Email sent to {recipient} with Message ID: {sent['id']}"
    except RuntimeError as e:
        return f"Gmail auth error: {e}"
    except Exception as e:
        return f"Failed to send email: {type(e).__name__}: {str(e)[:200]}"


@mcp.tool()
def read_emails(query: str, max_results: int = 5) -> str:
    """Reads emails from Gmail based on a search query."""
    try:
        creds = get_credentials()
        service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        results = service.users().messages().list(
            userId="me", labelIds=["INBOX"], q=query, maxResults=max_results
        ).execute()
        messages = results.get("messages", [])
        summaries = []
        for msg in messages:
            msg_data = service.users().messages().get(
                userId="me", id=msg["id"], format="metadata",
                metadataHeaders=["Subject"]
            ).execute()
            headers = msg_data.get("payload", {}).get("headers", [])
            subject = next((h["value"] for h in headers if h["name"] == "Subject"), "No Subject")
            snippet = msg_data.get("snippet", "")
            summaries.append(f"Subject: {subject}\nSnippet: {snippet}")
        return "\n\n".join(summaries) if summaries else "No emails found."
    except RuntimeError as e:
        return f"Gmail auth error: {e}"
    except Exception as e:
        return f"Failed to read emails: {type(e).__name__}: {str(e)[:200]}"


if __name__ == "__main__":
    mcp.run(transport="stdio")