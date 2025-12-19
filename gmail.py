import os
from datetime import datetime

from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("math_server")

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

# flow = InstalledAppFlow.from_client_secrets_file(
#     "credentials.json",
#     SCOPES
# )
# creds = flow.run_local_server(port=0)

# with open("token.json", "w") as token:
#     token.write(creds.to_json())
# print("Gmail token saved to token.json")
def get_credentials(scopes=SCOPES) -> Credentials:
    """Load or refresh credentials; start local auth if missing."""
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", scopes)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", scopes)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return creds

@mcp.tool()
def read_emails(query: str, max_results: int = 5) -> str:
    """
    Reads emails from the user's Gmail account based on a search query.

    Args:
        query (str): The search query to filter emails.
        max_results (int): The maximum number of emails to retrieve.

    Returns:
        str: A formatted string containing the email subjects and snippets.
    """
    creds = get_credentials(SCOPES)
    service = build("gmail", "v1", credentials=creds, cache_discovery=False)

    results = service.users().messages().list(userId="me", labelIds=["INBOX"],q=query, maxResults=max_results).execute()
    messages = results.get("messages", [])

    email_summaries = []
    for msg in messages:
        msg_data = service.users().messages().get(userId="me", id=msg["id"], format="metadata", metadataHeaders=["Subject"]).execute()
        headers = msg_data.get("payload", {}).get("headers", [])
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "No Subject")
        snippet = msg_data.get("snippet", "")
        email_summaries.append(f"Subject: {subject}\nSnippet: {snippet}\n")

    return "\n".join(email_summaries) if email_summaries else "No emails found."

def today_email(max_results=10):
    """
    Reads only today's emails from Gmail inbox.
    """
    
    creds = get_credentials(SCOPES)
    service = build("gmail", "v1", credentials=creds, cache_discovery=False)

    # Gmail search: newer_than:1d captures last 24h; works better than exact date.
    query = "in:inbox newer_than:1d"
    results = service.users().messages().list(
        userId="me",
        q=query,
        maxResults=max_results
    ).execute()
    messages = results.get("messages", [])

    emails = []

    for msg in messages:
        msg_data = service.users().messages().get(
            userId="me",
            id=msg["id"],
            format="metadata",
            metadataHeaders=["From", "Subject", "Date"]
        ).execute()

        headers = msg_data["payload"]["headers"]
        email_info = {h["name"]: h["value"] for h in headers}

        emails.append({
            "id": msg["id"],
            "from": email_info.get("From"),
            "subject": email_info.get("Subject"),
            "date": email_info.get("Date")
        })

    return emails

@mcp.tool()
def send_email(recipient: str, subject: str, body: str) -> str:
    """
    Sends an email using the user's Gmail account.

    Args:
        recipient (str): The email address of the recipient.
        subject (str): The subject of the email.
        body (str): The body content of the email.

    Returns:
        str: A confirmation message indicating the email was sent.
    """
    from email.mime.text import MIMEText
    import base64

    creds = get_credentials(SCOPES)
    service = build("gmail", "v1", credentials=creds, cache_discovery=False)

    message = MIMEText(body)
    message["to"] = recipient
    message["subject"] = subject

    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    message_body = {"raw": raw_message}

    sent_message = service.users().messages().send(userId="me", body=message_body).execute()
    print(f'Message Id: {sent_message["id"]} sent to {recipient}')

    return f"Email sent to {recipient} with Message ID: {sent_message['id']}"

# if __name__ == "__main__":
#     #send_email("pandasobhan22@gmail.com", "Test Email from MCP", "This is a test email sent using the Gmail API via MCP.")
#     today_emails = today_email()
#     if not today_emails:
#         print("No emails in the last 24 hours.")
#     else:
#         for email in today_emails:
#             print(f"From: {email['from']}, Subject: {email['subject']}, Date: {email['date']}")

if __name__ == "__main__":
    mcp.run(transport="stdio")