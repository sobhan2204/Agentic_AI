#!/usr/bin/env python3
"""
Gmail MCP Server - Integrable with existing MCP client
Save this as gmail.py
"""

import asyncio
import json
import signal
import sys
import pickle
import os.path
import base64
from typing import List, Dict, Any
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from mcp.server.fastmcp import FastMCP
from email.mime.text import MIMEText

# Initialize FastMCP server
mcp = FastMCP("gmail-mcp-server")

# Gmail API setup
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
CREDS_FILE = "client_secret.json"  # Relative path for better integration
TOKEN_FILE = "token.pickle"

# Global service instance
gmail_service = None

def get_gmail_service():
    """Authenticate and return Gmail API service."""
    global gmail_service
    if gmail_service is not None:
        return gmail_service
        
    print("Attempting to authenticate Gmail API...", file=sys.stderr)
    creds = None
    
    # Check for token file
    if os.path.exists(TOKEN_FILE):
        print(f"Loading existing token from {TOKEN_FILE}", file=sys.stderr)
        try:
            with open(TOKEN_FILE, "rb") as token:
                creds = pickle.load(token)
        except Exception as e:
            print(f"Error loading token: {e}", file=sys.stderr)
            creds = None
    
    # Authenticate if needed
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing expired token...", file=sys.stderr)
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"Error refreshing token: {e}", file=sys.stderr)
                creds = None
        
        if not creds:
            if not os.path.exists(CREDS_FILE):
                raise FileNotFoundError(f"Credentials file {CREDS_FILE} not found. Please download it from Google Cloud Console.")
            
            print(f"Running OAuth flow with {CREDS_FILE}", file=sys.stderr)
            flow = InstalledAppFlow.from_client_secrets_file(CREDS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        try:
            with open(TOKEN_FILE, "wb") as token:
                print(f"Saving token to {TOKEN_FILE}", file=sys.stderr)
                pickle.dump(creds, token)
        except Exception as e:
            print(f"Warning: Could not save token: {e}", file=sys.stderr)
    
    gmail_service = build("gmail", "v1", credentials=creds)
    print("Gmail service initialized successfully", file=sys.stderr)
    return gmail_service

@mcp.tool()
async def send_email(to: str, subject: str, body: str) -> str:
    """
    Send an email from your Gmail account
    
    Args:
        to: Recipient email address
        subject: Email subject line
        body: Email body content
    
    Returns:
        Success message with message ID or error message
    """
    print(f"Sending email to {to} with subject: {subject}", file=sys.stderr)
    try:
        service = get_gmail_service()
        
        # Create the email message
        message = MIMEText(body)
        message["to"] = to
        message["subject"] = subject
        
        # Encode the message
        #raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

        
        # Send the email
        result = service.users().messages().send(
            userId="me", 
            #body={"raw": raw}
        ).execute()
        
        success_msg = f"✅ Email sent successfully to {to}. Message ID: {result['id']}"
        print(success_msg, file=sys.stderr)
        return success_msg
        
    except Exception as e:
        error_msg = f" Failed to send email: {str(e)}"
        print(error_msg, file=sys.stderr)
        return error_msg

@mcp.tool()
async def search_emails(query: str, limit: int = 5) -> str:
    """
    Search emails in your Gmail account
    
    Args:
        query: Gmail search query (e.g., 'from:example@gmail.com', 'subject:important')
        limit: Maximum number of results to return (1-10, default: 5)
    
    Returns:
        JSON formatted list of matching emails
    """
    print(f"Searching emails with query: {query}", file=sys.stderr)
    limit = max(1, min(limit, 10))  # Ensure limit is between 1 and 10
    
    try:
        service = get_gmail_service()
        # Search for messages
        results = service.users().messages().list(
            userId="me", 
            q=query, 
            maxResults=limit
        ).execute()
        
        messages = results.get("messages", [])
        if not messages:
            return f"No emails found matching query: {query}"
        
        summaries = []
        for msg in messages:
            try:
                # Get message metadata
                msg_data = service.users().messages().get(
                    userId="me", 
                    id=msg["id"], 
                    format="metadata"
                ).execute()
                
                # Extract headers
                headers = {h["name"]: h["value"] for h in msg_data["payload"]["headers"]}
                
                summaries.append({
                    "id": msg["id"],
                    "from": headers.get("From", ""),
                    "to": headers.get("To", ""),
                    "subject": headers.get("Subject", ""),
                    "date": headers.get("Date", ""),
                    "snippet": msg_data.get("snippet", "")
                })
            except Exception as e:
                print(f"Error processing message {msg['id']}: {e}", file=sys.stderr)
                continue
        
        result = f"Found {len(summaries)} emails:\n" + json.dumps(summaries, indent=2)
        print(f"Search completed: {len(summaries)} emails found", file=sys.stderr)
        return result
        
    except Exception as e:
        error_msg = f" Failed to search emails: {str(e)}"
        print(error_msg, file=sys.stderr)
        return error_msg

@mcp.tool()
async def read_email(message_id: str) -> str:
    """
    Read a specific email by its ID
    
    Args:
        message_id: Gmail message ID
    
    Returns:
        JSON formatted email content including headers and body
    """
    print(f"Reading email with ID: {message_id}", file=sys.stderr)
    
    try:
        service = get_gmail_service()
        
        # Get the full message
        msg = service.users().messages().get(
            userId="me", 
            id=message_id, 
            format="full"
        ).execute()
        
        # Extract headers
        headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
        
        # Extract body text
        def extract_text_from_payload(payload):
            """Recursively extract text from email payload"""
            if payload.get("mimeType") == "text/plain":
                data = payload.get("body", {}).get("data")
                if data:
                    return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
            elif payload.get("mimeType") == "text/html":
                # Fallback to HTML if no plain text
                data = payload.get("body", {}).get("data")
                if data:
                    return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
            elif payload.get("parts"):
                # Check multipart message
                for part in payload["parts"]:
                    text = extract_text_from_payload(part)
                    if text and part.get("mimeType") == "text/plain":
                        return text
                # If no plain text found, try HTML parts
                for part in payload["parts"]:
                    text = extract_text_from_payload(part)
                    if text:
                        return text
            return ""
        
        body = extract_text_from_payload(msg["payload"])
        
        # Prepare email content
        email_content = {
            "id": message_id,
            "from": headers.get("From", ""),
            "to": headers.get("To", ""),
            "subject": headers.get("Subject", ""),
            "date": headers.get("Date", ""),
            "body": body or msg.get("snippet", "No body content available")
        }
        
        result = json.dumps(email_content, indent=2, ensure_ascii=False)
        print(f"Email read successfully: {headers.get('Subject', 'No Subject')}", file=sys.stderr)
        return result
        
    except Exception as e:
        error_msg = f" Failed to read email: {str(e)}"
        print(error_msg, file=sys.stderr)
        return error_msg

@mcp.tool()
async def get_labels() -> str:
    """
    Get all Gmail labels
    
    Returns:
        JSON formatted list of Gmail labels
    """
    print("Fetching Gmail labels", file=sys.stderr)
    
    try:
        service = get_gmail_service()
        
        results = service.users().labels().list(userId='me').execute()
        labels = results.get('labels', [])
        
        label_list = [{"id": label["id"], "name": label["name"]} for label in labels]
        
        result = f"Found {len(label_list)} labels:\n" + json.dumps(label_list, indent=2)
        print(f"Labels fetched: {len(label_list)} labels", file=sys.stderr)
        return result
        
    except Exception as e:
        error_msg = f" Failed to get labels: {str(e)}"
        print(error_msg, file=sys.stderr)
        return error_msg

# Server shutdown handler
def shutdown(signum, frame):
    print("Shutting down Gmail MCP Server...", file=sys.stderr)
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)

# Run the server
if __name__ == "__main__":
    print("Starting Gmail MCP Server on stdio", file=sys.stderr)
    try:
        mcp.run(transport="stdio")
    except Exception as e:
        print(f"Error running Gmail server: {e}", file=sys.stderr)
        sys.exit(1)