import os
import asyncio
#from mcp.server.fastmcp import FastMCP

#mcp = FastMCP("google_calendar")

from dotenv import load_dotenv
load_dotenv()

import datetime
import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file client_secret_calendar.json.
SCOPES = ["https://www.googleapis.com/auth/calendar"]


def g_calender():
  """Shows basic usage of the Google Calendar API.
  Prints the start and name of the next 10 events on the user's calendar.
  """
  creds = None
  # The file client_secret_calendar.json stores the user's access and refresh tokens, and is
  # created automatically when the authorization flow completes for the first
  # time.
  if os.path.exists("client_secret_calendar.json"):
    creds = Credentials.from_authorized_user_file("client_secret_calendar.json", SCOPES)
  # If there are no (valid) credentials available, let the user log in.
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      flow = InstalledAppFlow.from_client_secrets_file(
          "credentials.json", SCOPES
      )
      creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open("client_secret_calendar.json", "w") as token:
      token.write(creds.to_json())

  try:
    service = build("calendar", "v3", credentials=creds)

    # Call the Calendar API
    now = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
    print("Getting the upcoming 10 events")
    events_result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=now,
            maxResults=10,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    events = events_result.get("items", [])

    if not events:
      print("No upcoming events found.")
      return

    # Prints the start and name of the next 10 events
    for event in events:
      start = event["start"].get("dateTime", event["start"].get("date"))
      print(start, event["summary"])

  except HttpError as error:
    print(f"An error occurred: {error}")

if __name__ == "__g_calender__":
     g_calender()

#@mcp.tools
async def create_event(summary:str , start_time , end_time , description=""):
  """Create a new event on the calendar
  Agr:
  summary = a short on what the event is about
  start_time = staring time of event
  end_time  = ending time of event
  discription = detail information of event     
  """
 
  print("Creating event")
  service = await g_calender()
  if not service:
   return None
    
  event = {
        "summary": summary,
        "description": description,
        "start": {
            "dateTime": start_time.isoformat(),
            "timeZone": "UTC",
        },
        "end": {
            "dateTime": end_time.isoformat(),
            "timeZone": "UTC",
        }
    }
  
  try:
    event = service.events().insert(calendarId='primary', body=event).execute()
    print("Event created")
    return event
  except HttpError as error:
    print(f"An error occurred: {error}")
    return None


async def get_upcomming_event(n : int = 5) -> str:
 """
 Tell me all the upcomming events in my calender
 Arg:
 n = number of events you want to see
 return:
 all the upcomming events
 """

 print("Fetching events")
 service = g_calender()
 if not service:
  return "No service avaliable"
 
 now = datetime.datetime.utcnow().isoformat() + 'Z'
 events_result = service.events().list(
        calendarId='primary', timeMin=now,
        maxResults=n, singleEvents=True,
        orderBy='startTime'
    ).execute()

 events = events_result.get('items', [])
 if not events:
        return "No upcoming events found."

 output = ""
 for i, event in enumerate(events, 1):
        start = event['start'].get('dateTime', event['start'].get('date'))
        summary = event.get('summary', 'No Title')
        output += f"{i}. {summary} at {start}\n"
 return output.strip()



#if __name__ == "__main__":
#   mcp.run(transport="stdio")
async def cal():
    await create_event(
        "Is event is my birthday",
        "20-08-2025",
        "21-08-2025",
        description="Beautiful event"
    )

asyncio.run(cal())