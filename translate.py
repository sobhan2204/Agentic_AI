from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Translate")

from lara_sdk import Translator, Credentials

LARA_ACCESS_KEY_ID = "BU2BEJ6MOKP4E4UM2U7AMGTFUP"      # Replace with your Access Key ID
LARA_ACCESS_KEY_SECRET = "Cq93F6yV-fmyObtlcMu5BELjYfXWHvsnh5AEDKcdR-0"  # Replace with your Access Key SECRET

@mcp.tool()
def translate(sentence : str , target : str) -> str | list[str] | list:
    '''summary_
    convert the sentence to the target language
    Args:
      sentence : write any sentence in english 
      target : language you want to convert to (it-IT, fr-FR, de-DE, es-ES, etc.)
    '''
    credentials = Credentials(access_key_id=LARA_ACCESS_KEY_ID, access_key_secret=LARA_ACCESS_KEY_SECRET)
    lara = Translator(credentials)

    # This translates your text from English ("en-US") to Italian ("it-IT").
    res = lara.translate(sentence,
                         source="en-US",
                         target=target)

    # Returns the translated text
    return res.translation

if __name__ == '__main__':
    mcp.run(transport="stdio")