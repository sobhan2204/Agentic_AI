from mcp.server.fastmcp import FastMCP
from googletrans import Translator
import asyncio

mcp = FastMCP("Translate")
translator = Translator()


@mcp.tool()
async def translate(sentence: str, target: str) -> dict:
    """
    Translate text to target language.

    Args:
        sentence: Text to translate
        target: Target language code (e.g., 'es', 'fr', 'de')

    Returns:
        dict with translated text or error
    """

    try:
        result = await asyncio.to_thread(
            translator.translate,
            sentence,
            dest=target
        )
        return {"translated_text": result.text}

    except Exception as e:
        return {"error": f"Translation failed: {str(e)}"}


if __name__ == "__main__":
    mcp.run(transport="stdio")
