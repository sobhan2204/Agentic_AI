from mcp.server.fastmcp import FastMCP
import asyncio

mcp = FastMCP("Translate")

@mcp.tool()
async def translate(sentence: str, target_language: str) -> str:
    """
    Translate text to a target language.

    Args:
        sentence: The text to translate (e.g., "hello", "good morning")
        target_language: The language to translate into (e.g., "french", "es", "german", "ja")

    Returns:
        str: The translated text
    """
    try:
        # Import inside the function to avoid any startup-time failures
        from deep_translator import GoogleTranslator

        target = target_language.strip().lower()

        result = await asyncio.to_thread(
            GoogleTranslator(source="auto", target=target).translate,
            sentence.strip()
        )

        return result if result else "Translation returned empty result"

    except Exception as e:
        return f"Translation failed: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")