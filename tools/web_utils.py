# In a tools file, e.g., tools/web_utils.py or similar

import logging
from newspaper import Article, ArticleException
import requests # newspaper3k uses requests, so good to handle its exceptions

# Ensure NLTK 'punkt' is downloaded, as discussed previously.
# You might have a central place for this check in your app's startup
# or ensure it's done during setup. For example:
# import nltk
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt', quiet=True)


def fetch_web_article_text_tool(url: str) -> dict:
    """
    Fetches the main text content and title from a given web URL using newspaper3k.
    This tool is intended to be called by an agent that needs the text content
    of a web article before passing it to another agent or tool for processing.

    Args:
        url: The URL of the web article.

    Returns:
        A dictionary containing:
        {
            "status": "success" | "error",
            "title": "extracted_article_title" | None,
            "text": "extracted_article_text" | None,
            "original_url": url,
            "message": "A message about the outcome."
        }
    """
    logging.info(f"Attempting to fetch web article content from URL: {url}")

    try:
        article = Article(url, fetch_images=False, memoize_articles=True) # fetch_images=False for speed
        article.download()
        article.parse()

        extracted_title = article.title
        extracted_text = article.text

        if not extracted_text:
            logging.warning(f"Newspaper3k could not extract text for: {url}")
            return {
                "status": "error",
                "title": extracted_title, # Might still get a title
                "text": None,
                "original_url": url,
                "message": f"Could not extract the main content from the article at {url}."
            }

        logging.info(f"Successfully fetched text for article: '{extracted_title}' from {url}")
        return {
            "status": "success",
            "title": extracted_title,
            "text": extracted_text,
            "original_url": url,
            "message": "Successfully fetched article content."
        }

    except ArticleException as e:
        logging.error(f"Newspaper3k ArticleException for URL {url}: {e}")
        return {
            "status": "error",
            "title": None,
            "text": None,
            "original_url": url,
            "message": f"Could not process the article from {url}. It might not be a standard news article or the URL is invalid. Error: {e}"
        }
    except requests.exceptions.RequestException as e:
        logging.error(f"Network error (requests.exceptions.RequestException) for URL {url}: {e}")
        return {
            "status": "error",
            "title": None,
            "text": None,
            "original_url": url,
            "message": f"Could not connect to the URL {url}. Please check the address. Error: {e}"
        }
    except Exception as e:
        logging.error(f"An unexpected error occurred while fetching web article {url}: {e}", exc_info=True)
        return {
            "status": "error",
            "title": None,
            "text": None,
            "original_url": url,
            "message": f"An unexpected error occurred while fetching the article from {url}. Error: {e}"
        }