import requests
import os

class GoogleBooksAPI:
    BASE_URL = "https://www.googleapis.com/books/v1/volumes"

    @staticmethod
    def search_books(query: str, max_results: int = 5):
        params = {
            "q": query,
            "maxResults": max_results,
            "printType": "books"
        }
        
        api_key = os.getenv("GOOGLE_BOOKS_API_KEY")
        if api_key:
            params["key"] = api_key

        try:
            response = requests.get(GoogleBooksAPI.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            books = []
            if "items" in data:
                for item in data["items"]:
                    info = item.get("volumeInfo", {})
                    published_date = info.get("publishedDate", "0000")
                    try:
                        year = int(published_date[:4])
                    except:
                        year = 0
                    
                    books.append({
                        "book_id": item.get("id"),
                        "title": info.get("title", "Sem título"),
                        "authors": info.get("authors", ["Desconhecido"]),
                        "description": info.get("description", "Sem descrição disponível."),
                        "categories": info.get("categories", []),
                        "published_year": year,
                        "thumbnail": info.get("imageLinks", {}).get("thumbnail", "")
                    })
            return books
        except requests.exceptions.HTTPError as e:
            print(f"Error fetching from Google Books (HTTP {e.response.status_code}): {e}")
            return []
        except Exception as e:
            print(f"Error fetching from Google Books: {e}")
            return []