from unittest.mock import patch, Mock
import pytest
from backend.google_books import GoogleBooksAPI

def test_search_books_success():
    # Mock da resposta do requests.get
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "items": [
            {
                "id": "123",
                "volumeInfo": {
                    "title": "Test Book",
                    "authors": ["Test Author"],
                    "description": "A test description",
                    "categories": ["Tech"],
                    "publishedDate": "2023-01-01",
                    "imageLinks": {"thumbnail": "http://img.url"}
                }
            }
        ]
    }

    with patch("requests.get", return_value=mock_response):
        books = GoogleBooksAPI.search_books("python")
        
        assert len(books) == 1
        assert books[0]["title"] == "Test Book"
        assert books[0]["book_id"] == "123"
        assert books[0]["published_year"] == 2023

def test_search_books_empty():
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"totalItems": 0} # Sem a chave 'items'

    with patch("requests.get", return_value=mock_response):
        books = GoogleBooksAPI.search_books("non_existent_query")
        assert books == []

def test_search_books_error():
    with patch("requests.get", side_effect=Exception("Connection Error")):
        books = GoogleBooksAPI.search_books("error")
        assert books == []