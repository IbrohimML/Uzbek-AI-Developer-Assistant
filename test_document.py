from app.core.document import Document


document = Document(
    content="Python is a programming language.",
    metadata={
        "source": "python_docs",
        "page": 1
    }
)

print("Content:", document.content)
print("Metadata:", document.metadata)
print("Document type:", type(document))