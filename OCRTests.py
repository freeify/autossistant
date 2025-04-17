import pytesseract
from PIL import Image

# UNCOMMENT AND FIX THIS LINE - point to where you installed Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image = Image.open(r"C:\Users\EphraimMataranyika\Pictures\Screenshots\Omni Parser\Screenshot 2025-04-13 114811.png")
text = pytesseract.image_to_string(image)
print(text)