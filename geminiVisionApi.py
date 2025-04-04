from PIL import Image  # Correct way to import PIL
import os
import google.generativeai as genai

# Set up file paths (ensure paths are valid on your system)
image_path_1 = r"C:\Users\EphraimMataranyika\Pictures\Screenshots\Omni Parser\Screenshot 2025-03-21 101155.png"

image_path_2 = r"C:\Users\EphraimMataranyika\Pictures\Screenshots\Omni Parser\organized_icons\top_navigation\icon_186_top_navigation.png"  # Update this with your actual second image path

try:
    # Open the first image
    sample_file_1 = Image.open(image_path_1)

    # Open the second image (commented out until you have the correct path)
    # sample_file_2 = Image.open(image_path_2)

    # Configure the Gemini API (provide your API key here or ensure ADC is set up)
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY', 'AIzaSyBzGPxlqFj1eDO-fB16Yv7xW6iSeObJeik'))  # Replace with your API key

    # Initialize the model
    model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")

    # Create the prompt
    prompt = "describe the appearence of the icon"

    # Generate content
    response = model.generate_content([sample_file_1, prompt])

    # Print the response
    print("Response:", response.text)

except FileNotFoundError:
    print("Error: One or more image files not found. Please check the file paths.")
except genai.errors.AuthenticationError as auth_err:
    print("Authentication Error: Ensure your API key or ADC is configured correctly.")
except Exception as e:
    print(f"An error occurred: {str(e)}")