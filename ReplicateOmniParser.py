import os
import replicate

# Set your Replicate API token
os.environ["REPLICATE_API_TOKEN"] = "r8_84rIIme1v9PsHUARiOzSHBSzuXfYQOg0gc1XD"  # Replace with your actual token

input = {
    "image": r"C:\Users\EphraimMataranyika\Pictures\Screenshots\Omni Parser\Screenshot 2025-02-19 104440.png"
}

output = replicate.run(
    "microsoft/omniparser-v2:49cf3d41b8d3aca1360514e83be4c97131ce8f0d99abfc365526d8384caa88df",
    input=input
)
print(output)