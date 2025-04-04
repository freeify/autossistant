# Load model directly
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering

processor = AutoProcessor.from_pretrained("microsoft/OmniParser")
model = AutoModelForVisualQuestionAnswering.from_pretrained("microsoft/OmniParser")