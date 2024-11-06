import re
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM
import soundfile as sf

class Yumeka:
    def __init__(self, device):
        self.device = device

        # Initialize LLM
        self.llm_tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M')
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        self.llm = AutoModelForCausalLM.from_pretrained('Daviduche03/SmolLM2-Instruct').to(self.device)

        # Initialize TTS
        self.tts_model = ParlerTTSForConditionalGeneration.from_pretrained(
            "parler-tts/parler-tts-mini-jenny-30H"
        ).to(self.device)
        self.tts_tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-jenny-30Hparler-tts/parler-tts-large-v1")

    def _clean_llm_output(self, text):
        """Clean the LLM output text by removing any special tokens or unwanted formatting."""
        result = re.search(r'<assistant>(.*?)<assistant/>', text, re.DOTALL)
        return result.group(1).strip() if result else text.strip()

    def generate(self, prompt, max_new_tokens=500):
        """
        Generate text and convert it to speech.

        Args:
            prompt (str): Input text prompt
            max_new_tokens (int): Maximum number of tokens to generate

        Returns:
            tuple: (audio_array, sample_rate, generated_text)
        """
        # Generate text with LLM
        tokenized_prompt = self.llm_tokenizer(
            "<user>" + prompt + "<user/>",
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        llm_output = self.llm.generate(
            tokenized_prompt["input_ids"],
            attention_mask=tokenized_prompt["attention_mask"],
            max_new_tokens=max_new_tokens,
            pad_token_id=self.llm.config.pad_token_id,
            eos_token_id=self.llm.config.eos_token_id
        )

        # Decode and clean the generated text
        output_text = self.llm_tokenizer.decode(llm_output[0], skip_special_tokens=True)
        output_text = self._clean_llm_output(output_text)

        # Generate speech
        tts_prompt_input_ids = self.tts_tokenizer(
            output_text,
            return_tensors="pt"
        ).input_ids.to(self.device)

        description = "Jenny speaks at an average pace with an animated delivery in a very confined sounding environment with clear audio quality."
        tts_description_input_ids = self.tts_tokenizer(
            description,
            return_tensors="pt"
        ).input_ids.to(self.device)

        generation = self.tts_model.generate(
            input_ids=tts_description_input_ids,
            prompt_input_ids=tts_prompt_input_ids
        )

        # Convert to audio array
        audio_arr = generation.cpu().numpy().squeeze()

        return audio_arr, self.tts_model.config.sampling_rate, output_text

    def save_audio(self, audio_arr, sample_rate, filename):
        """Save the generated audio to a file."""
        sf.write(filename, audio_arr, sample_rate)


# Usage example
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Yumeka(device)

    # Generate audio
    audio_arr, sample_rate, output_text = model.generate("What is machine learning?")

    # Save the audio
    model.save_audio(audio_arr, sample_rate, "parler_tts_out.wav")

    # Print the generated text
    print("Generated text:", output_text)