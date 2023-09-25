---
SFX AI: Step-by-Step Guide to Running AudioCraft
---

# Step-by-Step Guide to Running AudioCraft on MacOS, Windows, or Linux

AudioCraft is a powerful tool for audio generation, including MusicGen and AudioGen functionalities. Follow these steps to get AudioCraft up and running on your MacOS, Windows, or Linux system:

## System Requirements and Dependencies
Before installing AudioCraft, ensure that you meet the following system requirements and dependencies:

- **Python 3.9**: Make sure you have Python 3.9 installed on your system.
- **PyTorch 2.0.0**: Install PyTorch by running the following command: pip install 'torch>=2.0'
- **ffmpeg (optional, but recommended)**: ffmpeg is used for audio and video processing. It's recommended to install it, although it's optional.

### Installation

1. **Ensure Python 3.9 is Installed**:
 - Verify that Python 3.9 is installed on your system. You can check your Python version by running:
   ```
   python --version
   ```

2. **Install PyTorch**:
 - If you haven't already installed PyTorch, use the following command:
   ```
   pip install 'torch>=2.0'
   ```

3. **Install AudioCraft**:
 - You have several installation options for AudioCraft:
   - For a stable release, run:
     ```
     pip install -U audiocraft
     ```
   - For the latest (possibly unstable) version from GitHub, run:
     ```
     pip install -U git+https://github.com/facebookresearch/audiocraft#egg=audiocraft
     ```
   - If you've cloned the AudioCraft repository locally, you can install it from the local directory using:
     ```
     pip install -e .
     ```

4. **Install ffmpeg (if needed)**:
 - Depending on your system, you may need to install ffmpeg:
   - On Ubuntu/Debian:
     ```
     sudo apt-get install ffmpeg
     ```
   - On Anaconda or Miniconda (conda-forge channel):
     ```
     conda install 'ffmpeg<5' -c conda-forge
     ```

## Running MusicGen

Now that you have AudioCraft installed, you can use MusicGen for audio generation. MusicGen provides pretrained models for generating music from text descriptions.

1. **Choose a Pretrained Model**:
 - MusicGen offers several pretrained models. Choose the one that suits your needs, such as `facebook/musicgen-medium` or `facebook/musicgen-melody`.

2. **Python Snippet to Get Started**:
 - Use the following Python code snippet to generate music with your chosen model:
   ```python
   import torchaudio
   from audiocraft.models import MusicGen
   from audiocraft.data.audio import audio_write
   
   # Load the pretrained model (e.g., 'facebook/musicgen-melody')
   model = MusicGen.get_pretrained('facebook/musicgen-melody')
   
   # Set generation parameters (e.g., generate 8 seconds of music)
   model.set_generation_params(duration=8)
   
   # Generate unconditional audio samples (e.g., 4 samples)
   wav = model.generate_unconditional(4)
   
   # Generate audio based on descriptions (e.g., 'happy rock', 'energetic EDM', 'sad jazz')
   descriptions = ['happy rock', 'energetic EDM', 'sad jazz']
   wav = model.generate(descriptions)
   
   # Load a melody for generating music with melody and descriptions
   melody, sr = torchaudio.load('./assets/bach.mp3')
   wav = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr)
   
   # Save generated audio with loudness normalization
   for idx, one_wav in enumerate(wav):
       audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
   ```

## Running AudioGen

AudioGen is a straightforward API for generating sound from text descriptions. Follow these steps to use AudioGen:

1. **Choose a Pretrained Model**:
 - AudioGen provides the pretrained model `facebook/audiogen-medium` for text-to-sound generation.

2. **Python Example**:
 - Use the following Python code snippet to generate sound with AudioGen:
   ```python
   import torchaudio
   from audiocraft.models import AudioGen
   from audiocraft.data.audio import audio_write
   
   # Load the pretrained model (e.g., 'facebook/audiogen-medium')
   model = AudioGen.get_pretrained('facebook/audiogen-medium')
   
   # Set generation parameters (e.g., generate 5 seconds of sound)
   model.set_generation_params(duration=5)
   
   # Generate unconditional audio samples (e.g., 4 samples)
   wav = model.generate_unconditional(4)
   
   # Generate audio based on descriptions (e.g., 'dog barking', 'siren of an emergency vehicle', 'footsteps in a corridor')
   descriptions = ['dog barking', 'siren of an emergency vehicle', 'footsteps in a corridor']
   wav = model.generate(descriptions)
   
   # Save generated audio with loudness normalization
   for idx, one_wav in enumerate(wav):
       audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
   ```
