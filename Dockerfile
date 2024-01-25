FROM dataloopai/dtlpy-agent:cpu.py3.8.opencv4.7

RUN pip install --user \
    torch \
    ftfy \
    regex \
    git+https://github.com/openai/CLIP.git



# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/cpu/clip-search:0.1.0 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/cpu/clip-search:0.1.0
