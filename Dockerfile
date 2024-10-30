FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.py3.10.pytorch2

RUN pip install --user \
    torch \
    ftfy \
    regex \
    pillow>=11.0.0 \
    git+https://github.com/openai/CLIP.git



# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/cpu/clip-search:0.1.0 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/cpu/clip-search:0.1.0
