FROM hub.dataloop.ai/dtlpy-runner-images/gpu:python3.10_cuda11.8_pytorch2

RUN pip install --user \
    ftfy \
    regex \
    'pillow>=11.0.0' \
    git+https://github.com/openai/CLIP.git



# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/cpu/clip-search:0.1.0 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/cpu/clip-search:0.1.0
