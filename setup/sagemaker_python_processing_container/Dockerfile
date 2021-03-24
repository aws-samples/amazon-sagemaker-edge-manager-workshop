FROM FROM public.ecr.aws/sam/build-python3.7:1.20.0

RUN pip3 install pandas==0.25.3 scikit-learn==0.21.3 PyWavelets==1.1.1
ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3"]
