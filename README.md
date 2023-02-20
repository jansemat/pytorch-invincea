# pytorch-invincea

*Disclaimer: This was for a deep learning course project*

This is an implementation of the Invincea deep learning model to detect algorithmically-generated domain names. It utilizes parallel CNN models for feature detection, and fully connected neural networks for classification. 

The pytorch script `invincea.py` depends on you having benign and malicious URLs to train on - change the `BENIGN_PATH` and `MAL_PATH` variables within the script accordingly. The script will also produce a file (the `OUTFILE` variable) with training/evaluation statistics.

You can run the script just like so
```py
$ python3 invincea.py
```

All referenced works can be found in `final.report.pdf` or `final.slides.pdf`.

