---
layout: post

title:  "[Python] Python 안에서 리눅스명령어 사용 "

subtitle:   "[Python] Python 안에서 리눅스명령어 사용 "

categories: python

tags: Python linux command subprocess

comments: true

img: 

---

## Python 안에서 리눅스명령어 사용 

```python
import subprocess
subprocess.call("python test.py" , shell=True)
```

