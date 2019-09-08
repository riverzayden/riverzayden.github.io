---
layout: post

title:  "[Python] Flag parser 사용법"

subtitle:   "[Python] Flag parser 사용법"

categories: python

tags: Python flag parser

comments: true

img: 

---


## Flag parser 사용법

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-port", "--port", type=int, default=None,
                help='Flask port')
args = parser.parse_args()

print(args.port)
```



## **bool사용법**

```python
parser.add_argument("-include_top", "--include_top", type=lambda x: (str(x).lower() == 'true'), default=False,
                help='include_top')
```

