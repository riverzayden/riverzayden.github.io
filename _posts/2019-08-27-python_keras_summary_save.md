---
layout: post

title:  "[Deep_Learning] Keras Model Summary Save "

subtitle:   "[Deep_Learning] Keras Model Summary Save "

categories: dl

tags: Python Deep_Learning keras summary save

comments: true

img: 

---

## Keras Model Summary Save

```python
from contextlib import redirect_stdout

with open('modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()
```

