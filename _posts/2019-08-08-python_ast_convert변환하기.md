---
layout: post

title:  "[Python] ast 함수 사용 ( string-> dict, list )"

subtitle:   "[Python] ast 함수 사용 ( string-> dict, list )"

categories: python

tags: Python string dict list convert ast

comments: true

img: 

---

## ast 함수 사용 ( String을 dict or list로 변환) 

```python
import ast



str_dict = "{'a': 3, 'b': 5}"

print (type(str_dict))           # <type 'str'>



convert_dict = ast.literal_eval(str_dict)

print (type(convert_dict))   # <type 'dict'>

print (convert_dict['a'])      #  3


```

