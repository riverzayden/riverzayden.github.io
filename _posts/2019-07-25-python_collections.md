---
layout: post

title:  "[Python] Python Collections패키지 사용방법"

subtitle:   "[Python] Python Collections패키지 사용방법"

categories: python

tags: Python collections packages dict_count

comments: true

img: 


---



## Collections 패키지 사용법



1. List

```python
# collections.Counter 예제 (1)
# list를 입력값으로 함
import collections
lst = ['aa', 'cc', 'dd', 'aa', 'bb', 'ee']
print(collections.Counter(lst))
'''
결과
Counter({'aa': 2, 'cc': 1, 'dd': 1, 'bb': 1, 'ee': 1})
'''


```



2.  Dictionary  - 요소의 갯수가 많은 것 부터 출력해준다.

```python
# collections.Counter 예제 (2)
# dictionary를 입력값으로 함
import collections
print(collections.Counter({'가': 3, '나': 2, '다': 4}))
'''
결과
Counter({'다': 4, '가': 3, '나': 2})
'''

```



3. 값= 개수 형태
   *   `collections.Counter()`에는 `값=개수`형태로 입력이 가능하다.
     예를들어, `collections.Counter(a=2, b=3, c=2)`는 `['a', 'a', 'b', 'b', 'b', 'c', 'c']`와 같다.
     아래의 예제(3)의 출력값을 통해 확인할 수 있다.

```python
# collections.Counter 예제 (3)
# '값=개수' 입력값으로 함
import collections
c = collections.Counter(a=2, b=3, c=2)
print(collections.Counter(c))
print(sorted(c.elements()))
'''
결과
Counter({'b': 3, 'c': 2, 'a': 2})
['a', 'a', 'b', 'b', 'b', 'c', 'c']
'''


```



4. 문자열 입력  ( 문자: 개수) 의 딕셔너리 형태로 반환 

```python
import collections
container = collections.Counter()
container.update("aabcdeffgg")
print(container)
'''
결과
Counter({'f': 2, 'g': 2, 'a': 2, 'e': 1, 'b': 1, 'c': 1, 'd': 1})
'''

```



5. most_common 

* `most_common`은 입력된 값의 요소들 중 빈도수(frequency)가 높은 순으로 상위 nnn개를 리스트(list) 안의 투플(tuple) 형태로 반환한다. nn을 입력하지 않은 경우, 요소 전체를 [('값', 개수)]의 형태로 반환한다.

```python
import collections
c2 = collections.Counter('apple, orange, grape')
print(c2.most_common())
print(c2.most_common(3))
'''
결과
[('a', 3), ('p', 3), ('e', 3), ('g', 2), (',', 2), ('r', 2), (' ', 2), ('n', 1), ('l', 1), ('o', 1)]
[('a', 3), ('p', 3), ('e', 3)]
'''

```



6. subtract()   
   * 요소를 서로 빼는 것 

```python
import collections
c3 = collections.Counter('hello python')
c4 = collections.Counter('i love python')
c3.subtract(c4)
print(c3)
'''
결과
Counter({'l': 1, 'h': 1, 'n': 0, 't': 0, 'p': 0, 'e': 0, 'o': 0, 'y': 0, 'i': -1, 'v': -1, ' ': -1})
'''

```



7. 덧셈

```python
import collections
a = collections.Counter(['a', 'b', 'c', 'b', 'd', 'a'])
b = collections.Counter('aaeroplane')
print(a)
print(b)
print(a+b)
'''
결과
Counter({'b': 2, 'a': 2, 'd': 1, 'c': 1})
Counter({'a': 3, 'e': 2, 'n': 1, 'r': 1, 'o': 1, 'p': 1, 'l': 1})
Counter({'a': 5, 'b': 2, 'e': 2, 'n': 1, 'l': 1, 'd': 1, 'r': 1, 'o': 1, 'p': 1, 'c': 1})
'''
```



8. 교집합

```python
import collections
a = collections.Counter('aabbccdd')
b = collections.Counter('aabbbce')
print(a & b)
'''
결과
Counter({'b': 2, 'a': 2, 'c': 1})
'''
print(a | b)
'''
결과
Counter({'b': 3, 'c': 2, 'd': 2, 'a': 2, 'e': 1})
'''

```

