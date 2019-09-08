---
layout: post

title:  "[Python] Python String Replace"

subtitle:   "[Python] Python String Replace"

categories: python

tags: Python string replace

comments: true

img: 

---

## Python String Replace

1. replace(old, new, count) -> replace("찾을값", "바꿀값", 바꿀횟수)

```python
text = '123,456,789,999'

replaceAll= text.replace(",","")
replace_t1 = text.replace(",", "",1)
replace_t2 = text.replace(",", "",2)
replace_t3 = text.replace(",", "",3)
print("결과 :")
print(replaceAll)
print(replace_t1)
print(replace_t2)
print(replace_t3)

'''
결과 : 
123456789999
123456,789,999
123456789,999
123456789999
'''
```



2. 우측부터 변경법 

```python
def replaceRight(original, old, new, count_right):
    repeat=0
    text = original
    
    count_find = original.count(old)
    if count_right > count_find : # 바꿀 횟수가 문자열에 포함된 old보다 많다면
        repeat = count_find # 문자열에 포함된 old의 모든 개수(count_find)만큼 교체한다
    else :
        repeat = count_right # 아니라면 입력받은 개수(count)만큼 교체한다

    for _ in range(repeat):
        find_index = text.rfind(old) # 오른쪽부터 index를 찾기위해 rfind 사용
        text = text[:find_index] + new + text[find_index+1:]
    
    return text

text = '123,456,789,999'
#text.replace(",", "", -1); print(text) #안됨

#text = replaceRight(text, ",", "", 2)
print("결과 :")
print(replaceRight(text, ",", "", 0)) 
print(replaceRight(text, ",", "", 1))
print(replaceRight(text, ",", "", 2))
print(replaceRight(text, ",", "", 3))
print(replaceRight(text, ",", "", 4))
'''
결과 :
123,456,789,999
123,456,789999
123,456789999
123456789999
123456789999
'''
```

