---

layout: post

title:  "[Python] Multiprocessing Csv Save "

subtitle:   "[Python] Multiprocessing Csv Save "

categories: python

tags: python multiprocessing csv save

comments: true

img: 

---

#### Multiprocessing Csv Save

```python
from multiprocessing import Process,Queue
import os
def big_table_save(num, table):
    table.to_csv(os.path.join('./test2','_'+str(num)+'.csv'),index=False,header=None)
    print("success")
    
def total_table_save(table):
    counting = len(table)
    if counting <= 100000:
        small_table_save(table)
    else:
        procs = []
        numbers=list(range(0,10))
        quotient, remainder = (counting//10, counting%10)
        for index, number in enumerate(numbers):
            if number!=9:
                part_table=table[quotient*number:quotient*(number+1)]
            else:
                part_table=table[quotient*number:]
            proc = Process(target=big_table_save, args=( number, part_table  )) 
            procs.append(proc) 
            proc.start()   
        for proc in procs: 
            proc.join()
```



