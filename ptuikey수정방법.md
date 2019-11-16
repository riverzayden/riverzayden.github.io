@@ https://github.com/statslabs/rmath





1. git clone https://github.com/statslabs/rmath.git

2.  

   ```
   cd rmath
   mkdir build && cd build
   cmake ..
   ```

3. Compile and install the library.

```
make
make install
```



4. Example 파일 만들기

```
mkdir example && cd example
vim CMakeLists.txt
vim demo.c
```

vim CMakeLists.txt

```txt
cmake_minimum_required(VERSION 3.0)
project(example)
add_executable(example demo.c)

find_package(Rmath 1.0.0 REQUIRED)
target_link_libraries(example Rmath::Rmath)
```

vim demo.c

```c
#include <stdio.h>
#include "Rmath.h"

int main() {
  double shape1,shape2;

  printf("Enter first shape parameter: ");
  scanf("%lf",&shape1);

  printf("Enter first shape parameter: ");
  scanf("%lf",&shape2);

  printf("Critical value is %lf\n",ptukey(shape1,1,shape2, 10, TRUE,FALSE));

  return 0;
}

```





5. Perform a out-of-source build.

   ```
   mkdir build && cd build
   cmake ..
   make
   ```





6. Run the program.

   ```
   ./example
   ```



7. 결과  ( 5, 3)

   Critical value is 0.986384





8. 로그를 보기 위해 수정할 것 

/src/ptukey.c 파일에서 각 printf 를 사용하여 로그를 하나하나 찍어서 사용할 것 