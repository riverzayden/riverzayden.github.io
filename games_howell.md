```R
library(userfriendlyscience)
one.way <- oneway(InsectSprays$spray, y = InsectSprays$count, posthoc = 'games-howell')
one.way

games.howell <- function(grp, obs) {
  
  #Create combinations
  combs <- combn(unique(grp), 2)
  
  # Statistics that will be used throughout the calculations:
  # n = sample size of each group
  # groups = number of groups in data
  # Mean = means of each group sample
  # std = variance of each group sample
  n <- tapply(obs, grp, length)
  groups <- length(tapply(obs, grp, length))
  Mean <- tapply(obs, grp, mean)
  std <- tapply(obs, grp, var)
  
  statistics <- lapply(1:ncol(combs), function(x) {
    
    mean.diff <- Mean[combs[2,x]] - Mean[combs[1,x]]
    
    #t-values
    t <- abs(Mean[combs[1,x]] - Mean[combs[2,x]]) / sqrt((std[combs[1,x]] / n[combs[1,x]]) + (std[combs[2,x]] / n[combs[2,x]]))
    
    # Degrees of Freedom
    df <- (std[combs[1,x]] / n[combs[1,x]] + std[combs[2,x]] / n[combs[2,x]])^2 / # Numerator Degrees of Freedom
      ((std[combs[1,x]] / n[combs[1,x]])^2 / (n[combs[1,x]] - 1) + # Part 1 of Denominator Degrees of Freedom 
         (std[combs[2,x]] / n[combs[2,x]])^2 / (n[combs[2,x]] - 1)) # Part 2 of Denominator Degrees of Freedom
    
    #p-values
    p <- ptukey(t * sqrt(2), groups, df, lower.tail = FALSE)
    print('-----')
    print(paste(t * sqrt(2), '그룹은',groups,'디에프' ,df,'피는',p) )
    print('###########################')
    # Sigma standard error
    se <- sqrt(0.5 * (std[combs[1,x]] / n[combs[1,x]] + std[combs[2,x]] / n[combs[2,x]]))
    
    # Upper Confidence Limit
    upper.conf <- lapply(1:ncol(combs), function(x) {
      mean.diff + qtukey(p = 0.95, nmeans = groups, df = df) * se
    })[[1]]
    
    # Lower Confidence Limit
    lower.conf <- lapply(1:ncol(combs), function(x) {
      mean.diff - qtukey(p = 0.95, nmeans = groups, df = df) * se
    })[[1]]
    
    # Group Combinations
    grp.comb <- paste(combs[1,x], ':', combs[2,x])
    
    # Collect all statistics into list
    stats <- list(grp.comb, mean.diff, se, t, df, p, upper.conf, lower.conf)
  })
  
  # Unlist statistics collected earlier
  stats.unlisted <- lapply(statistics, function(x) {
    unlist(x)
  })
  
  # Create dataframe from flattened list
  results <- data.frame(matrix(unlist(stats.unlisted), nrow = length(stats.unlisted), byrow=TRUE))
  
  # Select columns set as factors that should be numeric and change with as.numeric
  results[c(2, 3:ncol(results))] <- round(as.numeric(as.matrix(results[c(2, 3:ncol(results))])), digits = 3)
  
  # Rename data frame columns
  colnames(results) <- c('groups', 'Mean Difference', 'Standard Error', 't', 'df', 'p', 'upper limit', 'lower limit')
  
  return(results)
  
}

games.howell(InsectSprays$spray, InsectSprays$count)
```

```output
   groups Mean Difference Standard Error     t     df     p upper limit lower limit
1   A : B           0.833          1.299 0.454 21.784 0.997       6.562      -4.896
2   A : C         -12.417          1.044 8.407 14.739 0.000      -7.607     -17.226
3   A : D          -9.583          1.090 6.214 16.735 0.000      -4.641     -14.525
4   A : E         -11.000          1.026 7.580 13.910 0.000      -6.236     -15.764
5   A : F           2.167          1.593 0.962 20.523 0.925       9.229      -4.895
6   B : C         -13.250          0.961 9.754 15.499 0.000      -8.855     -17.645
7   B : D         -10.417          1.011 7.289 17.758 0.000      -5.868     -14.965
8   B : E         -11.833          0.941 8.894 14.523 0.000      -7.492     -16.175
9   B : F           1.333          1.539 0.613 19.498 0.989       8.192      -5.526
10  C : D           2.833          0.651 3.078 20.872 0.056       5.715      -0.048
11  C : E           1.417          0.536 1.868 21.631 0.447       3.783      -0.949
12  C : F          14.583          1.331 7.748 13.201 0.000      20.810       8.357
13  D : E          -1.417          0.621 1.612 19.570 0.601       1.351      -4.185
14  D : F          11.750          1.367 6.076 14.479 0.000      18.063       5.437
15  E : F          13.167          1.317 7.071 12.699 0.000      19.364       6.969
```



```data
> InsectSprays
   count spray
1     10     A
2      7     A
3     20     A
4     14     A
5     14     A
6     12     A
7     10     A
8     23     A
9     17     A
10    20     A
11    14     A
12    13     A
13    11     B
14    17     B
15    21     B
16    11     B
17    16     B
18    14     B
19    17     B
20    17     B
21    19     B
22    21     B
23     7     B
24    13     B
25     0     C
26     1     C
27     7     C
28     2     C
29     3     C
30     1     C
31     2     C
32     1     C
33     3     C
34     0     C
35     1     C
36     4     C
37     3     D
38     5     D
39    12     D
40     6     D
41     4     D
42     3     D
43     5     D
44     5     D
45     5     D
46     5     D
47     2     D
48     4     D
49     3     E
50     5     E
51     3     E
52     5     E
53     3     E
54     6     E
55     1     E
56     1     E
57     3     E
58     2     E
59     6     E
60     4     E
61    11     F
62     9     F
63    15     F
64    22     F
65    15     F
66    16     F
67    13     F
68    10     F
69    26     F
70    26     F
71    24     F
72    13     F
```





```python
import pandas as pd
import itertools as it
from statsmodels.stats.libqsturng import psturng
import numpy as np
data = pd.read_csv('./data.csv')
group_unique = set(data['spray'])
group_size = len(group_unique)
combs = it.combinations(range(group_size), 2)
n_i = data.groupby('spray').count() 
value_mean_i = data.groupby('spray').mean() 
value_var_i = data.groupby('spray').var()
data = pd.DataFrame(combs,columns=['group1','group2'])

data['mean_1'] = value_mean_i.values[data['group1']]
data['mean_2'] = value_mean_i.values[data['group2']]
data['var_1'] = value_var_i.values[data['group1']]
data['var_2'] = value_var_i.values[data['group2']]
data['n_1'] = n_i.values[data['group1']]
data['n_2'] = n_i.values[data['group2']]
data['meandiff'] = abs(data['mean_1'] - data['mean_2'])

data['t'] =  data['meandiff'] / np.sqrt (  data['var_1'] / data['n_1']  + data['var_2'] / data['n_2'] )
data['df'] = ((data['var_1'] / data['n_1'] + data['var_2'] / data['n_2'])**2 )  / \
((data['var_1'] / data['n_1'])**2 / (data['n_1']-1) + (data['var_2'] / data['n_2'])**2 / (data['n_2']-1))
data['t2'] = data['t'] * np.sqrt(2)
data['p'] = psturng(data['t2'], group_size, data['df'])
data

```

<img width="613" alt="캡처" src="https://user-images.githubusercontent.com/49559408/68122164-13b98200-ff4d-11e9-983d-1218c3eb7627.PNG">

