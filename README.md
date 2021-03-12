## Importing neccesary libraries


```python
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
movies_df = pd.read_csv('ml-latest/movies.csv')
ratings_df = pd.read_csv('ml-latest/ratings.csv')
```


```python
movies_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>



### Store year of movie into new column name as year


```python
# For specifying the parenthesis
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))', expand=False) 

# Removing the parenthesis
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)', expand=False) 

# Removing year from the title column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')

# Apply Strip() to make sure every character end with white spaces
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

movies_df.head()
```

    <ipython-input-6-59f1c77d94f0>:6: FutureWarning: The default value of regex will change from True to False in a future version.
      movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')





<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji</td>
      <td>Adventure|Children|Fantasy</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men</td>
      <td>Comedy|Romance</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale</td>
      <td>Comedy|Drama|Romance</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II</td>
      <td>Comedy</td>
      <td>1995</td>
    </tr>
  </tbody>
</table>
</div>




```python
# converting genre into list
movies_df['genres'] = movies_df.genres.str.split('|')
movies_df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>[Adventure, Animation, Children, Comedy, Fantasy]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji</td>
      <td>[Adventure, Children, Fantasy]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men</td>
      <td>[Comedy, Romance]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale</td>
      <td>[Comedy, Drama, Romance]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II</td>
      <td>[Comedy]</td>
      <td>1995</td>
    </tr>
  </tbody>
</table>
</div>




```python
movieWithGenres_df = movies_df.copy()
```


```python
for index, row in movies_df.iterrows():
    for genre in row['genres']:
        movieWithGenres_df.at[index, genre] = 1

# Filling 0 in the place of NaN values.
movieWithGenres_df = movieWithGenres_df.fillna(0)
movieWithGenres_df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Children</th>
      <th>Comedy</th>
      <th>Fantasy</th>
      <th>Romance</th>
      <th>...</th>
      <th>Horror</th>
      <th>Mystery</th>
      <th>Sci-Fi</th>
      <th>IMAX</th>
      <th>Documentary</th>
      <th>War</th>
      <th>Musical</th>
      <th>Western</th>
      <th>Film-Noir</th>
      <th>(no genres listed)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>[Adventure, Animation, Children, Comedy, Fantasy]</td>
      <td>1995</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji</td>
      <td>[Adventure, Children, Fantasy]</td>
      <td>1995</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men</td>
      <td>[Comedy, Romance]</td>
      <td>1995</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale</td>
      <td>[Comedy, Drama, Romance]</td>
      <td>1995</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II</td>
      <td>[Comedy]</td>
      <td>1995</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



## Ratings


```python
ratings_df.head()
```




<div>

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>169</td>
      <td>2.5</td>
      <td>1204927694</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2471</td>
      <td>3.0</td>
      <td>1204927438</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>48516</td>
      <td>5.0</td>
      <td>1204927435</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>2571</td>
      <td>3.5</td>
      <td>1436165433</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>109487</td>
      <td>4.0</td>
      <td>1436165496</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Removing unnecessary Column
ratings_df = ratings_df.drop('timestamp', 1)
ratings_df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>169</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2471</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>48516</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>2571</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>109487</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



# Recommendation System


```python
userInput = [
        {'title':'Carrington', 'rating':4},
        {'title':'Heat', 'rating':5},
        {'title':'Nixon', 'rating':4.5},
        {'title':'Powder', 'rating':3},
        {'title':'Screamers', 'rating':5},
        {'title':"Things to Do in Denver When You're Dead", 'rating':4.5},
        {'title':'Dunston Checks In', 'rating':3.5},
        {'title':'Catwalk', 'rating':2},
    
    ]
inputMovies = pd.DataFrame(userInput)
inputMovies
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Carrington</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Heat</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Nixon</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Powder</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Screamers</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Things to Do in Denver When You're Dead</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Dunston Checks In</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Catwalk</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Filtering out the movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]

# Merging after then we will get movieId.
inputMovies = pd.merge(inputId, inputMovies)

# Removing unneccessary column
inputMovies = inputMovies.drop('genres', 1).drop('year', 1)

inputMovies
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>Heat</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>73608</td>
      <td>Heat</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>131274</td>
      <td>Heat</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14</td>
      <td>Nixon</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24</td>
      <td>Powder</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>35</td>
      <td>Carrington</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>76</td>
      <td>Screamers</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>119832</td>
      <td>Screamers</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>81</td>
      <td>Things to Do in Denver When You're Dead</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>87</td>
      <td>Dunston Checks In</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>10</th>
      <td>108</td>
      <td>Catwalk</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Filtering out the movies from user input
userMovies = movieWithGenres_df[movieWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]
userMovies
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Children</th>
      <th>Comedy</th>
      <th>Fantasy</th>
      <th>Romance</th>
      <th>...</th>
      <th>Horror</th>
      <th>Mystery</th>
      <th>Sci-Fi</th>
      <th>IMAX</th>
      <th>Documentary</th>
      <th>War</th>
      <th>Musical</th>
      <th>Western</th>
      <th>Film-Noir</th>
      <th>(no genres listed)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Heat</td>
      <td>[Action, Crime, Thriller]</td>
      <td>1995</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>Nixon</td>
      <td>[Drama]</td>
      <td>1995</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>24</td>
      <td>Powder</td>
      <td>[Drama, Sci-Fi]</td>
      <td>1995</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>35</td>
      <td>Carrington</td>
      <td>[Drama, Romance]</td>
      <td>1995</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>75</th>
      <td>76</td>
      <td>Screamers</td>
      <td>[Action, Sci-Fi, Thriller]</td>
      <td>1995</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>80</th>
      <td>81</td>
      <td>Things to Do in Denver When You're Dead</td>
      <td>[Crime, Drama, Romance]</td>
      <td>1995</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>86</th>
      <td>87</td>
      <td>Dunston Checks In</td>
      <td>[Children, Comedy]</td>
      <td>1996</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>106</th>
      <td>108</td>
      <td>Catwalk</td>
      <td>[Documentary]</td>
      <td>1996</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14752</th>
      <td>73608</td>
      <td>Heat</td>
      <td>[Comedy, Drama, Romance]</td>
      <td>1972</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25729</th>
      <td>119832</td>
      <td>Screamers</td>
      <td>[Action, Horror]</td>
      <td>1979</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>28425</th>
      <td>131274</td>
      <td>Heat</td>
      <td>[Action, Drama, Thriller]</td>
      <td>1986</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>11 rows × 24 columns</p>
</div>




```python
# Resetting the index to avoid future issues
userMovies = userMovies.reset_index(drop=True)

# Dropping unnecessary attributes
userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

userGenreTable
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Children</th>
      <th>Comedy</th>
      <th>Fantasy</th>
      <th>Romance</th>
      <th>Drama</th>
      <th>Action</th>
      <th>Crime</th>
      <th>Thriller</th>
      <th>Horror</th>
      <th>Mystery</th>
      <th>Sci-Fi</th>
      <th>IMAX</th>
      <th>Documentary</th>
      <th>War</th>
      <th>Musical</th>
      <th>Western</th>
      <th>Film-Noir</th>
      <th>(no genres listed)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
inputMovies['rating']
```




    0     5.0
    1     5.0
    2     5.0
    3     4.5
    4     3.0
    5     4.0
    6     5.0
    7     5.0
    8     4.5
    9     3.5
    10    2.0
    Name: rating, dtype: float64




```python
# Now applying dot product to get weights, the maximum weight will be recommends to the user
userProfile = userGenreTable.transpose().dot(inputMovies['rating'])

userProfile
```




    Adventure              0.0
    Animation              0.0
    Children               5.0
    Comedy                 9.5
    Fantasy                0.0
    Romance               13.0
    Drama                 25.0
    Action                13.5
    Crime                  9.0
    Thriller              10.0
    Horror                 3.5
    Mystery                0.0
    Sci-Fi                 8.0
    IMAX                   0.0
    Documentary            5.0
    War                    0.0
    Musical                0.0
    Western                0.0
    Film-Noir              0.0
    (no genres listed)     0.0
    dtype: float64




```python
# gets the genre of every movies from out main movie dataset
genreTable = movieWithGenres_df.set_index(movieWithGenres_df['movieId'])

genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

genreTable.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Children</th>
      <th>Comedy</th>
      <th>Fantasy</th>
      <th>Romance</th>
      <th>Drama</th>
      <th>Action</th>
      <th>Crime</th>
      <th>Thriller</th>
      <th>Horror</th>
      <th>Mystery</th>
      <th>Sci-Fi</th>
      <th>IMAX</th>
      <th>Documentary</th>
      <th>War</th>
      <th>Musical</th>
      <th>Western</th>
      <th>Film-Noir</th>
      <th>(no genres listed)</th>
    </tr>
    <tr>
      <th>movieId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
genreTable.shape
```




    (34208, 20)




```python
#  Multiplying the genre by the weights and take weightage average
recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
recommendationTable_df.head()
```




    movieId
    1    0.142857
    2    0.049261
    3    0.221675
    4    0.467980
    5    0.093596
    dtype: float64




```python
# Sorting ours recommendations into decreasing order
recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
recommendationTable_df.head()
```




    movieId
    127341    0.788177
    4719      0.788177
    76153     0.788177
    75408     0.788177
    150268    0.738916
    dtype: float64



## This is final recommendation *movies*


```python

movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(30).keys())]
```




<div>


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>Money Train</td>
      <td>[Action, Comedy, Crime, Drama, Thriller]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>143</th>
      <td>145</td>
      <td>Bad Boys</td>
      <td>[Action, Comedy, Crime, Drama, Thriller]</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>455</th>
      <td>459</td>
      <td>Getaway, The</td>
      <td>[Action, Adventure, Crime, Drama, Romance, Thr...</td>
      <td>1994</td>
    </tr>
    <tr>
      <th>1398</th>
      <td>1432</td>
      <td>Metro</td>
      <td>[Action, Comedy, Crime, Drama, Thriller]</td>
      <td>1997</td>
    </tr>
    <tr>
      <th>4625</th>
      <td>4719</td>
      <td>Osmosis Jones</td>
      <td>[Action, Animation, Comedy, Crime, Drama, Roma...</td>
      <td>2001</td>
    </tr>
    <tr>
      <th>4774</th>
      <td>4869</td>
      <td>Burnt Money (Plata Quemada)</td>
      <td>[Action, Crime, Drama, Romance, Thriller]</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>4861</th>
      <td>4956</td>
      <td>Stunt Man, The</td>
      <td>[Action, Adventure, Comedy, Drama, Romance, Th...</td>
      <td>1980</td>
    </tr>
    <tr>
      <th>4932</th>
      <td>5027</td>
      <td>Another 48 Hrs.</td>
      <td>[Action, Comedy, Crime, Drama, Thriller]</td>
      <td>1990</td>
    </tr>
    <tr>
      <th>5530</th>
      <td>5628</td>
      <td>Wasabi</td>
      <td>[Action, Comedy, Crime, Drama, Thriller]</td>
      <td>2001</td>
    </tr>
    <tr>
      <th>7124</th>
      <td>7235</td>
      <td>Ichi the Killer (Koroshiya 1)</td>
      <td>[Action, Comedy, Crime, Drama, Horror, Thriller]</td>
      <td>2001</td>
    </tr>
    <tr>
      <th>7346</th>
      <td>7483</td>
      <td>Foreign Land (Terra Estrangeira)</td>
      <td>[Action, Crime, Drama, Romance, Thriller]</td>
      <td>1996</td>
    </tr>
    <tr>
      <th>9488</th>
      <td>27781</td>
      <td>Svidd Neger</td>
      <td>[Comedy, Crime, Drama, Horror, Mystery, Romanc...</td>
      <td>2003</td>
    </tr>
    <tr>
      <th>10852</th>
      <td>43853</td>
      <td>Business, The</td>
      <td>[Action, Comedy, Crime, Drama, Thriller]</td>
      <td>2005</td>
    </tr>
    <tr>
      <th>12991</th>
      <td>61553</td>
      <td>Fifth Commandment, The</td>
      <td>[Action, Comedy, Crime, Drama, Thriller]</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>13250</th>
      <td>64645</td>
      <td>The Wrecking Crew</td>
      <td>[Action, Adventure, Comedy, Crime, Drama, Thri...</td>
      <td>1968</td>
    </tr>
    <tr>
      <th>13812</th>
      <td>69136</td>
      <td>Don</td>
      <td>[Action, Comedy, Crime, Drama, Musical, Thriller]</td>
      <td>1978</td>
    </tr>
    <tr>
      <th>14397</th>
      <td>71999</td>
      <td>Aelita: The Queen of Mars (Aelita)</td>
      <td>[Action, Adventure, Drama, Fantasy, Romance, S...</td>
      <td>1924</td>
    </tr>
    <tr>
      <th>15001</th>
      <td>75408</td>
      <td>Lupin III: Sweet Lost Night (Rupan Sansei: Swe...</td>
      <td>[Action, Animation, Comedy, Crime, Drama, Myst...</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>15073</th>
      <td>76153</td>
      <td>Lupin III: First Contact (Rupan Sansei: Faasut...</td>
      <td>[Action, Animation, Comedy, Crime, Drama, Myst...</td>
      <td>2002</td>
    </tr>
    <tr>
      <th>16055</th>
      <td>81132</td>
      <td>Rubber</td>
      <td>[Action, Adventure, Comedy, Crime, Drama, Film...</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>16504</th>
      <td>83266</td>
      <td>Kaho Naa... Pyaar Hai</td>
      <td>[Action, Adventure, Comedy, Drama, Mystery, Ro...</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>20694</th>
      <td>101137</td>
      <td>Dead Man Down</td>
      <td>[Action, Crime, Drama, Romance, Thriller]</td>
      <td>2013</td>
    </tr>
    <tr>
      <th>22095</th>
      <td>106078</td>
      <td>Great Texas Dynamite Chase, The</td>
      <td>[Action, Comedy, Crime, Drama, Romance]</td>
      <td>1976</td>
    </tr>
    <tr>
      <th>23329</th>
      <td>110486</td>
      <td>Big Jim McLain</td>
      <td>[Action, Crime, Drama, Romance, Thriller]</td>
      <td>1952</td>
    </tr>
    <tr>
      <th>25821</th>
      <td>120408</td>
      <td>Friday Foster</td>
      <td>[Action, Crime, Drama, Romance, Thriller]</td>
      <td>1975</td>
    </tr>
    <tr>
      <th>26108</th>
      <td>121370</td>
      <td>Happy New Year</td>
      <td>[Action, Comedy, Crime, Drama, Musical, Romance]</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>27514</th>
      <td>127341</td>
      <td>Longshot</td>
      <td>[Action, Comedy, Crime, Drama, Romance, Thriller]</td>
      <td>2001</td>
    </tr>
    <tr>
      <th>29780</th>
      <td>135781</td>
      <td>Saheb Biwi Aur Gangster</td>
      <td>[Action, Crime, Drama, Romance, Thriller]</td>
      <td>2011</td>
    </tr>
    <tr>
      <th>32307</th>
      <td>144338</td>
      <td>Holiday</td>
      <td>[Action, Children, Comedy, Crime, Drama, Romance]</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>33914</th>
      <td>150268</td>
      <td>Dilwale</td>
      <td>[Action, Children, Comedy, Crime, Drama, Romance]</td>
      <td>2015</td>
    </tr>
  </tbody>
</table>
</div>


