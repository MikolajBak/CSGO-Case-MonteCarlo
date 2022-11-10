
The idea for this project stems from a 3kliksphilip YouTube video series Unboxing Millionaire, where in the [latest instalment](https://www.youtube.com/watch?v=bWweCxxl8hs) he set a challenge of guessing the profit/loss of opening CS:GO Recoil cases until 2 gloves are opened. I will attempt to find an approximate solution to this problem by performing Monte Carlo simulations.

# Introduction
There's a number of fact finding activities required to lay-down the groundwork for future analysis, such as finding the exact likelihood a given item is retrieved and the cost that item could be sold for.
## Weapon Quality
Weapon Quality is a big driver in the price of a skin, with a general rule of higher quality skins having a higher price. 

As part of legislation introduced in China in 2016, Perfect World had to publish the likelihoods of each quality of weapons. Here's an [article](https://www.csgo.com.cn/news/gamebroad/20170911/206155.shtml) in Chinese that contains the breakdown. The breakdown can be seen in the table below.

| Rarity | Odds Compared to level below | Overall Probability |
| ---- | ----- | ---- |
| Special Item | 2:5 | 0.25575% |
| Covert | 1:5 | 0.63939% |
| Classified | 1:5 | 3.19693% |
| Restricted | 1:5 | 15.98465% |
| Mil-spec | - | 79.92327% |

Different skins of the same rarity have the same chance of dropping, meaning that in the Recoil case with 7 Mil-spec weapons, we can just divide the chance of a Mil-spec weapon by 7 to get an 11.418% probability of dropping a given skin.

### Stat-Trak
On top of rarity, another driver of skin price is the inclusion of a Stat-Trak counter, which display the number of kills achieved with this given weapon. This will be simple to factor in as there is a flat 10% chance that a given skin is Stat-Trak.

## Weapon Wear
Wear is a slightly more complicated topic to cover. Once again the general rule of thumb is that weapons with less wear are going be have a higher price (with a few exceptions). 

The wear of a weapon is determined by its float. Float is the value between 0-1 that is the opacity of the "wear" layer applied on top of the skin. The float values are then banded into five distinct categories with Factory New being the least damaged, and usually the more expensive. In most cases this discrete level of wear is used to determine the price of a skin, meaning that two skins with different floats, but within the same band, will be valued the same. There are extreme cases where very low or high floats are valued more by collectors, but this is hard to generalise and therefore outside of my scope.

| Wear Level | Float Range | 
| --- | --- |
| Factory New | 0-0.7 |
| Minimal Wear | 0.07-0.15 |
| Field-Tested | 0.15-0.38 |
| Well-Worn | 0.38-0.45 |
| Battle-Scarred | 0.45-1|

### Float Rabbit Hole
The question of quality became a bit of a rabbit hole that led to this [article](https://blog.csgofloat.com/analysis-of-float-value-and-paint-seed-distribution-in-cs-go/)  by user STEP7750 from a blog dedicated to CS:GO skin trading and learning more about the float values. In that article the author uses empirical data to make assumptions about the process of generating the float number. The key finding is that the floats are not uniformly distributed over the whole range of 0-1. 

![[AK-47 Case Hardened Distribution.png]]

When looking at the distribution of float values for an AK-47 Case Hardened we can see that we effectively have 5 uniform distributions that correlate to the bands of floats corresponding to each wear level. Using that information, STEP7750 calculated that the probability of an AK-47 being a given wear level.

| Wear Level | Probability |
| --- | --- |
| Factory New | ~3% |
| Minimal Wear | ~24% |
| Field-Tested | ~33% |
| Well-Worn | ~24% |
| Battle-Scarred | ~16% |

However, not every weapon skin has the same float range of 0-1. The default range in the game is 0.06-0.80, but it can be changed by the skin artist. STEP7750 repeated a similar experiment on Negev Loudmouth which has a float range of 0.14-0.65. 

![[Negev Loudmouth Distribution.png]]

Here we see the same distribution of 5 uniform distributions, but it's squished to fit within the defined range. An important thing to point out is that while we have the 5 distinct distributions, the discreet level of wear is still defined by the table above, which means in the case of Negev Loudmouth a Factory New finish is not possible as 0.07 lays outside of the range of possible floats. 

Using this information it can be deduced that the initial float is generated using the table of probabilities above, using the standard wear bands, and then the final float is calculated using min and max floats as follows:

```
final_float = float * (max_float - min_float) + min_float
```


## Cost and Reward

The cost of opening cases in CS:GO has two components, a fixed cost for the key at $2.49 and the cost of the case that varies based on supply and demand on the steam market place, currently sitting at $0.97. This brings the total cost of "pulling the arm" to $3.46.

Reward in this instance is the value of the opened weapon. As discussed above there are many factors that influence the value and we will have to simulate randomising all of these factors. I will use the website [csgostash.com](https://csgostash.com/) to collect the data about the guns available in the Recoil Case, mainly the value at each wear level (and Stat-Trak), and the range of possible floats to inform randomisation of wear. 

# Methodology

For the implementation of this simulation I will be using Python notebooks along with some standard data analysis libraries like pandas and numpy. 

A single run of the simulation will consist of a number of steps to calculate the profit. 
1. Randomly select a skin
2. Decide if Stat-Trak is applied
3. Calculate the float using 5 uniform distributions
4. Return the value of the item
5. Repeat until 2 gloves are obtained

## Data prep

But firstly we need some data. I'm sure there is a much better and smarter way of getting this data, but my brain wasn't at 100%, so I just copied the value of guns from csgostash into a csv and noted down their float ranges. The value of a gun was determined as the current lowest sell offer on the Steam market place. The drawback of this approach, other than being slow, is that it's stuck in time and this information is quite likely to fluctuate and the sell price might not really reflect the value. A more general problem with using csgostash is that the higher value items (Factory New gloves) were not traded very often on Steam, which made getting the exact value challenging and in many cases I just had to guess.

![[Value CSV.png]]

After that was done, I loaded the data onto a notebook on Google Colab and prepared the data for the simulation. The first step was to add the drop chance based on the information from the Perfect World blog post, making sure all the probabilities added up to 1. 

![[Python_case_df.png]]

Then the last step of data preparation was creation of a data frame which contained the wear levels and the associated float bands and probabilities. Since that was quite a small data frame, I was happy to just hard code it with the values quoted earlier.

```python
d = {'Wear': ['FN', 'MW', 'FT', 'WW', 'BS'], 
	 "MinFloat": [0, 0.07, 0.15, 0.38, 0.45],
	 "MaxFloat": [0.07, 0.15, 0.38, 0.45, 1], 
	 "Prob": [0.03, 0.24, 0.33, 0.24, 0.16]}

rarity_df = pd.DataFrame(d)
```

## Opening a Single Case

The first decision we need to make when a case is "opened" is what skin to return. This can be done quite easily using the probabilities added to each row of data in the data preparation step. Numpy `choice` function allows us to select an item from a list using a given probability density. This returns an index of the item in our skin data frame.

```python
# Draw a random item
item = df.iloc[choice(list(range(df.shape[0])), 1, p=df['Drop_Chance'])]
```

The next step doesn't need to be performed right now, but we can decide if the item we opened has Stat-Trak. We know that gloves can't have Stat-Trak, and all Special items are gloves, so by default we say all Special items are not Stat-Trak. Otherwise we check if a random uniform number between 0 and 1 is less that 0.1. This gives us the 10% chance of unboxing a Stat-Trak weapon.

```python
  # Decide if Stat-Trak
  is_gloves = item.iloc[0]['Rarity'] == 'Special'
  if is_gloves:
    stat_trak = False
  else:
    stat_trak = 0.1 > random.uniform(0,1)
```

Now we move onto deciding the level of wear our gun will have. This is the most challenging part of this process as there are multiple steps to getting the final wear. First step is to randomly choose which wear bin we want to start in. This can be achieved using a similar approach as to picking an item.

```python
# Decide initial quality bin
wear = rarity.iloc[choice(list(range(rarity.shape[0])), 1, p=rarity['Prob'])]
```

Once we know the initial bin, we can get the initial float value by sampling a uniform number within the float bounds of our bin.

```python
# Get float within the bin
minfloat = wear.iloc[0]['MinFloat']
maxfloat = wear.iloc[0]['MaxFloat']
initial_float = random.uniform(minfloat, maxfloat)
```

We're nearly there, but currently the float value is in the 0 to 1 range. Not all weapons in this case use this range, so we need to map the initial float into the range specified by the gun using the formula mentioned in the previous section. After we calculate the final float, we can change this back to a wear level using the wear data frame

```python
# Convert float to final float and get wear
minfloat_item = item.iloc[0]['Min_Float']
maxfloat_item = item.iloc[0]['Max_Float']
final_float = initial_float * (maxfloat_item - minfloat_item) + minfloat_item

selected_wear = rarity.loc[(rarity['MinFloat'] < final_float) 
						   & (rarity['MaxFloat'] > final_float), 'Wear'].item()
```

Last step is to construct the column name using the information we just generated to let us query the item data frame for the value of our gun. 

```python
# Return reward from given data frame
cost_col = 'Cost_' + selected_wear
if stat_trak:
  cost_col += '_ST'

reward = item.iloc[0][cost_col]

#return final_float, item, selected_wear, cost_col, reward
return reward, int(is_gloves), item.Skin.item()
```

This function will return the reward, indication if the item was gloves and the name of the item. In this example we can see that the opened weapon was a P250 Visions worth $7.95 and unsurprisingly it is not gloves.

```python
get_reward(case_df, rarity_df)
------
(7.95, 0, 'P250 | Visions')
```

## Simulation

As the objective of this exercise is to simulate the profit of opening Recoil Cases until 2 sets of gloves are opened, we need a loop that will keep opening cases until that condition is met. This can be achieved quite easily using a while loop. For a single run of the simulation we want to return the profit, i.e. total reward - total cost. 

```python
def single_run(unit_cost):
  gloves_count = 0
  iters = 0
  total_reward = 0

  while gloves_count < 2:
    iters += 1
    reward, gloves, _ = get_reward(case_df, rarity_df)
    total_reward += reward
    gloves_count += gloves

  return total_reward - (iters * unit_cost)
```

Then to repeat the experiment multiple times we just need to call this function `n` number of times and start looking at results. Quite a useful package for this is [tqdm](https://tqdm.github.io/) that displays a nice progress bar when executing code in loops. It even gives time remaining estimates based on history. 

```python
from tqdm.notebook import tqdm

for i in tqdm(range(90)):
  results.append(single_run(unit_cost=3.46))

res = pd.DataFrame({'Results': results})
res.Results.mean()
```

![[Progress bar.png]]
At this point I have run the simulation for 200 runs (in multiple batches) and the average profit is -$751.68. Another interesting way of looking at this data is to plot it on a histogram. This gives us the frequency density of different levels of profit. 

```python
plt.hist(res.Results, bins=70)
plt.gca().set(title='Profit Histogram', ylabel='Frequency')
```


![[Histogram.png]]
As seen here, there are some instances where getting two sets of gloves is profitable, however the vast majority of simulations resulted in a net loss, with the biggest loss totalling -$4875.88.