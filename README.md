# Predicting NHL Injuries

I attempted to build a model capable of predicting how many games an NHL player will miss in a season due to injury. I’m a huge fan of both hockey and its associated analytics movement, so I wanted to work with NHL data but I also wanted to find a relatively uncovered angle. I also wanted to put some of the conventional wisdom of the league to the test:
* Do smaller/lighter players get injured more?
* Is being injury-prone a real thing?
* Are older players more likely to get hurt?
* Are players from certain countries more or less likely to get injured? (ie, "soft" Europeans and Alex Ovechkin's "Russian machine does not break")

### Data Collection

Counting stats for sports are easy to come by and hockey is no different - I was able to download CSVs of all player stats and biometrics for the last ten NHL seasons from ![Natural Stat Trick]('www.naturalstattrick.com'). I combined the separate datasets in ![this notebook]('notebook.ipynb'). Unfortunately reliable player injury histories are much more difficult to come by. I was able to scrape lists of injuries from individual player profiles from ![TSN]('www.tsn.ca/nhl') using Selenium and BeautifulSoup. All injury scraping and parsing is contained in ![this notebook]('notebook.ipynb'). While I don't believe the TSN data is an exhaustive list of player injuries it is the best I could find so that's what I used.

### Feature Selection/Engineering

Mostly due to the amazing amount of counting stats Natural Stat Trick aggregates, I had an abundance of potential features for my models. I mostly used my domain knowledge as a longtime hockey fan to whittle down the list to anything I thought could correlate with injury rates, as well as predictors to test the conventional wisdom of what causes players to get hurt. I also removed goaltenders from my data set because their counting stats are completely different from those of other skaters. One of my models utilized sklearn's polynomial features to account for any feature interaction. Here is a partial list of individual features and my logic for choosing them:
* **Games Missed Due to Injury** what I'm trying to predict
* **Height/Weight** to see if smaller/lighter players get injured more often
* **Position** defensemen play more minutes and are more likely to block shots than other positions, wingers and defensemen are more likely than centers to engage in checking along the boards
* **Penalties Drawn** penalties are often assessed when a player is injured as the result of a dangerous and illegal play
* **Hits Delivered/Hits Taken** an indicator of physical play that could lead to more injuries
* **Shots Blocked** players sometimes suffer contusions and break bones blocking shots
* **Age** to see if older players get injured more often
* **Major Penalties** majors are often assessed for fighting
* **Being European/Being Russian** to see if either correlates with increased injury rates

### Additional Feature Engineering

I started off with some simple models because I wanted to evaluate what formulation of my data would work best before spending my time tweaking hyperparameters. I fit an Ordinary Least Squares regression on each of the following data sets:
* Each entry contains the total counting stats and games missed due to injury for the last ten seasons.
* Each entry contains the counting stats, games missed due to injury, and games missed last season for a single season for a player. In this and the following format, players can have multiple rows of data.
* Similar to the last format except it includes rolling averages of counting stats and games missed for all previous seasons.
Unsurprisingly, the last and most robust data set resulted in the lowest test MSE so I used that formulation of my data for the final modeling.

### Models
