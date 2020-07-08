
Here we show example Web tables and their surrounding natural language context 
in the pre-training corpora. We also include table captions in the NL context, 
as they are high-quality summaries of the table. 
During pre-training, we sample a window of 128 tokens from the NL context as the 
synthetic utterance paired with tables. 

Long context in the examples below is truncated 
by showing only the first few sentences in each paragraph.

### Example

**Context**

```
This is a comparison between U.S. states and sovereign states' Nominal Gross Domestic Product for the Alternative Future as based on International Monetary Fund and Bureau of Economic Analysis data. Many of the states of the United States have large gross domestic product (called gross state product) which would rank highly on a list of countries world GDP.

National GDPs and States GDPs (Table Caption)
```

**Table**

| #  | Country        | GDP (USD million) |
|----|----------------|-------------------|
|    | World          | 84,835,462        |
| 1  | China          | 13,457,267        |
| 2  | Japan          | 5,070,626         |
| 3  | Germany        | 4,029,140         |
| 4  | California     | 3,018,337         |
| 5  | United Kingdom | 2,808,899         |
| 6  | France         | 2,794,696         |
| 7  | India          | 2,689,992         |
| 8  | Italy          | 2,086,911         |
| 9  | Brazil         | 1,909,386         |
| 10 | Texas          | 1,818,585         |

### Example

**Context**

```
This article includes a list of countries and dependent territories sorted by their real gross domestic product growth rate; the rate of growth of the value of all final goods and services produced within a state in a given year. The statistics were compiled from the IMF World Economic Outlook Database with the vast majority of estimates corresponding to the 2018 calendar year. Values from other sources are referenced as such. Rates in bold italic are estimates. Rates in bold italic are estimates.
```

**Table**

| Rank | Country/region      | Real GDP growth rate (%) |
| ---- | ------------------- | ------------------------ |
| 1    | Libya               | 17.9                     |
| 2    | Eritrea             | 12.2                     |
| 3    | Rwanda              | 8.6                      |
| 4    | Ireland             | 8.3                      |
| 5    | Bangladesh          | 7.9                      |
| 6    | Ethiopia            | 7.7                      |
| 7    | Cambodia            | 7.5                      |
| 8    | Maldives            | 7.5                      |
| 9    | Ivory Coast         | 7.4                      |
| 10   | Antigua and Barbuda | 7.4                      |

### Example

**Context**

```
When it comes to Tom Cruise sci-fi movies, 'Oblivion's $38 million opening edges out 'Minority Report' ($35 million) but is still dwarfed by 'War of the Worlds' ($64 million, which, to be fair, also had Steven Spielberg in its corner)...

The weekend was also very kind to '42,' which took a small drop from last week for a weekend gross of $18 million and a $54 million total...

Meanwhile, 'The Croods' held onto the number three spot for the third weekend in a row, grossing $9.5 million for a $154 million total...
```
**Table**

| Film                       | Weekend             | Per Screen |
| -------------------------- | ------------------- | ---------- |
| Oblivion                   | $38,152,000         | $10.09     |
| 42                         | $18,025,000 (-34.4) | $5,546     |
| The Croods                 | $9,500,000 (-27.6)  | $2,766     |
| Scary Movie 5              | $6,296,000 (-55.5)  | $1,851     |
| G.I. Joe: Retaliation      | $5,775,000 (-47.0)  | $1,819     |
| The Place Beyond the Pines | $4,746,000 (+22.8)  | $3,078     |
| Olympus Has Fallen         | $4,500,000 (-37.9)  | $1,706     |
| Evil Dead                  | $4,100,000 (-56.8)  | $1,452     |
| Jurassic Park              | $4,008,000 (-54.8)  | $1,720     |
| Oz the Great and Powerful  | $3,048,000 (-37.3)  | $1,490     |

### Example

**Context**

```
The population development of the places in Brown. Source: US Census Bureau (web). 2000 and 2010 population of incorporated places in the boundaries of 2010.
```

**Table**

| Name           | Status  | County | Population Census 1990-04-01 | Population Census 2000-04-01 | Population Census 2010-04-01 |     |
| -------------- | ------- | ------ | ---------------------------- | ---------------------------- | ---------------------------- | --- |
| Mound Station  | Village | Brown  | 147                          | 124                          | 122                          |
| Mount Sterling | City    | Brown  | 1,994                        | 2,085                        | 2,025                        |
| Ripley         | Village | Brown  | 75                           | 105                          | 86                           |
| Versailles     | Village | Brown  | 480                          | 569                          | 478                          |

### Example

**Context**

```
Directory of Airports in Cook Islands. Runway lengths are based on available landing distance where possible.
```

**Table**

| Kind  | ICAO | IATA | City            | Name                    | Latitude   | Longitude   | Max Runway |
| ----- | ---- | ---- | --------------- | ----------------------- | ---------- | ----------- | ---------- |
| Small | NCAI | AIT  | Aitutaki        | AITUTAKI                | -18.831(S) | -159.764(W) | 5920 ft    |
| Small | NCAT | AIU  | Atiu Island     | Enua Airport            | -19.968(S) | -158.119(W) | 4100 ft    |
| Large | NCRG | RAR  | Avarua          | RAROTONGA INTL          | -21.203(S) | -159.806(W) | 7638 ft    |
| Small | NCMG | MGS  | Mangaia Island  | Mangaia Island Airport  | -21.896(S) | -157.907(W) | 3400 ft    |
| Small | NCMH | MHX  | Manihiki Island | Manihiki Island Airport | -10.377(S) | -161.002(W) | 3900 ft    |
| Small | NCMN |      | Manuae          | Manuae Airport          | -19.267(S) | -158.960(W) |            |
| Small | NCMK | MUK  | Mauke Island    | Mauke Airport           | -20.136(S) | -157.345(W) | 5900 ft    |
| Small | NCMR | MOI  | Mitiaro Island  | Mitiaro Island Airport  | -19.843(S) | -157.703(W) | 3400 ft    |
| Small | NCPY | PYE  | Penrhyn Island  | Tongareva Airport       | -9.014(S)  | -158.032(W) | 7500 ft    |