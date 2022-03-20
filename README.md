# snowisoDataMunging
This a repository for Python code that performs an initial processing and analysis of data from the SNOWISO project at EastGRIP. 

See https://steenlarsen.w.uib.no/erc-stg-snowiso/ for a summary of the project.

A bit about the data and what is performed here by the different streams of code 
The snow was collected at EastGRIP from 2016-2019. The snow was then processed for dD and d18O at two different sites (Germany at AWI, Iceland).

The code here cleans and merges the data from the two different institutions into a single data frame. It then processes the metadata from the field and includes important information into a second data frame. Select fields from the second data frame is included in the primary data frame with the isotope data.

Initial QC, EDA, error correlations, and a look at dxs patterns are included in this analysis.
